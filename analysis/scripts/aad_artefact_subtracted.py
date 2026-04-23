"""Iter-8 · Artefact-subtraction decoder.

For each subject, compute AAD accuracy under five EEG feature variants
(all using the iter-5 368-D spectral feature vector as the base):

    raw                 : baseline, no change (≡ iter-5).
    resid_motion        : within each CV fold, fit Ridge(Z→X) on the
                          training fold only where Z = concatenated
                          gaze+IMU+video features; the residual
                          X - Ẑ is used as input to the AAD classifier.
                          This is the canonical ``orthogonalise EEG
                          against motion'' procedure.
    resid_gaze          : same, but Z = gaze features only.
    resid_imu           : same, but Z = IMU features only.
    shuffle_motion      : shuffle Z across trials (break the
                          feature→trial binding) and then do the same
                          residualisation — a null control. If the
                          residualisation was leaking trial-level
                          information, this shuffle should also hurt
                          accuracy. It shouldn't, because trial-shuffled
                          Z has no relationship to X beyond chance.

Per-subject 5-fold stratified CV, three label collapses
(hemisphere / inner-outer / 4-class). L2-logistic regression.

CLI:
    python aad_artefact_subtracted.py --subject 3 --out results/iter8/s3.parquet
"""
from __future__ import annotations
import argparse, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from aad_utils import RESULTS_DIR, list_subjects, load_trials_csv
from aad_utils.config import ATTENDED_HEMISPHERE


def load_feature_tables(subject: int):
    """Load iter-5 EEG spectral features, iter-3 gaze features,
    iter-6 IMU & video features for one subject, aligned on trial.
    Returns (trials_df, EEG_cols, GAZE_cols, IMU_cols, VID_cols)."""
    eeg_p = RESULTS_DIR / "eeg_spectral" / f"s{subject}.features.parquet"
    if not eeg_p.exists(): return None
    E = pd.read_parquet(eeg_p)

    G = pd.read_parquet(RESULTS_DIR / "fusion_gaze_features.parquet")
    G = G[G["subject"] == subject]

    I = pd.read_parquet(RESULTS_DIR / "imu_aad" / f"s{subject}.features.parquet") \
        if (RESULTS_DIR/"imu_aad"/f"s{subject}.features.parquet").exists() else None
    V = pd.read_parquet(RESULTS_DIR / "video_aad" / f"s{subject}.features.parquet") \
        if (RESULTS_DIR/"video_aad"/f"s{subject}.features.parquet").exists() else None

    E_cols = [c for c in E.columns if c not in ("subject","trial","attended","snr")]
    G_cols = [c for c in G.columns if c not in ("subject","trial","attended","group","snr")]
    I_cols = [c for c in I.columns if c not in ("subject","trial","attended","snr")] if I is not None else []
    V_cols = [c for c in V.columns if c not in ("subject","trial","attended","snr","fps","n_frames")] if V is not None else []

    # Inner-join on subject+trial.
    df = E.merge(G, on=["subject","trial","attended"], suffixes=("_e","_g"))
    if I is not None: df = df.merge(I, on=["subject","trial","attended"], suffixes=("","_i"))
    if V is not None: df = df.merge(V, on=["subject","trial","attended"], suffixes=("","_v"))
    # merge may rename columns — reconcile
    def present(cols):
        return [c for c in cols if c in df.columns]
    return df, present(E_cols), present(G_cols), present(I_cols), present(V_cols)


def evaluate(df, E_cols, Z_cols_dict, label_fn, task, nc, residual_mode):
    """Residual AAD with 5-fold CV.
    residual_mode ∈ {"raw", "gaze", "imu", "motion", "shuffle_motion"}."""
    X = df[E_cols].fillna(0).values
    y = np.array([label_fn(a) for a in df["attended"].values])
    if residual_mode == "raw":
        Zcols = []
    elif residual_mode == "gaze":
        Zcols = Z_cols_dict["gaze"]
    elif residual_mode == "imu":
        Zcols = Z_cols_dict["imu"]
    elif residual_mode in ("motion", "shuffle_motion"):
        Zcols = Z_cols_dict["gaze"] + Z_cols_dict["imu"] + Z_cols_dict["video"]
    else:
        raise ValueError(residual_mode)

    Z = df[Zcols].fillna(0).values if Zcols else None
    if residual_mode == "shuffle_motion" and Z is not None:
        rng = np.random.default_rng(123)
        perm = rng.permutation(len(Z))
        Z = Z[perm]

    if len(np.unique(y)) < nc or pd.Series(y).value_counts().min() < 2:
        return np.nan, np.nan
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    accs, n_resid = [], 0
    for tr, te in skf.split(X, y):
        sc_X = StandardScaler().fit(X[tr])
        Xtr = sc_X.transform(X[tr]); Xte = sc_X.transform(X[te])
        if Z is not None:
            sc_Z = StandardScaler().fit(Z[tr])
            Ztr = sc_Z.transform(Z[tr]); Zte = sc_Z.transform(Z[te])
            M = Ridge(alpha=10.0).fit(Ztr, Xtr)
            Xtr = Xtr - M.predict(Ztr)
            Xte = Xte - M.predict(Zte)
            n_resid = Z.shape[1]
        clf = LogisticRegression(max_iter=3000, C=0.5, solver="lbfgs").fit(Xtr, y[tr])
        accs.append(accuracy_score(y[te], clf.predict(Xte)))
    return float(np.mean(accs)), n_resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    t0 = time.time()
    pkg = load_feature_tables(a.subject)
    if pkg is None:
        print(f"[S{a.subject}] missing feature tables"); return
    df, E_cols, G_cols, I_cols, V_cols = pkg
    Z_cols_dict = {"gaze": G_cols, "imu": I_cols, "video": V_cols}
    print(f"[S{a.subject}] {len(df)} trials × EEG {len(E_cols)} / gaze {len(G_cols)} / "
          f"imu {len(I_cols)} / video {len(V_cols)}", flush=True)

    tasks = [
        ("hemisphere",  lambda a_: 0 if ATTENDED_HEMISPHERE[a_]=="L" else 1, 2),
        ("inner_outer", lambda a_: 0 if a_ in (2,3) else 1, 2),
        ("4class",      lambda a_: a_-1, 4),
    ]
    modes = ["raw", "gaze", "imu", "motion", "shuffle_motion"]
    rows = []
    for task, lbl, nc in tasks:
        for mode in modes:
            acc, nz = evaluate(df, E_cols, Z_cols_dict, lbl, task, nc, mode)
            rows.append(dict(subject=a.subject, task=task, mode=mode,
                             chance=1/nc, acc=acc, n_residualised_features=nz))
    R = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    R.to_parquet(a.out)
    print(f"[S{a.subject}] done in {time.time()-t0:.0f}s", flush=True)
    print(R.to_string(index=False))


if __name__ == "__main__":
    main()
