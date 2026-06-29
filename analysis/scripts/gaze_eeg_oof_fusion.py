"""Late OOF fusion of the gaze-logreg baseline with the multipath_mm EEG decoder.

Both branches produce per-trial, out-of-sample physical-speaker posteriors under
within-subject CV. We fuse them at the trial level (no need for matched folds -- each
trial only needs one OOS posterior from each model) with the fusion weight tuned
OUT-OF-FOLD, then report 4-class + hemisphere accuracy against the baselines.

  gaze branch : baseline_logreg (StandardScaler + L2 logreg C=0.5) on the exact
                fusion_gaze_features.parquet features (the published 0.547 model).
  EEG branch  : multipath_mm per-trial posteriors exported to post_*.parquet in the
                run dir (pass --eeg_run <run_dir>).

Usage:
  python scripts/gaze_eeg_oof_fusion.py --eeg_run /fs/scratch/.../multimodal_aad__multipath_mm-eegonly__<ts>
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RESULTS = "/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/results"
HEMI = np.array([0, 0, 1, 1])        # speakers 1,2 = Left ; 3,4 = Right (physical idx 0..3)
INOUT = np.array([1, 0, 0, 1])       # {2,3} inner ; {1,4} outer
EPS = 1e-6


def gaze_oos_posteriors(seed=0):
    """Per-subject 5-fold OOS posteriors from the baseline gaze logreg."""
    G = pd.read_parquet(os.path.join(RESULTS, "fusion_gaze_features.parquet"))
    cols = [c for c in G.columns if c not in ("subject", "trial", "attended", "group", "snr")]
    rows = []
    for s, g in G.groupby("subject"):
        g = g.dropna(subset=["attended"]).copy()
        X = g[cols].fillna(0).values
        y = g["attended"].values.astype(int)          # 1..4
        if len(g) < 20:
            continue
        kf = KFold(5, shuffle=True, random_state=seed)
        P = np.full((len(g), 4), np.nan)
        for tr, te in kf.split(X):
            pipe = Pipeline([("sc", StandardScaler()),
                             ("c", LogisticRegression(max_iter=3000, C=0.5))])
            pipe.fit(X[tr], y[tr])
            proba = pipe.predict_proba(X[te])          # cols sorted by class label
            for j, cls in enumerate(pipe.classes_):
                P[te, cls - 1] = proba[:, j]
        for i, (_, r) in enumerate(g.iterrows()):
            rows.append(dict(subject=int(s), trial=int(r["trial"]), attended=int(r["attended"]) - 1,
                             g0=P[i, 0], g1=P[i, 1], g2=P[i, 2], g3=P[i, 3]))
    return pd.DataFrame(rows)


def eeg_oos_posteriors(run_dir):
    parts = sorted(glob.glob(os.path.join(run_dir, "models", "post_*.parquet"))) \
        or sorted(glob.glob(os.path.join(run_dir, "post_*.parquet")))
    if not parts:
        raise SystemExit(f"no post_*.parquet under {run_dir}")
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    return df.rename(columns={"p0": "e0", "p1": "e1", "p2": "e2", "p3": "e3"})


def _acc(P, y):
    return float((P.argmax(1) == y).mean())


def _collapse_acc(P, y, m):
    return float((m[P.argmax(1)] == m[y]).mean())


def fuse(gz, eeg, seed=0):
    M = gz.merge(eeg[["subject", "trial", "e0", "e1", "e2", "e3"]], on=["subject", "trial"])
    y = M["attended"].values.astype(int)
    G = M[["g0", "g1", "g2", "g3"]].values
    E = M[["e0", "e1", "e2", "e3"]].values
    lG, lE = np.log(G + EPS), np.log(E + EPS)
    n = len(M)
    print(f"matched trials: {n}  (subjects {M.subject.nunique()})")

    # single-branch (OOS) accuracy
    print(f"\n  gaze-only : 4c={_acc(G, y):.3f}  hemi={_collapse_acc(G, y, HEMI):.3f}  io={_collapse_acc(G, y, INOUT):.3f}")
    print(f"  eeg-only  : 4c={_acc(E, y):.3f}  hemi={_collapse_acc(E, y, HEMI):.3f}  io={_collapse_acc(E, y, INOUT):.3f}")

    # fusion with alpha tuned OUT-OF-FOLD (log-linear) + a logistic stacker
    kf = KFold(5, shuffle=True, random_state=seed)
    grid = np.linspace(0, 1, 21)
    fa_pred = np.zeros(n, int)
    st_pred = np.zeros(n, int)
    alphas = []
    for tr, te in kf.split(G):
        # alpha: maximise fused 4-class acc on the tuning fold
        best_a, best = 0.5, -1
        for a in grid:
            f = a * lE[tr] + (1 - a) * lG[tr]
            acc = (f.argmax(1) == y[tr]).mean()
            if acc > best:
                best, best_a = acc, a
        alphas.append(best_a)
        fa_pred[te] = (best_a * lE[te] + (1 - best_a) * lG[te]).argmax(1)
        # logistic stacker on the 8 log-posteriors
        st = LogisticRegression(max_iter=3000, C=1.0)
        st.fit(np.hstack([lG[tr], lE[tr]]), y[tr])
        st_pred[te] = st.predict(np.hstack([lG[te], lE[te]]))

    def rep(name, pred):
        print(f"  {name:11s}: 4c={(pred==y).mean():.3f}  "
              f"hemi={(HEMI[pred]==HEMI[y]).mean():.3f}  io={(INOUT[pred]==INOUT[y]).mean():.3f}")
    print(f"\n  alpha (eeg weight) per fold: {[round(a,2) for a in alphas]}")
    rep("FUSE-alpha", fa_pred)
    rep("FUSE-stack", st_pred)
    print("\n  baselines (trial/30s): gaze 0.547 / hemi 0.773 | EEG-spectral 0.448 / 0.716")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg_run", required=True, help="multipath_mm run dir with post_*.parquet")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    fuse(gaze_oos_posteriors(a.seed), eeg_oos_posteriors(a.eeg_run), a.seed)
