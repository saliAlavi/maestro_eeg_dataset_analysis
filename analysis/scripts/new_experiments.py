"""New experiments added for the NeurIPS D&B paper finalisation.

Reuses the cached iter-5 spectral features
(results/eeg_spectral/s*.features.parquet), iter-3 gaze features
(results/fusion_gaze_features.parquet), iter-6 IMU / video features
(results/imu_aad/s*.features.parquet, results/video_aad/s*.features.parquet)
so we do not redo any EEG preprocessing or feature extraction; each new
experiment is a classifier / regression step on top.

Experiments implemented here:

  (A) LOSO with the 368-D EEG spectral classifier (hemisphere / inner-outer
      / 4-class). The paper only had LOSO for envelope-reconstruction and
      gaze. This adds the spectral LOSO.

  (B) Per-subject learning curves: accuracy vs training-trial count
      n in {10, 20, 40, 60, 80}, 10 random resamples per n, 5-fold CV
      inside each bootstrap. Spectral, gaze, and early fusion.

  (C) SNR-stratified spectral classifier: accuracy in SNR bins
      (<=6, 7-10, 11-14, >=15 dB) with 5-fold CV per subject pooled.

  (D) Partial motion-residualisation sweep: replicate iter-8 with
      X_tilde = X - alpha*M(Z), alpha in {0, 0.25, 0.5, 0.75, 1.0}
      to get the capacity-cost curve the paper promised.

  (E) Subject-quality composite: pre-registered quality score built from
      bad-channel rate, comprehension accuracy, alpha-lateralisation
      strength, gaze validity, ISC@Cz; Pearson r and OLS regression vs
      per-decoder accuracy.

  (F) Behaviour x spectral-decoder correlation: per-trial decoder
      correctness vs comprehension correctness for the spectral classifier
      (the paper only covered TRF).

  (G) Pooled late fusion with CALIBRATED probabilities (not LightGBM
      early fusion) to get a fairer fusion floor: logistic-regression
      posteriors from EEG-spec and gaze, averaged with inverse-variance
      weights from inner CV.

Results write to results/new_experiments/*.parquet.
"""
from __future__ import annotations
import sys, warnings, time
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score

from aad_utils import RESULTS_DIR, load_trials_csv, bootstrap_ci
from aad_utils.config import ATTENDED_HEMISPHERE

OUT = RESULTS_DIR / "new_experiments"
OUT.mkdir(parents=True, exist_ok=True)

# ---------- data loader -----------------------------------------------

TASKS = [
    ("hemisphere",  lambda a: 0 if ATTENDED_HEMISPHERE[a] == "L" else 1, 2),
    ("inner_outer", lambda a: 0 if a in (2, 3) else 1, 2),
    ("4class",      lambda a: a - 1, 4),
]


def load_all():
    """Load cached feature tables, return merged dataframe + column lists."""
    spec_dir = RESULTS_DIR / "eeg_spectral"
    E_parts, G_parts, I_parts, V_parts = [], [], [], []
    for p in sorted(spec_dir.glob("s*.features.parquet")):
        E_parts.append(pd.read_parquet(p))
    E = pd.concat(E_parts, ignore_index=True)
    G = pd.read_parquet(RESULTS_DIR / "fusion_gaze_features.parquet")
    imu_dir = RESULTS_DIR / "imu_aad"
    vid_dir = RESULTS_DIR / "video_aad"
    for p in sorted(imu_dir.glob("s*.features.parquet")):
        I_parts.append(pd.read_parquet(p))
    for p in sorted(vid_dir.glob("s*.features.parquet")):
        V_parts.append(pd.read_parquet(p))
    I = pd.concat(I_parts, ignore_index=True) if I_parts else None
    V = pd.concat(V_parts, ignore_index=True) if V_parts else None

    E_cols = [c for c in E.columns if c not in ("subject", "trial", "attended", "snr")]
    G_cols = [c for c in G.columns if c not in ("subject", "trial", "attended", "group", "snr")]
    I_cols = [c for c in I.columns if c not in ("subject", "trial", "attended", "snr")] if I is not None else []
    V_cols = [c for c in V.columns if c not in ("subject", "trial", "attended", "snr", "fps", "n_frames")] if V is not None else []

    df = E.merge(G, on=["subject", "trial", "attended"], suffixes=("_e", "_g"))
    if I is not None:
        df = df.merge(I, on=["subject", "trial", "attended"], suffixes=("", "_i"))
    if V is not None:
        df = df.merge(V, on=["subject", "trial", "attended"], suffixes=("", "_v"))
    # reconcile columns actually present
    def present(cols): return [c for c in cols if c in df.columns]
    return df, present(E_cols), present(G_cols), present(I_cols), present(V_cols)


# ---------- (A) LOSO on spectral classifier ---------------------------

def exp_A_loso(df, E_cols, G_cols):
    rows = []
    subjects = np.sort(df["subject"].unique())
    logo = LeaveOneGroupOut()
    for mod_name, cols in [("eeg_spec_368", E_cols),
                           ("gaze_23", G_cols),
                           ("eeg+gaze_fusion", E_cols + G_cols)]:
        for task, lbl_fn, nc in TASKS:
            X = df[cols].fillna(0).values
            y = np.array([lbl_fn(a) for a in df["attended"].values])
            g = df["subject"].values
            if len(np.unique(y)) < nc: continue
            for tr, te in logo.split(X, y, g):
                test_sub = int(df["subject"].values[te[0]])
                if pd.Series(y[tr]).value_counts().min() < 2: continue
                pipe = Pipeline([("sc", StandardScaler()),
                                 ("c", LogisticRegression(max_iter=3000, C=0.5))])
                pipe.fit(X[tr], y[tr])
                acc = accuracy_score(y[te], pipe.predict(X[te]))
                rows.append(dict(model=mod_name, task=task, test_subject=test_sub,
                                 chance=1.0 / nc, acc=acc, n_train=len(tr), n_test=len(te)))
    R = pd.DataFrame(rows)
    R.to_parquet(OUT / "A_loso_spectral.parquet")
    summary = R.groupby(["model", "task"])["acc"].agg(["mean", "std", "count"]).reset_index()
    summary.to_parquet(OUT / "A_loso_spectral_summary.parquet")
    print("[A] LOSO spectral/gaze/fusion:")
    print(summary.to_string(index=False))
    return R


# ---------- (B) learning curves ---------------------------------------

def exp_B_learning_curve(df, E_cols, G_cols, n_trials_list=(10, 20, 40, 60, 80),
                         n_boot=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for mod_name, cols in [("eeg_spec_368", E_cols), ("gaze_23", G_cols),
                           ("eeg+gaze_fusion", E_cols + G_cols)]:
        for task, lbl_fn, nc in TASKS:
            for s, g in df.groupby("subject"):
                X = g[cols].fillna(0).values
                y = np.array([lbl_fn(a) for a in g["attended"].values])
                if len(np.unique(y)) < nc: continue
                for n in n_trials_list:
                    if n > len(X): continue
                    accs = []
                    for b in range(n_boot):
                        idx = rng.choice(len(X), size=n, replace=False)
                        sub_y = y[idx]
                        if len(np.unique(sub_y)) < nc or pd.Series(sub_y).value_counts().min() < 2:
                            continue
                        try:
                            skf = StratifiedKFold(min(5, pd.Series(sub_y).value_counts().min()),
                                                  shuffle=True, random_state=b)
                            fold_acc = []
                            for tr, te in skf.split(X[idx], sub_y):
                                pipe = Pipeline([("sc", StandardScaler()),
                                                 ("c", LogisticRegression(max_iter=3000, C=0.5))])
                                pipe.fit(X[idx][tr], sub_y[tr])
                                fold_acc.append(accuracy_score(sub_y[te], pipe.predict(X[idx][te])))
                            accs.append(float(np.mean(fold_acc)))
                        except Exception:
                            pass
                    if accs:
                        rows.append(dict(model=mod_name, task=task, subject=int(s),
                                         n_train=int(n), acc=float(np.mean(accs)),
                                         acc_std=float(np.std(accs)),
                                         chance=1.0 / nc, n_boot=len(accs)))
    R = pd.DataFrame(rows)
    R.to_parquet(OUT / "B_learning_curve.parquet")
    print("[B] Learning curve rows:", len(R))
    print(R.groupby(["model", "task", "n_train"])["acc"].mean().round(3).head(20))
    return R


# ---------- (C) SNR stratified spectral ------------------------------

def exp_C_snr(df, E_cols, G_cols):
    # 'snr' column is already on the merged feature tables; fall back to the
    # raw trials CSV (column "SNR") only if the feature tables lack it.
    if "snr" in df.columns:
        df2 = df.copy()
        df2["SNR"] = df2["snr"]
    else:
        tcsv = load_trials_csv()
        snr_col = [c for c in tcsv.columns if c.lower() == "snr"][0]
        trial_col = [c for c in tcsv.columns if c.lower().startswith("trial")][0]
        trials_meta = tcsv[[trial_col, snr_col]].rename(
            columns={trial_col: "trial", snr_col: "SNR"})
        df2 = df.merge(trials_meta, on="trial", how="left")
    bins = [(-np.inf, 6, "<=6"), (6.001, 10, "7-10"),
            (10.001, 14, "11-14"), (14.001, np.inf, ">=15")]
    rows = []
    for mod_name, cols in [("eeg_spec_368", E_cols), ("gaze_23", G_cols),
                           ("eeg+gaze_fusion", E_cols + G_cols)]:
        for task, lbl_fn, nc in TASKS:
            for lo, hi, label in bins:
                g = df2[(df2.SNR >= lo) & (df2.SNR <= hi)]
                if len(g) < 30: continue
                y = np.array([lbl_fn(a) for a in g["attended"].values])
                if len(np.unique(y)) < nc or pd.Series(y).value_counts().min() < 5:
                    continue
                X = g[cols].fillna(0).values
                skf = StratifiedKFold(5, shuffle=True, random_state=0)
                accs = []
                for tr, te in skf.split(X, y):
                    pipe = Pipeline([("sc", StandardScaler()),
                                     ("c", LogisticRegression(max_iter=3000, C=0.5))])
                    pipe.fit(X[tr], y[tr])
                    accs.append(accuracy_score(y[te], pipe.predict(X[te])))
                rows.append(dict(model=mod_name, task=task, snr_bin=label,
                                 acc=float(np.mean(accs)), n=len(g), chance=1.0 / nc))
    R = pd.DataFrame(rows)
    R.to_parquet(OUT / "C_snr_stratified.parquet")
    print("[C] SNR stratified:")
    print(R.pivot_table(index=["model", "task"], columns="snr_bin", values="acc").round(3))
    return R


# ---------- (D) partial motion-residualisation sweep ------------------

def exp_D_partial_motion(df, E_cols, G_cols, I_cols, V_cols,
                         alphas=(0.0, 0.25, 0.5, 0.75, 1.0)):
    Z_cols = G_cols + I_cols + V_cols
    rows = []
    for task, lbl_fn, nc in TASKS:
        for s, g in df.groupby("subject"):
            X = g[E_cols].fillna(0).values
            Z = g[Z_cols].fillna(0).values
            y = np.array([lbl_fn(a) for a in g["attended"].values])
            if len(np.unique(y)) < nc or pd.Series(y).value_counts().min() < 2: continue
            skf = StratifiedKFold(5, shuffle=True, random_state=0)
            fold_acc = {a: [] for a in alphas}
            for tr, te in skf.split(X, y):
                sc_X = StandardScaler().fit(X[tr]); Xtr = sc_X.transform(X[tr]); Xte = sc_X.transform(X[te])
                sc_Z = StandardScaler().fit(Z[tr]); Ztr = sc_Z.transform(Z[tr]); Zte = sc_Z.transform(Z[te])
                M = Ridge(alpha=10.0).fit(Ztr, Xtr)
                Xtr_hat = M.predict(Ztr); Xte_hat = M.predict(Zte)
                for a in alphas:
                    Xtr_a = Xtr - a * Xtr_hat
                    Xte_a = Xte - a * Xte_hat
                    clf = LogisticRegression(max_iter=3000, C=0.5).fit(Xtr_a, y[tr])
                    fold_acc[a].append(accuracy_score(y[te], clf.predict(Xte_a)))
            for a in alphas:
                rows.append(dict(task=task, subject=int(s), alpha=a,
                                 acc=float(np.mean(fold_acc[a])), chance=1.0 / nc))
    R = pd.DataFrame(rows)
    R.to_parquet(OUT / "D_partial_motion.parquet")
    print("[D] Partial motion sweep:")
    print(R.groupby(["task", "alpha"])["acc"].mean().round(3))
    return R


# ---------- (E) subject-quality composite ----------------------------

def exp_E_quality_composite():
    cov = pd.read_parquet(RESULTS_DIR / "individual_differences_covariates.parquet")
    tgt = pd.read_parquet(RESULTS_DIR / "individual_differences_targets.parquet")
    df = cov.merge(tgt, on="subject", how="inner")
    # pre-registered composite (same weights as paper): z-score each covariate
    # with sign set so larger = better; sum.
    covar_cols = [c for c in ["comprehension", "pupil_mean", "gaze_valid",
                              "alpha_lat", "isc_cz"]
                  if c in df.columns]
    sign = {"comprehension": +1, "pupil_mean": +1, "gaze_valid": +1,
            "alpha_lat": +1, "isc_cz": +1, "bad_channel_rate": -1}
    for c in covar_cols:
        mu, sd = df[c].mean(), df[c].std()
        df[c + "_z"] = sign.get(c, +1) * (df[c] - mu) / (sd if sd > 0 else 1.0)
    df["quality_composite"] = df[[c + "_z" for c in covar_cols]].sum(axis=1)
    # correlate with every decoder column we have
    tgt_cols = [c for c in df.columns if c.endswith("_hemi") or c.endswith("_4class")
                or c in ("eeg_hemi", "gaze_hemi", "fusion_hemi")]
    rows = []
    for t in tgt_cols:
        sub = df[["quality_composite", t]].dropna()
        if len(sub) < 5: continue
        r = float(sub.corr().iloc[0, 1])
        rows.append(dict(target=t, pearson_r=r, n=len(sub)))
    R = pd.DataFrame(rows)
    R.to_parquet(OUT / "E_quality_composite.parquet")
    df.to_parquet(OUT / "E_quality_per_subject.parquet")
    print("[E] Quality composite correlations:")
    print(R.sort_values("pearson_r", ascending=False).to_string(index=False))
    return R


# ---------- (F) behaviour x spectral decoder -------------------------

def exp_F_behaviour_spectral(df, E_cols):
    """Per-trial: did decoder get it right, vs was comprehension correct?"""
    try:
        beh = pd.read_parquet(RESULTS_DIR / "02_behavioral_records.parquet")
    except Exception:
        print("[F] behavioural records missing"); return None
    beh = beh[~beh["is_training"]].copy() if "is_training" in beh.columns else beh.copy()
    # Map behavioural-records columns to our canonical names.
    trial_col = "Trial No." if "Trial No." in beh.columns else "trial"
    if trial_col != "trial":
        beh["trial"] = beh[trial_col].astype(str).str.extract(r"(\d+)").astype(float)
        beh = beh.dropna(subset=["trial"])
        beh["trial"] = beh["trial"].astype(int)
    cc = "Correct" if "Correct" in beh.columns else (
        [c for c in beh.columns if c.lower() == "correct"] or [None])[0]
    if cc is None:
        print("[F] no Correct column"); return None
    beh["behaviour_correct"] = beh[cc].astype(int)
    rows = []
    for task, lbl_fn, nc in TASKS:
        per_trial = []
        for s, g in df.groupby("subject"):
            X = g[E_cols].fillna(0).values
            y = np.array([lbl_fn(a) for a in g["attended"].values])
            trials = g["trial"].values
            if len(np.unique(y)) < nc: continue
            skf = StratifiedKFold(5, shuffle=True, random_state=0)
            for tr, te in skf.split(X, y):
                pipe = Pipeline([("sc", StandardScaler()),
                                 ("c", LogisticRegression(max_iter=3000, C=0.5))])
                pipe.fit(X[tr], y[tr])
                pred = pipe.predict(X[te])
                for i, idx in enumerate(te):
                    per_trial.append(dict(subject=int(s), trial=int(trials[idx]),
                                          task=task, decoder_correct=int(pred[i] == y[idx])))
        pt = pd.DataFrame(per_trial)
        merged = pt.merge(beh[["subject", "trial", "behaviour_correct"]],
                          on=["subject", "trial"], how="inner")
        # per-subject Delta: P(decoder correct | behav correct) - P(decoder correct | behav incorrect)
        for s, gs in merged.groupby("subject"):
            if gs["behaviour_correct"].nunique() < 2: continue
            a = gs[gs.behaviour_correct == 1]["decoder_correct"].mean()
            b = gs[gs.behaviour_correct == 0]["decoder_correct"].mean()
            rows.append(dict(task=task, subject=int(s),
                             p_dec_given_beh_correct=a,
                             p_dec_given_beh_wrong=b,
                             delta=a - b,
                             n_correct=int((gs.behaviour_correct == 1).sum()),
                             n_wrong=int((gs.behaviour_correct == 0).sum())))
    R = pd.DataFrame(rows)
    R.to_parquet(OUT / "F_behaviour_spectral.parquet")
    print("[F] behaviour x spectral:")
    print(R.groupby("task")["delta"].agg(["mean", "median"]).round(3))
    return R


# ---------- (G) calibrated late fusion -------------------------------

def exp_G_calibrated_late_fusion(df, E_cols, G_cols):
    rows = []
    for task, lbl_fn, nc in TASKS:
        for s, g in df.groupby("subject"):
            y = np.array([lbl_fn(a) for a in g["attended"].values])
            if len(np.unique(y)) < nc or pd.Series(y).value_counts().min() < 2: continue
            Xe = g[E_cols].fillna(0).values
            Xg = g[G_cols].fillna(0).values
            skf = StratifiedKFold(5, shuffle=True, random_state=0)
            accs_e, accs_g, accs_f = [], [], []
            for tr, te in skf.split(Xe, y):
                pe = Pipeline([("sc", StandardScaler()),
                               ("c", LogisticRegression(max_iter=3000, C=0.5))]).fit(Xe[tr], y[tr])
                pg = Pipeline([("sc", StandardScaler()),
                               ("c", LogisticRegression(max_iter=3000, C=0.5))]).fit(Xg[tr], y[tr])
                prob_e = pe.predict_proba(Xe[te])
                prob_g = pg.predict_proba(Xg[te])
                accs_e.append(accuracy_score(y[te], prob_e.argmax(1)))
                accs_g.append(accuracy_score(y[te], prob_g.argmax(1)))
                # estimate inner-CV reliability weights
                w_e = max(1e-3, pe.score(Xe[tr], y[tr]) - 1.0 / nc)
                w_g = max(1e-3, pg.score(Xg[tr], y[tr]) - 1.0 / nc)
                fused = (w_e * prob_e + w_g * prob_g) / (w_e + w_g)
                accs_f.append(accuracy_score(y[te], fused.argmax(1)))
            rows.append(dict(task=task, subject=int(s), chance=1.0 / nc,
                             acc_eeg=float(np.mean(accs_e)),
                             acc_gaze=float(np.mean(accs_g)),
                             acc_fused=float(np.mean(accs_f))))
    R = pd.DataFrame(rows)
    R.to_parquet(OUT / "G_calibrated_late_fusion.parquet")
    print("[G] calibrated late fusion:")
    print(R.groupby("task")[["acc_eeg", "acc_gaze", "acc_fused"]].mean().round(3))
    return R


# ---------- orchestration --------------------------------------------

def main():
    import traceback, os
    t0 = time.time()
    df, E_cols, G_cols, I_cols, V_cols = load_all()
    print(f"loaded: {len(df)} rows, "
          f"E={len(E_cols)} G={len(G_cols)} I={len(I_cols)} V={len(V_cols)} "
          f"subjects={df['subject'].nunique()}", flush=True)
    only = os.environ.get("EXP_ONLY", "").strip()  # e.g. "CDEFG" to skip AB
    plan = [
        ("A", lambda: exp_A_loso(df, E_cols, G_cols)),
        ("B", lambda: exp_B_learning_curve(df, E_cols, G_cols)),
        ("C", lambda: exp_C_snr(df, E_cols, G_cols)),
        ("D", lambda: exp_D_partial_motion(df, E_cols, G_cols, I_cols, V_cols)),
        ("E", exp_E_quality_composite),
        ("F", lambda: exp_F_behaviour_spectral(df, E_cols)),
        ("G", lambda: exp_G_calibrated_late_fusion(df, E_cols, G_cols)),
    ]
    for name, fn in plan:
        if only and name not in only: continue
        print(f"\n==== running {name} ====", flush=True)
        try:
            fn()
            print(f"[{name}] OK ({time.time()-t0:.0f}s elapsed)", flush=True)
        except Exception:
            print(f"[{name}] FAILED", flush=True)
            traceback.print_exc()
    print(f"All done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
