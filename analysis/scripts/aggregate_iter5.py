"""Aggregate iter-5 spectral-feature EEG classifier results + fusion with gaze."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb

from aad_utils import RESULTS_DIR, FIGURES_DIR, set_pub_style, save_fig, bootstrap_ci, COLORS
from aad_utils.config import ATTENDED_HEMISPHERE
set_pub_style()

SPEC_DIR = RESULTS_DIR / "eeg_spectral"
files = sorted(SPEC_DIR.glob("s*.parquet"))
files = [f for f in files if ".features.parquet" not in f.name]
print(f"Found {len(files)} result files")
if not files:
    raise SystemExit("no iter-5 results")

dfs = [pd.read_parquet(f) for f in files]
R = pd.concat(dfs, ignore_index=True)
print("\n=== Pooled EEG spectral-feature accuracy (per task × classifier) ===")
for task, g in R.groupby("task"):
    ch = g["chance"].iloc[0]
    for clf, gg in g.groupby("classifier"):
        m, lo, hi = bootstrap_ci(gg["acc"].values)
        print(f"  {task:<12} {clf:<10} pooled={m:.3f} [{lo:.3f},{hi:.3f}] chance={ch:.3f}")

# Per-subject, hemisphere (logreg) — the flagship number.
pivot = R[R["classifier"]=="logreg"].pivot(index="subject", columns="task", values="acc")
print("\n=== Per-subject logreg accuracy ===")
print(pivot.round(3).to_string())

# --- Fusion with gaze ---
print("\n=== Fusion: gaze features + EEG spectral features ===")
feat_files = sorted(SPEC_DIR.glob("s*.features.parquet"))
F_all = pd.concat([pd.read_parquet(f) for f in feat_files], ignore_index=True)
G_all = pd.read_parquet(RESULTS_DIR / "fusion_gaze_features.parquet")
merged = F_all.merge(G_all, on=["subject","trial","attended"], suffixes=("_eeg", "_gz"))
print(f"Merged rows: {len(merged)}  (eeg×gaze intersection)")

eeg_cols = [c for c in F_all.columns if c not in ("subject","trial","attended","snr")]
gaze_cols = [c for c in G_all.columns if c not in ("subject","trial","attended","group","snr")]

def eval_fusion_task(lbl_fn, task, nc):
    rows = []
    for s, g in merged.groupby("subject"):
        if len(g) < 30: continue
        y = g["attended"].apply(lbl_fn).values
        if len(np.unique(y)) < nc or pd.Series(y).value_counts().min() < 2: continue
        Xe = g[eeg_cols].fillna(0).values
        Xg = g[gaze_cols].fillna(0).values
        Xc = np.concatenate([Xe, Xg], axis=1)
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        acc_e, acc_g, acc_c = [], [], []
        for tr,te in skf.split(Xc, y):
            pipe_e = Pipeline([("sc",StandardScaler()),("c",LogisticRegression(max_iter=3000,C=0.5))])
            pipe_g = Pipeline([("sc",StandardScaler()),("c",LogisticRegression(max_iter=3000,C=0.5))])
            pipe_c = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, verbosity=-1)
            pipe_e.fit(Xe[tr], y[tr]); acc_e.append(accuracy_score(y[te], pipe_e.predict(Xe[te])))
            pipe_g.fit(Xg[tr], y[tr]); acc_g.append(accuracy_score(y[te], pipe_g.predict(Xg[te])))
            pipe_c.fit(Xc[tr], y[tr]); acc_c.append(accuracy_score(y[te], pipe_c.predict(Xc[te])))
        rows.append(dict(subject=s, task=task, chance=1/nc,
                         acc_eeg_spec=float(np.mean(acc_e)),
                         acc_gaze=float(np.mean(acc_g)),
                         acc_early_fusion=float(np.mean(acc_c))))
    return pd.DataFrame(rows)

tasks = [
    ("hemisphere",  lambda a: 0 if ATTENDED_HEMISPHERE[a]=="L" else 1, 2),
    ("inner_outer", lambda a: 0 if a in (2,3) else 1, 2),
    ("4class",      lambda a: a-1, 4),
]
fusion_all = []
for t, lbl, nc in tasks:
    rf = eval_fusion_task(lbl, t, nc)
    fusion_all.append(rf)
    print(f"\n--- {t} ---")
    print(rf.round(3).to_string(index=False))
    for c in ("acc_eeg_spec","acc_gaze","acc_early_fusion"):
        m, lo, hi = bootstrap_ci(rf[c].values)
        print(f"  {c:<18} pooled={m:.3f} [{lo:.3f},{hi:.3f}]")
FUS = pd.concat(fusion_all, ignore_index=True)
FUS.to_parquet(RESULTS_DIR / "iter5_fusion.parquet")
R.to_parquet(RESULTS_DIR / "iter5_summary.parquet")
print("\nSaved iter5_summary.parquet, iter5_fusion.parquet")
