"""Aggregate iter-4 outputs and run extended fusion on top.

Reads:
    results/aad_v4_4class/s*.parquet
    results/predictability/s*.parquet
    results/gaze_residualised/s*.parquet
    results/fusion_gaze_features.parquet (from aad_fusion.py)

Writes:
    results/iter4_summary.parquet
    results/iter4_fusion.parquet
Figures → analysis/figures/iter4_*.pdf
"""
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


def section(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


# --- 1) 4-class EEG decoder ---
section("[1] 4-class EEG decoder (CCA mel-28, full trial)")
fc_files = sorted((RESULTS_DIR / "aad_v4_4class").glob("s*.parquet"))
if fc_files:
    df4 = pd.concat([pd.read_parquet(p) for p in fc_files], ignore_index=True)
    full = df4[df4["window_s"] == "full"]
    per_s = full.groupby("subject").agg(
        acc_4class=("correct_trial", "mean"),
        acc_hemi=("correct_hemisphere", "mean"),
        acc_inout=("correct_inner_outer", "mean"),
        n=("trial", "count"),
    )
    print(per_s.round(3).to_string())
    for col, ch in [("acc_4class", 0.25), ("acc_hemi", 0.5), ("acc_inout", 0.5)]:
        m, lo, hi = bootstrap_ci(per_s[col].values)
        print(f"  pooled {col} = {m:.3f}  [{lo:.3f}, {hi:.3f}]  chance={ch}")
    per_s.to_parquet(RESULTS_DIR / "iter4_4class_per_subject.parquet")

    # Window sweep.
    ws = df4[df4["window_s"] != "full"].copy()
    ws["window_s"] = pd.to_numeric(ws["window_s"])
    print("\nWindow sweep (mean across subjects):")
    print(ws.groupby("window_s")[["correct_trial","correct_hemisphere","correct_inner_outer"]].mean().round(3).to_string())
else:
    print("no 4class results found")
    df4 = None

# --- 2) EEG ↔ gaze predictability ---
section("[2] EEG↔gaze predictability and gaze→attended")
pred_files = sorted((RESULTS_DIR / "predictability").glob("s*.parquet"))
if pred_files:
    dfp = pd.concat([pd.read_parquet(p) for p in pred_files], ignore_index=True)
    print("EEG → gaze (per-target held-out Pearson r across subjects):")
    e2g = dfp[dfp["analysis"] == "eeg_to_gaze"]
    print(e2g.groupby("target")["value"].agg(["mean","std","count"]).round(3).to_string())
    print("\nCCA EEG↔gaze first-component correlation:")
    cc = dfp[dfp["analysis"] == "cca_eeg_gaze"]
    print(cc.groupby("target")["value"].agg(["mean","std"]).round(3).to_string())
    print("\nGaze → attended (logistic classifier):")
    g2a = dfp[dfp["analysis"] == "gaze_to_attended"]
    for tg, g in g2a.groupby("target"):
        m, lo, hi = bootstrap_ci(g["value"].values)
        ch = g["chance"].iloc[0]
        print(f"  {tg:<12} pooled={m:.3f}  [{lo:.3f},{hi:.3f}]  chance={ch:.3f}")
    dfp.to_parquet(RESULTS_DIR / "iter4_predictability.parquet")

# --- 3) Gaze-residualised EEG AAD ---
section("[3] Gaze-residualised EEG AAD (hemisphere task)")
gr_files = sorted((RESULTS_DIR / "gaze_residualised").glob("s*.parquet"))
if gr_files:
    dfr = pd.concat([pd.read_parquet(p) for p in gr_files], ignore_index=True)
    per = dfr.groupby(["subject","condition"])["correct_trial"].mean().unstack()
    print(per.round(3).to_string())
    print()
    for cond in per.columns:
        m, lo, hi = bootstrap_ci(per[cond].values)
        print(f"  {cond:<22}  pooled={m:.3f}  [{lo:.3f},{hi:.3f}]")
    if "baseline" in per and "gaze_residualised" in per:
        delta = per["gaze_residualised"] - per["baseline"]
        print(f"  mean delta (residualised − baseline): {delta.mean():+.3f}")
        from scipy.stats import wilcoxon
        try:
            stat, p = wilcoxon(delta.dropna())
            print(f"  Wilcoxon p = {p:.3f}")
        except Exception as e:
            print(f"  Wilcoxon failed: {e}")
    dfr.to_parquet(RESULTS_DIR / "iter4_gaze_residualised.parquet")

# --- 4) Extended fusion on the 3 task framings × 2 EEG backbones ---
section("[4] Extended fusion (Tasks A/B/C × CCA-mel-28 & 4-class EEG)")

def extended_fusion():
    gaze_p = RESULTS_DIR / "fusion_gaze_features.parquet"
    if not gaze_p.exists():
        print("  missing gaze features; skipping"); return
    G = pd.read_parquet(gaze_p)
    feat_cols = [c for c in G.columns if c not in ("subject","trial","attended","group","snr")]

    # EEG-only CCA-mel-28 probs (existing).
    from scripts.aad_fusion import load_eeg_oof
    E_cca = load_eeg_oof(RESULTS_DIR / "aad_v3", features="cca_mel")

    # EEG-only 4-class probs via iter-4 rho_1..4 (softmax).
    if df4 is not None:
        f4 = df4[df4["window_s"] == "full"].copy()
        rho = f4[["rho_1","rho_2","rho_3","rho_4"]].values
        # Average across folds per (subject, trial).
        f4["trial_key"] = f4["subject"].astype(str) + "_" + f4["trial"].astype(str)
        rho_avg = f4.groupby(["subject","trial","attended"])[["rho_1","rho_2","rho_3","rho_4"]].mean().reset_index()
        exp_rho = np.exp(rho_avg[["rho_1","rho_2","rho_3","rho_4"]].values * 5)
        P = exp_rho / exp_rho.sum(axis=1, keepdims=True)
        rho_avg[["p1","p2","p3","p4"]] = P
    else:
        rho_avg = pd.DataFrame()

    tasks = [
        ("hemisphere",  lambda a: 0 if ATTENDED_HEMISPHERE[a]=="L" else 1, 2),
        ("inner_outer", lambda a: 0 if a in (2,3) else 1, 2),
        ("4class",      lambda a: a-1, 4),
    ]

    rows = []
    for task_name, lbl_fn, nc in tasks:
        for s, g in G.groupby("subject"):
            if len(g) < 30: continue
            y = g["attended"].apply(lbl_fn).values
            if len(np.unique(y)) < nc: continue
            if pd.Series(y).value_counts().min() < 2: continue
            Xg = g[feat_cols].fillna(0).values
            skf = StratifiedKFold(5, shuffle=True, random_state=0)
            # Gaze-only.
            gaz_acc = []
            for tr,te in skf.split(Xg, y):
                m = Pipeline([("sc",StandardScaler()),("c",LogisticRegression(max_iter=2000,multi_class="auto"))]).fit(Xg[tr], y[tr])
                gaz_acc.append(accuracy_score(y[te], m.predict(Xg[te])))
            # EEG 4-class → task probs.
            eeg_acc_4c = np.nan; early_acc = np.nan
            if len(rho_avg):
                sub = rho_avg[rho_avg["subject"] == s].merge(g[["trial"]], on="trial", how="inner")
                if len(sub) == len(g):
                    Pe = sub[["p1","p2","p3","p4"]].values
                    if task_name == "hemisphere":
                        eeg_prob = Pe[:, [2,3]].sum(axis=1)  # P(right)
                        eeg_feat = Pe
                    elif task_name == "inner_outer":
                        eeg_prob = Pe[:, [0,3]].sum(axis=1)  # P(outer)
                        eeg_feat = Pe
                    else:
                        eeg_prob = Pe.argmax(axis=1)
                        eeg_feat = Pe
                    # Early fusion: LightGBM on [gaze feats, EEG 4-class probs]
                    Xc = np.concatenate([Xg, eeg_feat], axis=1)
                    early_acc_folds = []
                    for tr,te in skf.split(Xc, y):
                        m = lgb.LGBMClassifier(n_estimators=200, verbosity=-1).fit(Xc[tr], y[tr])
                        early_acc_folds.append(accuracy_score(y[te], m.predict(Xc[te])))
                    early_acc = float(np.mean(early_acc_folds))
                    # EEG-only accuracy (4-class logits as a features).
                    eeg_acc_folds = []
                    for tr,te in skf.split(eeg_feat, y):
                        m = lgb.LGBMClassifier(n_estimators=200, verbosity=-1).fit(eeg_feat[tr], y[tr])
                        eeg_acc_folds.append(accuracy_score(y[te], m.predict(eeg_feat[te])))
                    eeg_acc_4c = float(np.mean(eeg_acc_folds))
            rows.append(dict(subject=s, task=task_name, chance=1.0/nc,
                             acc_gaze=float(np.mean(gaz_acc)),
                             acc_eeg_4c=eeg_acc_4c, acc_early_fusion=early_acc))
    R = pd.DataFrame(rows)
    R.to_parquet(RESULTS_DIR / "iter4_fusion.parquet")
    print(R.round(3).to_string(index=False))
    print()
    for task_name in [t[0] for t in tasks]:
        sub = R[R["task"] == task_name]
        for c in ("acc_gaze","acc_eeg_4c","acc_early_fusion"):
            vals = sub[c].dropna().values
            if len(vals):
                m, lo, hi = bootstrap_ci(vals)
                print(f"  {task_name:<12} {c:<20}  pooled={m:.3f} [{lo:.3f},{hi:.3f}]")

extended_fusion()
print("\nSaved: iter4_summary parquets under results/ ; figures will be added separately.")
