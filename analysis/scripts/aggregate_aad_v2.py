"""Aggregate iter-2 AAD results and render the iter-2 summary figures."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aad_utils import RESULTS_DIR, FIGURES_DIR, set_pub_style, save_fig, bootstrap_ci, COLORS

set_pub_style()

AGG_DIR = RESULTS_DIR / "aad_v2"
out_files = sorted(AGG_DIR.glob("*.parquet"))
if not out_files:
    print("No result files under", AGG_DIR); sys.exit(0)

dfs = [pd.read_parquet(p) for p in out_files]
df = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(out_files)} files, {len(df)} rows")

# Tag condition from filename: sN_{tag}_{target}.parquet — match to columns.
def tag(row):
    t = []
    t.append("correct" if row["correct_only"] else "all")
    t.append("ica" if row["apply_ica"] else "noica")
    t.append(row["target_mode"])
    return "·".join(t)

df["condition"] = df.apply(tag, axis=1)

# --- Per-(subject, condition, window) mean accuracy ---
agg = (df.groupby(["subject", "condition", "window_s"])["correct_trial"]
         .agg(["mean", "count"]).reset_index())
agg.to_parquet(RESULTS_DIR / "aad_v2_summary.parquet")

# Print pooled accuracy per condition (full-trial).
print("\n=== Pooled full-trial accuracy per condition ===")
full = df[df["window_s"] == "full"]
print(full.groupby("condition")["correct_trial"].agg(["mean", "count"]))

# --- Per-subject accuracy bar ---
primary = full[full["condition"] == "all·ica·derivative"]
if len(primary):
    per_s = primary.groupby("subject")["correct_trial"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(per_s.index.astype(str), per_s.values, color=COLORS["eeg"])
    ax.axhline(0.5, color=COLORS["chance"], ls="--", label="chance")
    ax.set_xlabel("subject"); ax.set_ylabel("AAD accuracy (full trial, 5-fold)")
    ax.set_title("iter-2 · all-trials · ICA · envelope-derivative")
    ax.legend()
    save_fig(fig, "iter2_per_subject", FIGURES_DIR)
    plt.close()

# --- Window-length sweep with bootstrap CI ---
if len(primary):
    ws_df = df[(df["condition"] == "all·ica·derivative") & (df["window_s"] != "full")]
    ws_df = ws_df.copy(); ws_df["window_s"] = ws_df["window_s"].astype(float)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    xs, means, los, his = [], [], [], []
    for w, g in ws_df.groupby("window_s"):
        vals = g.groupby("subject")["correct_trial"].mean().values
        m, lo, hi = bootstrap_ci(vals)
        xs.append(w); means.append(m); los.append(lo); his.append(hi)
    ax.errorbar(xs, means, yerr=[np.array(means)-np.array(los), np.array(his)-np.array(means)],
                fmt="o-", color=COLORS["eeg"], capsize=3)
    ax.axhline(0.5, color=COLORS["chance"], ls="--")
    ax.set_xscale("log"); ax.set_xlabel("decision window (s)"); ax.set_ylabel("AAD accuracy")
    ax.set_title("iter-2 · window-length sweep (mean ± bootstrap 95%)")
    save_fig(fig, "iter2_window_sweep", FIGURES_DIR); plt.close()

# --- Correct-only vs all comparison ---
comp = full[full["condition"].isin(["all·ica·derivative", "correct·ica·derivative"])]
if len(comp):
    wide = comp.groupby(["subject", "condition"])["correct_trial"].mean().unstack()
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    a = wide["all·ica·derivative"]; c = wide["correct·ica·derivative"]
    ax.scatter(a, c, color=COLORS["attended"])
    for s in wide.index:
        ax.annotate(f"S{s}", (wide.loc[s, 'all·ica·derivative'], wide.loc[s, 'correct·ica·derivative']), fontsize=7)
    lo, hi = min(a.min(), c.min()) - 0.02, max(a.max(), c.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.5)
    ax.set_xlabel("all-trials accuracy"); ax.set_ylabel("correct-only accuracy")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_title("Correct-only vs. all-trials")
    save_fig(fig, "iter2_correct_vs_all", FIGURES_DIR); plt.close()
    # Paired test
    from scipy.stats import wilcoxon
    diff = c - a
    try:
        stat, p = wilcoxon(diff.dropna())
        print(f"\nCorrect-only − all-trials: mean Δ = {diff.mean():+.3f}  Wilcoxon p = {p:.3f}")
    except Exception as e:
        print("Wilcoxon failed:", e)

# --- Ablation comparison bar ---
abl = full.groupby("condition")["correct_trial"].agg(["mean", "std", "count"]).reset_index()
print("\n=== Ablation summary ===")
print(abl.to_string(index=False))
fig, ax = plt.subplots(figsize=(6, 3.5))
order = abl.sort_values("mean")["condition"].tolist()
means = [abl.loc[abl["condition"] == c, "mean"].values[0] for c in order]
stds = [abl.loc[abl["condition"] == c, "std"].values[0] for c in order]
ax.bar(order, means, yerr=stds, color=COLORS["eeg"], capsize=4)
ax.axhline(0.5, color=COLORS["chance"], ls="--")
ax.set_ylabel("pooled AAD accuracy")
ax.tick_params(axis="x", rotation=20)
ax.set_title("iter-2 · ablations (full-trial)")
save_fig(fig, "iter2_ablations", FIGURES_DIR); plt.close()
print("\nFigures saved to analysis/figures/iter2_*.pdf")
