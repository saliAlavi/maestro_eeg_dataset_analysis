"""Aggregate iter-3 AAD results across all 4 feature backbones."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from aad_utils import RESULTS_DIR, FIGURES_DIR, set_pub_style, save_fig, bootstrap_ci, COLORS

set_pub_style()
AGG = RESULTS_DIR / "aad_v3"
files = sorted(AGG.glob("*.parquet"))
if not files:
    print("no files"); sys.exit(0)
dfs = []
for p in files:
    d = pd.read_parquet(p)
    d["source_file"] = p.name
    dfs.append(d)
df = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(files)} files, {len(df)} rows. Features seen: {df['features'].unique()}")

# Pooled full-trial accuracy by (features).
full = df[df["window_s"] == "full"]
pool = full.groupby("features")["correct_trial"].agg(["mean", "std", "count"]).round(3)
print("\n=== Iter-3 pooled full-trial accuracy (chance=0.5) ===")
print(pool.to_string())

# Per-subject per-features pivot.
per_s = full.groupby(["subject", "features"])["correct_trial"].mean().unstack()
print("\n=== Per-subject × features ===")
print(per_s.round(3).to_string())

# Best backbone per subject (inner adaptive pick).
best = per_s.idxmax(axis=1)
print("\nBest backbone per subject:")
print(best.to_string())
best_scores = per_s.max(axis=1)
print(f"\nOracle adaptive pooled = {best_scores.mean():.3f}")

per_s.to_parquet(RESULTS_DIR / "aad_v3_per_subject.parquet")
pool.to_parquet(RESULTS_DIR / "aad_v3_pooled.parquet")

# Window sweep per backbone.
if (df["window_s"] != "full").any():
    ws = df[df["window_s"] != "full"].copy()
    ws["window_s"] = pd.to_numeric(ws["window_s"])
    print("\n=== Window sweep (mean across subjects, per feature) ===")
    for feat, g in ws.groupby("features"):
        per_win = g.groupby(["subject", "window_s"])["correct_trial"].mean().reset_index()
        summ = per_win.groupby("window_s")["correct_trial"].agg(["mean", "std"]).round(3)
        print(f"\n{feat}:")
        print(summ.to_string())

# Figure: per-subject bars for the best backbone.
fig, ax = plt.subplots(figsize=(7, 3.5))
best_feat = pool["mean"].idxmax()
data = full[full["features"] == best_feat].groupby("subject")["correct_trial"].mean().sort_values()
ax.bar(data.index.astype(str), data.values, color=COLORS["eeg"])
ax.axhline(0.5, color=COLORS["chance"], ls="--", label="chance")
ax.set_title(f"iter-3 · {best_feat} · per-subject full-trial AAD")
ax.set_ylabel("accuracy"); ax.set_xlabel("subject"); ax.legend()
save_fig(fig, "iter3_per_subject_best", FIGURES_DIR)
plt.close(fig)

# Figure: backbone comparison boxplot.
fig, ax = plt.subplots(figsize=(6, 3.5))
order = sorted(full["features"].unique())
by_s = [per_s[f].dropna().values for f in order]
ax.boxplot(by_s, labels=order)
ax.axhline(0.5, color=COLORS["chance"], ls="--")
ax.set_ylabel("within-subject full-trial accuracy")
ax.set_title("iter-3 · backbone comparison (per-subject distributions)")
ax.tick_params(axis="x", rotation=20)
save_fig(fig, "iter3_backbone_box", FIGURES_DIR)
plt.close(fig)

print("\nFigures saved.")
