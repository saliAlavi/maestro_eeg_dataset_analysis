"""Publication figures for the new-experiments block.

Reads analysis/results/new_experiments/*.parquet and writes:

  F10_learning_curve.{pdf,png}
  F11_partial_motion.{pdf,png}
  F12_snr_per_decoder.{pdf,png}
  F13_quality_composite.{pdf,png}
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from aad_utils import RESULTS_DIR, FIGURES_DIR, set_pub_style, save_fig, COLORS
set_pub_style()

OUT = RESULTS_DIR / "new_experiments"
FIG = FIGURES_DIR
FIG.mkdir(parents=True, exist_ok=True)

# ---------- learning curves ------------------------------------------
B = pd.read_parquet(OUT / "B_learning_curve.parquet")
fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=False)
for ax, task in zip(axes, ["hemisphere", "inner_outer", "4class"]):
    sub = B[B.task == task]
    for m, g in sub.groupby("model"):
        lc = g.groupby("n_train")["acc"].mean()
        lc_sd = g.groupby("n_train")["acc"].std()
        ax.errorbar(lc.index, lc.values, yerr=lc_sd.values, marker="o",
                    capsize=3, label=m)
    chance = 0.5 if task != "4class" else 0.25
    ax.axhline(chance, ls="--", c="k", lw=0.8)
    ax.set_xlabel("train trials per subject"); ax.set_title(task)
axes[0].set_ylabel("accuracy")
axes[-1].legend(loc="lower right", fontsize=7, frameon=False)
plt.tight_layout()
save_fig(fig, "F10_learning_curve", FIG)

# ---------- partial motion sweep -------------------------------------
D = pd.read_parquet(OUT / "D_partial_motion.parquet")
fig, ax = plt.subplots(figsize=(5.0, 3.2))
for task in ["hemisphere", "inner_outer", "4class"]:
    g = D[D.task == task].groupby("alpha")["acc"].agg(["mean", "std"])
    ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", capsize=3, label=task)
ax.set_xlabel(r"$\alpha$ (fraction of motion subspace removed)")
ax.set_ylabel("spectral-classifier accuracy")
ax.axhline(0.5, ls="--", c="gray", lw=0.7)
ax.axhline(0.25, ls="--", c="gray", lw=0.7)
ax.legend(frameon=False, fontsize=8)
plt.tight_layout()
save_fig(fig, "F11_partial_motion", FIG)

# ---------- SNR stratification ---------------------------------------
try:
    C = pd.read_parquet(OUT / "C_snr_stratified.parquet")
    order = ["<=6", "7-10", "11-14", ">=15"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
    for ax, task in zip(axes, ["hemisphere", "inner_outer", "4class"]):
        sub = C[C.task == task]
        for m, g in sub.groupby("model"):
            g2 = g.set_index("snr_bin").reindex(order)
            ax.plot(order, g2["acc"], marker="o", label=m)
        chance = 0.5 if task != "4class" else 0.25
        ax.axhline(chance, ls="--", c="k", lw=0.8)
        ax.set_title(task); ax.set_xlabel("SNR bin (dB)")
    axes[0].set_ylabel("accuracy")
    axes[-1].legend(loc="lower right", fontsize=7, frameon=False)
    plt.tight_layout()
    save_fig(fig, "F12_snr_per_decoder", FIG)
except FileNotFoundError:
    print("C_snr_stratified.parquet missing, skipping F12")

# ---------- quality composite scatter --------------------------------
try:
    E_subj = pd.read_parquet(OUT / "E_quality_per_subject.parquet")
    tgt_options = [c for c in ("gaze_hemi", "eeg_spectral_hemi",
                               "fusion_hemi", "eeg_hemi") if c in E_subj.columns]
    fig, axes = plt.subplots(1, len(tgt_options), figsize=(3.2 * len(tgt_options), 3.2),
                             sharey=True)
    if len(tgt_options) == 1: axes = [axes]
    for ax, t in zip(axes, tgt_options):
        sub = E_subj[["quality_composite", t]].dropna()
        ax.scatter(sub["quality_composite"], sub[t], s=28)
        r = sub.corr().iloc[0, 1]
        ax.set_title(f"{t}  r={r:+.2f}")
        ax.set_xlabel("quality composite (z)")
    axes[0].set_ylabel("accuracy")
    plt.tight_layout()
    save_fig(fig, "F13_quality_composite", FIG)
except FileNotFoundError:
    print("E_quality_per_subject.parquet missing, skipping F13")

print("figures written to", FIG)
