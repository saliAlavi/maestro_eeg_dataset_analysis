"""Render publication-quality data-quality figures from the cohort scan."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aad_utils import EEG_CHANNELS, FIGURES_DIR, RESULTS_DIR, set_pub_style, save_fig, COLORS

set_pub_style()

manifest = pd.read_parquet(RESULTS_DIR / "bad_channels_manifest.parquet")
long_df = pd.read_parquet(RESULTS_DIR / "bad_channels_per_channel.parquet")
N_SUBJ = manifest["subject"].nunique()

# --------------------------------------------------------------------------- #
# Figure DQ1 — bad-channel rate heatmap (channels × subjects)
# --------------------------------------------------------------------------- #
rate = (
    long_df.groupby(["subject", "channel"]).size()
    / manifest.groupby("subject")["trial"].count()
)
rate = rate.unstack(fill_value=0.0).reindex(columns=EEG_CHANNELS, fill_value=0.0)

fig, ax = plt.subplots(figsize=(9, 4.5))
im = ax.imshow(rate.values, aspect="auto", cmap="Reds", vmin=0, vmax=1)
ax.set_yticks(range(len(rate.index)))
ax.set_yticklabels([f"S{s}" for s in rate.index])
ax.set_xticks(range(len(EEG_CHANNELS)))
ax.set_xticklabels(EEG_CHANNELS, rotation=90)
plt.colorbar(im, ax=ax, label="fraction of trials flagged bad")
ax.set_title("DQ1 · Per-subject × per-channel bad-flag rate (all trials)")
save_fig(fig, "DQ1_bad_channel_rate", FIGURES_DIR)
plt.close(fig)

# --------------------------------------------------------------------------- #
# Figure DQ2 — top reasons by channel
# --------------------------------------------------------------------------- #
reason_counts = long_df.groupby(["channel", "reason"]).size().unstack(fill_value=0)
reason_counts = reason_counts.reindex(EEG_CHANNELS, fill_value=0)
fig, ax = plt.subplots(figsize=(10, 3.5))
bottom = np.zeros(len(EEG_CHANNELS))
palette = dict(flat="#555", saturated=COLORS["audio"], variance_outlier=COLORS["eeg"])
for col in ("flat", "saturated", "variance_outlier"):
    if col in reason_counts:
        ax.bar(EEG_CHANNELS, reason_counts[col].values, bottom=bottom,
               color=palette[col], label=col)
        bottom += reason_counts[col].values
ax.set_ylabel("# trials flagged")
ax.tick_params(axis="x", rotation=90)
ax.legend()
ax.set_title("DQ2 · Flag reason per channel (all subjects, all trials)")
save_fig(fig, "DQ2_reasons_by_channel", FIGURES_DIR)
plt.close(fig)

# --------------------------------------------------------------------------- #
# Figure DQ3 — per-trial bad-channel count timeline, per subject
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(10, 3.5))
for s in sorted(manifest["subject"].unique()):
    sub = manifest[manifest["subject"] == s].sort_values("trial")
    ax.plot(sub["trial"], sub["n_bad"], alpha=0.5, label=f"S{s}")
ax.set_xlabel("Eval index")
ax.set_ylabel("# bad channels")
ax.set_title("DQ3 · Bad-channel count per trial (all subjects)")
ax.legend(ncol=8, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.2))
save_fig(fig, "DQ3_badcount_per_trial", FIGURES_DIR)
plt.close(fig)

# --------------------------------------------------------------------------- #
# Figure DQ4 — mastoid status grid
# --------------------------------------------------------------------------- #
grid_m1 = manifest.pivot(index="subject", columns="trial", values="m1_flat").fillna(0)
grid_m2 = manifest.pivot(index="subject", columns="trial", values="m2_sat").fillna(0)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
axes[0].imshow(grid_m1.values, aspect="auto", cmap="Greys", vmin=0, vmax=1)
axes[0].set_title("M1 flat (black = flat)")
axes[1].imshow(grid_m2.values, aspect="auto", cmap="Oranges", vmin=0, vmax=1)
axes[1].set_title("M2 saturated (orange = sat)")
for ax in axes:
    ax.set_xlabel("trial")
    ax.set_yticks(range(len(grid_m1.index)))
    ax.set_yticklabels([f"S{s}" for s in grid_m1.index])
axes[0].set_ylabel("subject")
plt.suptitle("DQ4 · Mastoid electrode status across the cohort", y=1.01)
save_fig(fig, "DQ4_mastoid_status", FIGURES_DIR)
plt.close(fig)

print("Saved DQ figures to", FIGURES_DIR)
print("Rate table top-5 worst channels overall:")
overall = rate.mean(axis=0).sort_values(ascending=False).head(10)
print(overall.round(3))
