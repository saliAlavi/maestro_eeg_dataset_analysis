"""Figures for the paper, drawn from the measured run outputs.

    python -m src.tools.make_figures --out docs/iclr2027/figs
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402

from .make_macros import PREF, load       # noqa: E402

# a brand-neutral, colour-blind-safe categorical ramp, consistent across figures
C = {"eeg": "#3B6FB6", "eeg_gaze": "#C9752B", "eeg_fovea": "#4C9A6B",
     "all": "#8E5BA6", "eeg_imu": "#B8563F", "eeg_flow": "#7B8794",
     "behaviour": "#A0A0A0"}
LBL = {"eeg": "EEG", "eeg_gaze": "EEG + gaze", "eeg_imu": "EEG + head",
       "eeg_flow": "EEG + flow", "eeg_fovea": "EEG + fovea",
       "behaviour": "behaviour only", "all": "all five"}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9", lw=0.7)
    ax.set_axisbelow(True)


def window_sweep(df, probes, out):
    """Contribution against decision-window length, and the model's own
    audio-only floor against the linear probe, both per window."""
    d = df[df["tag"].str.match(r"^w\d+$")].copy()
    if d.empty:
        print("[skip] window sweep: no runs tagged w*")
        return
    # Only stream sets measured at every window show the trend; the 2 s / 5 s
    # sweep over all eleven stream sets is a table, not a line plot.
    core = ["eeg", "eeg_gaze", "eeg_fovea", "all"]
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.2))
    for v in core:
        g = d[d["variant"] == v]
        if g.empty:
            continue
        s = (g.groupby("window_sec")[PREF + "contribution"]
             .agg(["mean", "std", "count"]).sort_index())
        se = s["std"] / np.sqrt(s["count"].clip(lower=1))
        ax[0].errorbar(s.index, s["mean"], yerr=se, marker="o", ms=4, lw=1.6,
                       capsize=2, color=C[v], label=LBL[v])
    ax[0].axhline(0, color="0.5", lw=0.8, ls="--")
    ax[0].set_xlabel("decision window (s)")
    ax[0].set_ylabel("contribution\n(accuracy $-$ shuffled accuracy)")
    ax[0].legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    _style(ax[0])

    s = (d[d["variant"] == "all"].groupby("window_sec")[PREF + "null_mean"]
         .mean().sort_index())
    ax[1].plot(s.index, s.values, marker="o", ms=4, lw=1.6, color="#8E5BA6",
               label="model's shuffled accuracy")
    # the probe is a property of the candidate set at each window, so it is the
    # same for every stream set run at that window; take it wherever measured
    pr = (d.dropna(subset=["audio_only_probe"]).groupby("window_sec")
          ["audio_only_probe"].first().sort_index())
    if len(pr):
        ax[1].plot(pr.index, pr.values, marker="s", ms=4, ls="--", lw=1.4,
                   color="#C9752B", label="linear probe")
    ax[1].axhline(0.25, color="0.5", lw=0.8, ls=":", label="chance")
    ax[1].set_ylim(0.21, 0.31)
    ax[1].set_xlabel("decision window (s)")
    ax[1].set_ylabel("audio-only accuracy")
    ax[1].legend(frameon=False, fontsize=8)
    _style(ax[1])
    fig.tight_layout()
    fig.savefig(os.path.join(out, "window_sweep.pdf"), bbox_inches="tight")
    print("wrote window_sweep.pdf")


def credit_bars(df, out):
    """Shapley credit per stream, with and without the contribution hinge."""
    # the clean 2x2 of the main text; every arm carries listener adaptation, so
    # the only things varying are the hinge and modality dropout
    d = df[df["tag"] == "v2hinge2"]
    if d.empty:
        print("[skip] credit bars: no hinge 2x2 runs")
        return
    # MERIT is the arm with no hinge and no dropout: it keeps the most credit on
    # the brain on unseen listeners, which is what the paper selects for
    order = ["H1_hinge_drop", "H3_hinge_nodrop",
             "H2_nohinge_drop", "H4_nohinge_nodrop"]
    names = ["hinge\n+ dropout", "hinge\nno dropout",
             "no hinge\n+ dropout", "MERIT"]
    streams = ["eeg", "gaze", "imu", "video", "fovea"]
    cols = ["#3B6FB6", "#C9752B", "#4C9A6B", "#7B8794", "#8E5BA6"]
    # keep names aligned with the variants that are actually present
    have = set(d["variant"])
    pairs = [(o, n) for o, n in zip(order, names) if o in have]
    if not pairs:
        print("[skip] credit bars: none of the 2x2 variants found")
        return
    order, names = [o for o, _ in pairs], [n for _, n in pairs]
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    x = np.arange(len(order))
    bottom_pos = np.zeros(len(order))
    for m, c in zip(streams, cols):
        col = PREF + f"shapley_{m}"
        if col not in d:
            continue
        v = np.array([d[d["variant"] == o][col].mean() for o in order])
        v = np.nan_to_num(v)
        ax.bar(x, v, 0.6, bottom=bottom_pos, color=c,
               label={"imu": "head", "video": "flow"}.get(m, m))
        bottom_pos += v
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    for t in ax.get_xticklabels():          # matplotlib does not read "\bf"
        if t.get_text() == "MERIT":
            t.set_fontweight("bold")
    ax.set_ylabel("Shapley credit $\\phi_m$")
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "credit_bars.pdf"), bbox_inches="tight")
    print("wrote credit_bars.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/iclr2027/figs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9,
                         "savefig.dpi": 200, "pdf.fonttype": 42})
    df = load()
    f = "/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/credit_candidate_probes.csv"
    probes = pd.read_csv(f) if os.path.exists(f) else None
    if df.empty:
        print("no results yet")
        return
    window_sweep(df, probes, a.out)
    credit_bars(df, a.out)


if __name__ == "__main__":
    main()
