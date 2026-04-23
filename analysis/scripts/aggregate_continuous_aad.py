"""Aggregate continuous-AAD window-sweep and time-to-switch results from
existing per-trial outputs.

Reads:
    results/06_window_accuracy.parquet    (backward TRF × window)
    results/aad_v2_summary.parquet        (iter-2 window × condition sweep)
    results/aad_v4_4class/s*.parquet      (4-class decoder with per-window
                                           correct_trial / correct_hemisphere)

Writes:
    results/continuous_aad_window_sweep.parquet
    results/continuous_aad_time_to_switch.parquet
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def window_sweep() -> pd.DataFrame:
    rows: list[dict] = []

    # 1) backward TRF CCA mel-28 (from notebook 06) --
    fp = RESULTS / "06_window_accuracy.parquet"
    if fp.exists():
        df = pd.read_parquet(fp)
        for (subj, w), g in df.groupby(["subject", "window_s"]):
            rows.append({"model": "backward-trf-cca-mel-28",
                         "task": "envelope-attendance",
                         "subject": int(subj), "window_s": float(w),
                         "accuracy": float(g["correct"].mean()),
                         "n": int(len(g))})

    # 2) iter-2 aad_v2 summary (per-condition × window) --
    fp = RESULTS / "aad_v2_summary.parquet"
    if fp.exists():
        df = pd.read_parquet(fp)
        df = df[df["condition"] == "all·ica·derivative"]
        for (subj, w), g in df.groupby(["subject", "window_s"]):
            try:
                w_val = float(w)
            except (ValueError, TypeError):
                continue
            rows.append({"model": "backward-trf-iter2",
                         "task": "envelope-attendance",
                         "subject": int(subj),
                         "window_s": w_val,
                         "accuracy": float(g["mean"].iloc[0]),
                         "n": int(g["count"].iloc[0])})

    # 3) 4-class per-window (iter-4) --
    for fp in sorted(glob.glob(str(RESULTS / "aad_v4_4class/s*.parquet"))):
        df = pd.read_parquet(fp)
        for (subj, w), g in df.groupby(["subject", "window_s"]):
            # window_s is 'full' or int — skip full for the sweep
            if w == "full":
                continue
            # correct_trial = per-window fraction inside each trial row
            rows.append({"model": "cca-4class",
                         "task": "4class-identity",
                         "subject": int(subj), "window_s": float(w),
                         "accuracy": float(g["correct_trial"].mean()),
                         "n": int(len(g))})
            if "correct_hemisphere" in g.columns:
                rows.append({"model": "cca-4class",
                             "task": "hemisphere",
                             "subject": int(subj), "window_s": float(w),
                             "accuracy": float(g["correct_hemisphere"].mean()),
                             "n": int(len(g))})

    return pd.DataFrame(rows)


def time_to_switch() -> pd.DataFrame:
    """For every subject, per trial number, what is the accuracy of the
    4-class decoder at short windows — a proxy for attention locking
    speed. The 'switch' happens at every trial boundary.
    """
    rows: list[dict] = []
    for fp in sorted(glob.glob(str(RESULTS / "aad_v4_4class/s*.parquet"))):
        df = pd.read_parquet(fp)
        for w in ("1", "2", "4", "8", "16", 1.0, 2.0, 4.0, 8.0, 16.0):
            g = df[df["window_s"].astype(str) == str(w)]
            if g.empty:
                continue
            g = g.sort_values(["subject", "trial"])
            # Accuracy by trial-order position (proxy for time since session start)
            for subj, sub in g.groupby("subject"):
                for pos, row in sub.reset_index(drop=True).iterrows():
                    rows.append({
                        "subject": int(subj), "window_s": float(w),
                        "trial_pos": int(pos),
                        "correct_hemi": float(row.get("correct_hemisphere",
                                                      np.nan)),
                        "correct_4": float(row.get("correct_trial", np.nan)),
                    })
    return pd.DataFrame(rows)


def main() -> int:
    ws = window_sweep()
    out_ws = RESULTS / "continuous_aad_window_sweep.parquet"
    ws.to_parquet(out_ws, index=False)
    print(f"wrote {out_ws}  ({len(ws)} rows)")

    # print pooled per (model, task, window)
    pooled = ws.groupby(["model", "task", "window_s"])["accuracy"].agg(
        ["mean", "std", "count"]
    ).reset_index()
    print(pooled.to_string(index=False))

    tts = time_to_switch()
    out_tts = RESULTS / "continuous_aad_time_to_switch.parquet"
    tts.to_parquet(out_tts, index=False)
    print(f"\nwrote {out_tts}  ({len(tts)} rows)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
