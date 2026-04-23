"""Stratify per-trial decoder outputs by test-trial SNR and attended
direction. This measures conditional accuracy; it is not a
train-on-bin-A / test-on-bin-B retraining matrix (that requires a
separate SLURM job), but it answers the first-order reviewer
question 'does your decoder collapse at low SNR / at lateral-vs-midline
speakers?' from existing CV outputs.

Reads:
    results/aad_v4_4class/s*.parquet      (per-fold, per-trial predictions)
    results/gaze_residualised/s*.parquet  (baseline + residualised per-trial)
    results/07_gaze_features.parquet + 07_within_subject_gaze.parquet

Writes:
    results/generalisation_snr.parquet
    results/generalisation_direction.parquet
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def snr_bin(snr: float) -> str:
    if snr <= 6:
        return "≤6"
    if snr <= 10:
        return "7-10"
    if snr <= 14:
        return "11-14"
    return "≥15"


def az_bucket(az: float) -> str:
    if abs(az) < 30:
        return "inner (±22.5°)"
    return "outer (±67.5°)"


def hemi(az: float) -> str:
    return "left" if az < 0 else "right"


def aggregate_4class():
    rows: list[dict] = []
    for fp in sorted(glob.glob(str(RESULTS / "aad_v4_4class/s*.parquet"))):
        df = pd.read_parquet(fp)
        # only trial-level (window_s == 'full')
        sub = df[df["window_s"] == "full"].copy()
        if sub.empty:
            continue
        # attended azimuth via trial lookup (approx — use correct labels from
        # behavioural parquet)
        sub["snr_bin"] = sub["snr"].apply(snr_bin)
        # attended speaker label 1..4 → az
        az_map = {1: -67.5, 2: -22.5, 3: +22.5, 4: +67.5}
        sub["az"] = sub["attended"].map(az_map)
        sub["az_bucket"] = sub["az"].apply(az_bucket)
        sub["hemi"] = sub["az"].apply(hemi)
        rows.append(sub[["subject", "snr", "snr_bin", "az", "az_bucket",
                         "hemi", "correct_trial", "correct_hemisphere",
                         "correct_inner_outer"]])
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    all_df = pd.concat(rows, ignore_index=True)

    snr_tbl = all_df.groupby("snr_bin").agg(
        acc_4class=("correct_trial", "mean"),
        acc_hemi=("correct_hemisphere", "mean"),
        acc_inout=("correct_inner_outer", "mean"),
        n=("correct_trial", "count"),
    ).reset_index()

    dir_tbl = all_df.groupby(["hemi", "az_bucket"]).agg(
        acc_4class=("correct_trial", "mean"),
        acc_hemi=("correct_hemisphere", "mean"),
        acc_inout=("correct_inner_outer", "mean"),
        n=("correct_trial", "count"),
    ).reset_index()

    return snr_tbl, dir_tbl


def aggregate_gaze_resid():
    rows: list[dict] = []
    for fp in sorted(glob.glob(str(RESULTS / "gaze_residualised/s*.parquet"))):
        df = pd.read_parquet(fp)
        df["snr_bin"] = df["snr"].apply(snr_bin)
        az_map = {1: -67.5, 2: -22.5, 3: +22.5, 4: +67.5}
        df["az"] = df["attended"].map(az_map)
        df["hemi"] = df["az"].apply(hemi)
        df["az_bucket"] = df["az"].apply(az_bucket)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    all_df = pd.concat(rows, ignore_index=True)
    # accuracy per condition × snr_bin × hemi
    t = all_df.groupby(["condition", "snr_bin"]).agg(
        acc=("correct_trial", "mean"),
        n=("correct_trial", "count"),
    ).reset_index()
    return t


def main() -> int:
    snr_tbl, dir_tbl = aggregate_4class()
    snr_tbl.to_parquet(RESULTS / "generalisation_snr.parquet", index=False)
    dir_tbl.to_parquet(RESULTS / "generalisation_direction.parquet", index=False)
    print("SNR-stratified accuracy (4-class decoder, full trial):")
    print(snr_tbl.to_string(index=False))
    print("\nDirection-stratified accuracy:")
    print(dir_tbl.to_string(index=False))

    gr = aggregate_gaze_resid()
    if not gr.empty:
        gr.to_parquet(RESULTS / "generalisation_gaze_resid_snr.parquet",
                      index=False)
        print("\nGaze-residualised EEG decoder × SNR bin:")
        print(gr.to_string(index=False))

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
