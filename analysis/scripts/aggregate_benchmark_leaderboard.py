"""Aggregate per-subject AAD results into a unified leaderboard parquet.

Reads:
    results/fusion_*.parquet                (within-subject fusion per backbone)
    results/aad_v3_per_subject.parquet      (v3 backbones; 4-class-adjacent AAD)
    results/aad_v4_4class/s*.parquet        (4-class decoder per subject)
    results/eeg_spectral/s*.parquet         (iter-5 spectral decoder)
    results/video_aad/s*.parquet            (video-only)
    results/imu_aad/s*.parquet              (IMU-only)
    results/gaze_residualised/s*.parquet    (gaze-residualised EEG)
    results/07_loso_gaze.parquet            (LOSO gaze)
    results/06_loso_backward.parquet        (LOSO EEG backward TRF)

Writes:
    results/benchmark_leaderboard.parquet
        columns: model, task, split, subject (or 'pooled'), accuracy,
                 chance, n_trials
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def _read_all(pattern: str) -> pd.DataFrame:
    frames = []
    for fp in sorted(glob.glob(str(RESULTS / pattern))):
        if ".features" in fp:
            continue
        frames.append(pd.read_parquet(fp))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _boot_ci(vals: np.ndarray, n: int = 10_000, alpha: float = 0.05):
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(vals), size=(n, len(vals)))
    boots = vals[idx].mean(axis=1)
    return np.quantile(boots, [alpha / 2, 1 - alpha / 2])


def fusion_rows() -> list[dict]:
    """Gaze / EEG-backbone fusion results from fusion_*.parquet."""
    backbones = ["broadband", "mel28", "cca_mel", "split_delta_theta"]
    rows: list[dict] = []
    for bb in backbones:
        fp = RESULTS / f"fusion_{bb}.parquet"
        if not fp.exists():
            continue
        df = pd.read_parquet(fp)
        for _, r in df.iterrows():
            for col, tag in [
                ("acc_gaze", "gaze-only"),
                ("acc_eeg_lr", f"eeg-{bb}"),
                ("acc_late", f"fusion-late-{bb}"),
                ("acc_stack", f"fusion-stack-{bb}"),
                ("acc_early", f"fusion-early-{bb}"),
            ]:
                rows.append({
                    "model": tag, "task": "hemisphere",
                    "split": "within-5fold", "subject": int(r["subject"]),
                    "accuracy": float(r[col]),
                    "chance": 0.5, "n_trials": int(r["n"]),
                })
    return rows


def video_imu_spec_rows() -> list[dict]:
    """Video / IMU / EEG-spectral per-subject per-task rows."""
    rows: list[dict] = []
    for tag, patt in [("video", "video_aad/s*.parquet"),
                      ("imu", "imu_aad/s*.parquet"),
                      ("eeg-spectral", "eeg_spectral/s*.parquet")]:
        df = _read_all(patt)
        if df.empty:
            continue
        for _, r in df.iterrows():
            rows.append({
                "model": f"{tag}-{r['classifier']}",
                "task": r["task"],
                "split": "within-5fold",
                "subject": int(r["subject"]),
                "accuracy": float(r["acc"]),
                "chance": float(r["chance"]),
                "n_trials": 100,
            })
    return rows


def aad_v3_rows() -> list[dict]:
    """Per-subject CCA-mel-28 / broadband backbones on the full-trial task."""
    fp = RESULTS / "aad_v3_per_subject.parquet"
    if not fp.exists():
        return []
    df = pd.read_parquet(fp).reset_index()
    rows: list[dict] = []
    for _, r in df.iterrows():
        for bb in ["broadband", "cca_mel", "mel28", "split_delta_theta"]:
            if bb not in df.columns:
                continue
            rows.append({
                "model": f"backward-trf-{bb}",
                "task": "envelope-attendance",
                "split": "within-5fold",
                "subject": int(r["subject"]),
                "accuracy": float(r[bb]),
                "chance": 0.5,
                "n_trials": 100,
            })
    return rows


def aad_v4_rows() -> list[dict]:
    """4-class CCA decoder, per-subject pooled full-trial accuracy."""
    rows: list[dict] = []
    for fp in sorted(glob.glob(str(RESULTS / "aad_v4_4class/s*.parquet"))):
        df = pd.read_parquet(fp)
        full = df[df["window_s"] == "full"]
        if full.empty:
            continue
        subj = int(df["subject"].iloc[0])
        rows.append({
            "model": "cca-4class",
            "task": "4class-identity",
            "split": "within-5fold",
            "subject": subj,
            "accuracy": float(full["correct_trial"].mean()),
            "chance": 0.25,
            "n_trials": int(len(full)),
        })
        if "correct_hemisphere" in full.columns:
            rows.append({
                "model": "cca-4class",
                "task": "hemisphere",
                "split": "within-5fold",
                "subject": subj,
                "accuracy": float(full["correct_hemisphere"].mean()),
                "chance": 0.5,
                "n_trials": int(len(full)),
            })
    return rows


def gaze_resid_rows() -> list[dict]:
    """Gaze-residualised EEG AAD (iter-4 headline control)."""
    rows: list[dict] = []
    for fp in sorted(glob.glob(str(RESULTS / "gaze_residualised/s*.parquet"))):
        df = pd.read_parquet(fp)
        subj = int(df["subject"].iloc[0])
        for cond in df["condition"].unique():
            sub = df[df["condition"] == cond]
            rows.append({
                "model": f"backward-trf-cca-mel-{cond}",
                "task": "hemisphere-residualised",
                "split": "within-5fold",
                "subject": subj,
                "accuracy": float(sub["correct_trial"].mean()),
                "chance": 0.5,
                "n_trials": int(len(sub)),
            })
    return rows


def loso_rows() -> list[dict]:
    """LOSO splits."""
    rows: list[dict] = []
    for name, fp, model, task, chance in [
        ("loso-gaze", "07_loso_gaze.parquet", "gaze-logreg",
         "4class-identity", 0.25),
        ("loso-backward-trf", "06_loso_backward.parquet",
         "backward-trf-cca-mel", "hemisphere", 0.5),
    ]:
        fpath = RESULTS / fp
        if not fpath.exists():
            continue
        df = pd.read_parquet(fpath)
        if "acc" in df.columns:
            for _, r in df.iterrows():
                rows.append({"model": model, "task": task, "split": "LOSO",
                             "subject": int(r["subject"]), "accuracy": float(r["acc"]),
                             "chance": chance, "n_trials": 100})
        if "correct" in df.columns:
            for subj, g in df.groupby("test_subject"):
                rows.append({"model": model, "task": task, "split": "LOSO",
                             "subject": int(subj),
                             "accuracy": float(g["correct"].mean()),
                             "chance": chance, "n_trials": int(len(g))})
    # Per-subject LOSO CCA mel-28 (new SLURM output)
    for fpath in sorted(glob.glob(str(RESULTS / "loso_cca_mel28/s*.parquet"))):
        df = pd.read_parquet(fpath)
        subj = int(df["test_subject"].iloc[0])
        rows.append({"model": "backward-trf-cca-mel-28",
                     "task": "envelope-attendance", "split": "LOSO",
                     "subject": subj,
                     "accuracy": float(df["correct_trial"].mean()),
                     "chance": 0.5, "n_trials": int(len(df))})
    return rows


def main() -> int:
    all_rows: list[dict] = []
    all_rows += fusion_rows()
    all_rows += video_imu_spec_rows()
    all_rows += aad_v3_rows()
    all_rows += aad_v4_rows()
    all_rows += gaze_resid_rows()
    all_rows += loso_rows()

    lb = pd.DataFrame(all_rows)
    # Pooled per (model, task, split) with bootstrap CI over per-subject accs
    pooled: list[dict] = []
    for (model, task, split), g in lb.groupby(["model", "task", "split"]):
        vals = g["accuracy"].to_numpy()
        if len(vals) < 2:
            continue
        lo, hi = _boot_ci(vals)
        pooled.append({
            "model": model, "task": task, "split": split,
            "subject": -1, "accuracy": float(vals.mean()),
            "chance": float(g["chance"].iloc[0]),
            "n_trials": int(g["n_trials"].sum()),
            "n_subjects": int(len(vals)),
            "ci_lo": float(lo), "ci_hi": float(hi),
        })
    pooled_df = pd.DataFrame(pooled)
    print(pooled_df.sort_values(["task", "accuracy"],
                                ascending=[True, False]).to_string(index=False))

    out = RESULTS / "benchmark_leaderboard.parquet"
    lb.to_parquet(out, index=False)
    pooled_df.to_parquet(RESULTS / "benchmark_leaderboard_pooled.parquet",
                         index=False)
    print(f"\nwrote {out} ({len(lb)} rows)")
    print(f"wrote pooled ({len(pooled_df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
