"""Cohort-wide bad-channel & faulty-trial scan.

For every (subject, trial) with an `eeg_data.p` file, load the raw EEG and run
`detect_bad_channels` (plus extra diagnostics) to produce a per-trial manifest.

Outputs
-------
- analysis/results/bad_channels_manifest.parquet  — one row per trial, with
  bad-channel list and per-criterion channel counts.
- analysis/results/bad_channels_per_channel.parquet — long-format per
  (subject, trial, channel, reason).
- analysis/results/faulty_trials.parquet  — trials flagged as unrecoverable
  (>= `TRIAL_FAIL_BAD_FRAC` of channels bad, OR both mastoids bad + >= 5
  other bad channels).
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from aad_utils import (
    EEG_CHANNELS,
    EXPERIMENT_DIR,
    RESULTS_DIR,
    list_subjects,
    load_eeg_trial,
    eeg_raw_to_mne,
)
from aad_utils.preprocess import ADC_CLIP, detect_bad_channels

TRIAL_FAIL_BAD_FRAC = 0.25  # ≥25% bad → trial unrecoverable
N_TRIALS = 105  # 5 training + 100 main


def eval_exists(subject: int, k: int) -> bool:
    return (EXPERIMENT_DIR / f"Subject {subject}" / f"Eval-{k}" / "eeg_data.p").exists()


def scan_trial(subject: int, k: int) -> tuple[dict, list[dict]]:
    """Return (summary_row, long_format_rows)."""
    try:
        data, ts = load_eeg_trial(subject, k)
    except Exception as e:
        return (dict(subject=subject, trial=k, error=str(e)[:120]), [])

    raw = eeg_raw_to_mne(data)
    d = raw.get_data()  # (C, T)
    stds = d.std(axis=1)
    sat = (np.abs(d) >= ADC_CLIP).mean(axis=1)
    ac_var = np.var(np.diff(d, axis=1), axis=1)

    bads = detect_bad_channels(raw)
    # Per-criterion decomposition for the long-format table.
    long = []
    # Flat.
    for i, ch in enumerate(EEG_CHANNELS):
        if stds[i] < 1e-9:
            long.append(dict(subject=subject, trial=k, channel=ch, reason="flat",
                             value=float(stds[i])))
    # Saturated.
    for i, ch in enumerate(EEG_CHANNELS):
        if sat[i] >= 0.1:
            long.append(dict(subject=subject, trial=k, channel=ch, reason="saturated",
                             value=float(sat[i])))
    # Variance outlier (AC-var z > 6), using robust median/MAD on not-flat/sat channels.
    mask_ok = (stds >= 1e-9) & (sat < 0.1)
    if mask_ok.sum() >= 3:
        var_ok = ac_var[mask_ok]
        med = np.median(var_ok); mad = np.median(np.abs(var_ok - med)) or 1e-30
        for i, ch in enumerate(EEG_CHANNELS):
            if mask_ok[i]:
                z = (ac_var[i] - med) / (1.4826 * mad)
                if abs(z) > 6:
                    long.append(dict(subject=subject, trial=k, channel=ch,
                                     reason="variance_outlier", value=float(z)))

    summary = dict(
        subject=subject,
        trial=k,
        is_training=int(k <= 5),
        n_samples=int(d.shape[1]),
        duration_s=float(ts[-1] - ts[0]) if len(ts) > 1 else np.nan,
        n_bad=len(bads),
        bad_channels=";".join(bads),
        bad_fraction=len(bads) / len(EEG_CHANNELS),
        m1_flat=int("M1" in bads and stds[EEG_CHANNELS.index("M1")] < 1e-9),
        m1_sat=int(sat[EEG_CHANNELS.index("M1")] >= 0.1),
        m2_flat=int("M2" in bads and stds[EEG_CHANNELS.index("M2")] < 1e-9),
        m2_sat=int(sat[EEG_CHANNELS.index("M2")] >= 0.1),
        max_saturation_rate=float(sat.max()),
        mean_ac_std_uv=float(np.sqrt(ac_var.mean()) * 1e6),
    )
    return summary, long


def main():
    subjects = list_subjects()
    print(f"Scanning {len(subjects)} subjects × up to {N_TRIALS} trials each …")
    all_summ = []
    all_long = []
    t0 = time.time()
    for s in subjects:
        start = time.time()
        n_done = 0
        for k in range(1, N_TRIALS + 1):
            if not eval_exists(s, k):
                continue
            summ, long = scan_trial(s, k)
            all_summ.append(summ); all_long.extend(long); n_done += 1
        print(f"  Subject {s:2d}: {n_done} trials scanned in {time.time()-start:5.1f}s")
    print(f"\nTotal: {len(all_summ)} trials scanned in {time.time()-t0:.1f}s")

    df = pd.DataFrame(all_summ)
    long_df = pd.DataFrame(all_long)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RESULTS_DIR / "bad_channels_manifest.parquet")
    long_df.to_parquet(RESULTS_DIR / "bad_channels_per_channel.parquet")

    # Faulty-trial rule.
    df["faulty"] = (
        (df["bad_fraction"] >= TRIAL_FAIL_BAD_FRAC)
        | ((df["m1_flat"] + df["m1_sat"] >= 1) & (df["m2_flat"] + df["m2_sat"] >= 1) & (df["n_bad"] >= 7))
    ).astype(int)
    df[df["faulty"] == 1].to_parquet(RESULTS_DIR / "faulty_trials.parquet")

    # --- Summary report ---
    print("\n===== Summary =====")
    print(f"Trials scanned: {len(df)}")
    print(f"Trials with ≥1 bad channel: {(df['n_bad'] >= 1).sum()}  ({(df['n_bad']>=1).mean()*100:.1f}%)")
    print(f"Trials with ≥4 bad channels: {(df['n_bad'] >= 4).sum()}")
    print(f"Trials flagged faulty (≥{int(TRIAL_FAIL_BAD_FRAC*100)}% bad OR both-mastoids+≥7 bad): {df['faulty'].sum()}")
    print(f"\nM1 flat rate: {df['m1_flat'].mean()*100:.1f}% of trials")
    print(f"M2 saturated rate: {df['m2_sat'].mean()*100:.1f}% of trials")
    print()
    by_s = df.groupby("subject").agg(
        n=("trial", "count"),
        n_bad_mean=("n_bad", "mean"),
        n_bad_max=("n_bad", "max"),
        faulty=("faulty", "sum"),
        m1_flat=("m1_flat", "sum"),
        m2_sat=("m2_sat", "sum"),
    ).round(2)
    print("Per-subject summary:\n", by_s.to_string())

    # Most frequently bad channels overall.
    if len(long_df):
        ch_counts = long_df.groupby(["channel", "reason"]).size().unstack(fill_value=0)
        ch_counts["total"] = ch_counts.sum(axis=1)
        ch_counts = ch_counts.sort_values("total", ascending=False)
        print("\nTop-offender channels (count of trials flagged):\n", ch_counts.head(15).to_string())


if __name__ == "__main__":
    main()
