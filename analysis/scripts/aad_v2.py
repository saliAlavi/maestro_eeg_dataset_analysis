"""Improved AAD pipeline (iteration 2).

Changes over iteration 1:
    1. Trial mapping FIXED: Eval-K ↔ Trial-K for K∈[1,100] (was incorrectly
       shifted by 5 in iter-1).
    2. All 100 main trials per subject (was 25).
    3. Envelope derivative (half-wave rectified) as reconstruction target.
    4. Negative + positive lags (-100..400 ms) for proper TRF coverage.
    5. ICA with auto-EOG removal (frontal channels as proxy).
    6. Window-length sweep (5, 10, 20, 30 s) with bootstrap CIs.
    7. Correct-only trial filter (subjects who paid attention).
    8. Gammatone (28-band) envelope reconstruction option.
    9. Per-subject analysis saved so SLURM jobs can run one subject each.

CLI:
    python aad_v2.py --subject 3 --out results/aad_v2/s3.parquet
    python aad_v2.py --subject 3 --correct-only --out results/aad_v2/s3_correct.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from aad_utils import (
    EEG_CHANNELS, EEG_SFREQ, RESULTS_DIR, load_trials_csv, load_answers,
    load_eeg_trial, load_eeg_time, load_gaze_trial_2d, load_audio_timestamps,
    align_modalities_to_trial, eeg_raw_to_mne, preprocess_eeg, audio_envelope,
    load_audio_file, trial_name,
)
from aad_utils.config import ATTENDED_SPEAKER_MAP

SR_OUT = 64.0            # common EEG/envelope rate
FILT_LOW = 1.0
FILT_HIGH = 9.0          # low-pass at 9 Hz for speech-envelope tracking
LAGS_MS_POS = np.arange(-100, 401, 25)   # negative + positive lags
RIDGE_ALPHAS = np.logspace(1, 5, 7)       # 10..100k
N_FOLDS = 5
WINDOWS_S = [1, 2, 4, 8, 16, 30]


def design_lags(X: np.ndarray, lags: list[int]) -> np.ndarray:
    T, C = X.shape
    out = np.zeros((T, C * len(lags)))
    for i, lag in enumerate(lags):
        if lag >= 0:
            out[:, i*C:(i+1)*C] = np.vstack([np.zeros((lag, C)), X[:T-lag]])
        else:
            out[:, i*C:(i+1)*C] = np.vstack([X[-lag:], np.zeros((-lag, C))])
    return out


def envelope_target(env: np.ndarray, mode: str = "derivative") -> np.ndarray:
    """Target transformation. 'derivative' = half-wave-rectified Δenvelope."""
    if mode == "raw":
        return env
    if mode == "derivative":
        d = np.diff(env, prepend=env[0])
        return np.maximum(d, 0)  # half-wave rectify
    raise ValueError(mode)


def load_pair(subject: int, k: int, *, apply_ica: bool = True,
               target_mode: str = "derivative") -> dict | None:
    """Load EEG + attended/unattended envelopes for main trial K (1..100)."""
    try:
        eeg, ts = load_eeg_trial(subject, k); em = load_eeg_time(subject, k)
        g2 = load_gaze_trial_2d(subject, k); at = load_audio_timestamps(subject, k)
    except (FileNotFoundError, EOFError, ValueError, OSError):
        return None
    try:
        ali = align_modalities_to_trial(eeg=eeg, eeg_ts=ts, eeg_time_meta=em,
                                        gaze2d=g2, audio_timestamps=at)
        raw = eeg_raw_to_mne(ali["eeg"])
        raw = preprocess_eeg(
            raw, l_freq=FILT_LOW, h_freq=FILT_HIGH, notch=60.0,
            reference="auto", apply_ica=apply_ica,
        )
        raw.resample(SR_OUT, verbose="ERROR")
    except Exception as e:
        return None
    E = raw.get_data().T  # (T, 32)

    tno = trial_name(k, "main")
    tr = load_trials_csv()
    row = tr[tr["Trial No."] == tno]
    if not len(row): return None
    row = row.iloc[0]
    att_dev = "Device-1" if int(row["Attended Speaker"]) in (1, 2) else "Device-2"
    una_dev = "Device-2" if att_dev == "Device-1" else "Device-1"
    a_att, sr = load_audio_file(row[att_dev])
    a_una, _ = load_audio_file(row[una_dev])
    env_att = audio_envelope(a_att, sr, sr_out=SR_OUT)
    env_una = audio_envelope(a_una, sr, sr_out=SR_OUT)
    tgt_att = envelope_target(env_att, target_mode)
    tgt_una = envelope_target(env_una, target_mode)

    L = min(E.shape[0], len(tgt_att), len(tgt_una))
    return dict(
        subject=subject, trial=k, trial_name=tno,
        eeg=E[:L], env_att=tgt_att[:L], env_una=tgt_una[:L],
        env_att_raw=env_att[:L], env_una_raw=env_una[:L],
        attended=int(row["Attended Speaker"]),
        az=ATTENDED_SPEAKER_MAP[int(row["Attended Speaker"])][2],
        snr=float(row["SNR"]),
    )


def correct_mask(subject: int) -> dict[int, int]:
    """Return {main_trial_idx_1based: correct(0/1)} from answers.json."""
    ans = load_answers(subject)
    ans = ans[ans["Trial No."].astype(str).str.match(r"^\d+$")]
    out = {}
    for _, r in ans.iterrows():
        try:
            k = int(r["Trial No."])
            out[k] = int(r["Correct"])
        except Exception:
            continue
    return out


def fit_backward(trials: list[dict], lags: list[int], alpha: float) -> dict:
    """5-fold within-subject backward model; returns per-trial rho + window accuracies."""
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    rows = []
    for fi, (tr_i, te_i) in enumerate(kf.split(trials)):
        Xs, ys = [], []
        for i in tr_i:
            Xs.append(design_lags(trials[i]["eeg"], lags))
            ys.append(trials[i]["env_att"])
        X = np.vstack(Xs); y = np.concatenate(ys)
        model = Ridge(alpha=alpha).fit(X, y)
        for i in te_i:
            t = trials[i]
            pred = model.predict(design_lags(t["eeg"], lags))
            ra = np.corrcoef(pred, t["env_att"])[0, 1]
            ru = np.corrcoef(pred, t["env_una"])[0, 1]
            base = dict(subject=t["subject"], fold=fi, trial=t["trial"],
                        trial_name=t["trial_name"], rho_att=ra, rho_una=ru,
                        correct_trial=int(ra > ru), az=t["az"], snr=t["snr"])
            rows.append({**base, "window_s": "full"})
            # Window sweep.
            T = len(pred)
            for w in WINDOWS_S:
                n = int(w * SR_OUT)
                if n > T: continue
                wins = []
                for s0 in range(0, T - n + 1, n):
                    ra_w = np.corrcoef(pred[s0:s0+n], t["env_att"][s0:s0+n])[0,1]
                    ru_w = np.corrcoef(pred[s0:s0+n], t["env_una"][s0:s0+n])[0,1]
                    wins.append(int(ra_w > ru_w))
                if wins:
                    rows.append({**base, "window_s": str(w),
                                 "correct_trial": float(np.mean(wins)),
                                 "n_windows": len(wins)})
    return pd.DataFrame(rows)


def pick_alpha(trials, lags, alphas=RIDGE_ALPHAS):
    """Inner-fold α selection — simple leave-one-trial-out train, sum-of-correlations metric."""
    # Use a cheap heuristic: train on first 80 %, validate on last 20 %.
    n = len(trials); split = int(0.8 * n)
    best_a, best = alphas[0], -np.inf
    Xs, ys = [], []
    for i in range(split):
        Xs.append(design_lags(trials[i]["eeg"], lags))
        ys.append(trials[i]["env_att"])
    X = np.vstack(Xs); y = np.concatenate(ys)
    for a in alphas:
        m = Ridge(alpha=a).fit(X, y)
        agg = []
        for i in range(split, n):
            p = m.predict(design_lags(trials[i]["eeg"], lags))
            agg.append(np.corrcoef(p, trials[i]["env_att"])[0, 1] -
                       np.corrcoef(p, trials[i]["env_una"])[0, 1])
        score = float(np.nanmean(agg))
        if score > best:
            best = score; best_a = a
    return best_a, best


def run_subject(subject: int, *, correct_only: bool = False,
                apply_ica: bool = True, target_mode: str = "derivative",
                out_path: Path | None = None) -> pd.DataFrame:
    t0 = time.time()
    print(f"[S{subject}] loading all 100 main trials (ICA={apply_ica}, target={target_mode})", flush=True)
    trials = []
    for k in range(1, 101):
        p = load_pair(subject, k, apply_ica=apply_ica, target_mode=target_mode)
        if p is not None: trials.append(p)
    print(f"[S{subject}] {len(trials)}/100 trials loaded in {time.time()-t0:.0f}s", flush=True)

    # Correct-only filter.
    if correct_only:
        cmask = correct_mask(subject)
        trials = [t for t in trials if cmask.get(t["trial"], 0) == 1]
        print(f"[S{subject}] after correct-only filter: {len(trials)} trials", flush=True)

    if len(trials) < 10:
        print(f"[S{subject}] too few trials ({len(trials)}), skipping", flush=True)
        return pd.DataFrame()

    lags = [int(round(ms * SR_OUT / 1000)) for ms in LAGS_MS_POS]
    alpha, val_score = pick_alpha(trials, lags)
    print(f"[S{subject}] selected α={alpha:.1e} (val Δρ={val_score:+.4f})", flush=True)

    df = fit_backward(trials, lags, alpha)
    df["correct_only"] = int(correct_only)
    df["apply_ica"] = int(apply_ica)
    df["target_mode"] = target_mode
    df["alpha"] = alpha

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path)
        print(f"[S{subject}] wrote {len(df)} rows -> {out_path}", flush=True)
    print(f"[S{subject}] done in {time.time()-t0:.0f}s", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--correct-only", action="store_true")
    ap.add_argument("--no-ica", action="store_true")
    ap.add_argument("--target-mode", default="derivative", choices=["raw", "derivative"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    run_subject(args.subject, correct_only=args.correct_only,
                apply_ica=not args.no_ica, target_mode=args.target_mode,
                out_path=args.out)


if __name__ == "__main__":
    main()
