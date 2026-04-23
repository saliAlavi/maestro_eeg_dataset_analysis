"""Leave-one-subject-out CCA mel-28 decoder.

For the held-out test subject S, train CCA on all 15 other subjects'
concatenated trials, then test on S's 100 trials.

CLI:
    python scripts/aad_loso.py --test-subject 3 \
        --out results/loso_cca_mel28/s3.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA

from aad_utils import (
    load_trials_csv, load_eeg_trial, load_eeg_time,
    load_gaze_trial_2d, load_audio_timestamps,
    align_modalities_to_trial, eeg_raw_to_mne, preprocess_eeg,
    gammatone_envelope, load_audio_file, trial_name,
)
from aad_utils.config import ATTENDED_SPEAKER_MAP

SR_OUT = 64.0
LAGS_MS = np.arange(-200, 501, 50)  # halved lag density for memory
SUBSAMPLE = 2  # take every Nth time step to cap X size


def design_lags(X, lags):
    T, C = X.shape
    out = np.zeros((T, C * len(lags)))
    for i, lag in enumerate(lags):
        if lag >= 0:
            out[:, i*C:(i+1)*C] = np.vstack([np.zeros((lag, C)), X[:T-lag]])
        else:
            out[:, i*C:(i+1)*C] = np.vstack([X[-lag:], np.zeros((-lag, C))])
    return out


def load_one(subject, k):
    try:
        eeg, ts = load_eeg_trial(subject, k)
        em = load_eeg_time(subject, k)
        g2 = load_gaze_trial_2d(subject, k)
        at = load_audio_timestamps(subject, k)
        ali = align_modalities_to_trial(eeg=eeg, eeg_ts=ts, eeg_time_meta=em,
                                        gaze2d=g2, audio_timestamps=at)
        raw = eeg_raw_to_mne(ali["eeg"])
        raw = preprocess_eeg(raw, l_freq=1.0, h_freq=9.0, notch=60.0,
                             reference="auto", apply_ica=False)
        raw.resample(SR_OUT, verbose="ERROR")
        eeg_mat = raw.get_data().T

        tno = trial_name(k, "main")
        tr = load_trials_csv()
        row = tr[tr["Trial No."] == tno]
        if not len(row):
            return None
        row = row.iloc[0]
        att_dev = "Device-1" if int(row["Attended Speaker"]) in (1, 2) else "Device-2"
        una_dev = "Device-2" if att_dev == "Device-1" else "Device-1"
        a_att, sr = load_audio_file(row[att_dev])
        a_una, _ = load_audio_file(row[una_dev])
        mel_att = gammatone_envelope(a_att, sr, sr_out=SR_OUT)
        mel_una = gammatone_envelope(a_una, sr, sr_out=SR_OUT)
        L = min(eeg_mat.shape[0], len(mel_att), len(mel_una))
        return dict(subject=subject, trial=k, trial_name=tno,
                    eeg=eeg_mat[:L], att=mel_att[:L], una=mel_una[:L],
                    attended=int(row["Attended Speaker"]),
                    az=ATTENDED_SPEAKER_MAP[int(row["Attended Speaker"])][2],
                    snr=float(row["SNR"]))
    except Exception as exc:
        print(f"  !! S{subject} T{k} skipped: {exc}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-subject", type=int, required=True)
    ap.add_argument("--n-components", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    all_subj = list(range(1, 17))
    train_subj = [s for s in all_subj if s != args.test_subject]
    print(f"TEST={args.test_subject}  TRAIN={train_subj}")

    t0 = time.time()
    lags = [int(round(ms * SR_OUT / 1000)) for ms in LAGS_MS]

    # STREAM training data: build X, Y incrementally, free per-trial memory.
    Xs, Ys = [], []
    for s in train_subj:
        loaded = 0
        for k in range(1, 101):
            p = load_one(s, k)
            if p is None:
                continue
            Xi = design_lags(p["eeg"], lags)[::SUBSAMPLE].astype(np.float32)
            Yi = p["att"][::SUBSAMPLE].astype(np.float32)
            L = min(Xi.shape[0], Yi.shape[0])
            Xs.append(Xi[:L]); Ys.append(Yi[:L])
            loaded += 1
            del p, Xi, Yi
        print(f"  [train S{s}] {loaded}/100  Xs={sum(x.shape[0] for x in Xs)}", flush=True)

    # Test: keep dicts (only 100 trials, cheap)
    test_trials = []
    for k in range(1, 101):
        p = load_one(args.test_subject, k)
        if p is not None:
            test_trials.append(p)
    print(f"  [test S{args.test_subject}] {len(test_trials)}/100", flush=True)
    print(f"  loaded in {time.time()-t0:.0f}s", flush=True)

    X = np.vstack(Xs); Y = np.vstack(Ys)
    del Xs, Ys

    n_eff = max(1, min(args.n_components, X.shape[1], Y.shape[1], len(X)-1))
    print(f"  fitting CCA  X={X.shape}  Y={Y.shape}  n_comp={n_eff}")
    cca = CCA(n_components=n_eff, max_iter=200).fit(X, Y)

    rows = []
    for t in test_trials:
        Xt = design_lags(t["eeg"], lags)
        Xc, Ytt = cca.transform(Xt, t["att"])
        _, Yun = cca.transform(Xt, t["una"])
        if Xc.ndim == 1:
            Xc = Xc[:, None]
            Ytt = Ytt[:, None]
            Yun = Yun[:, None]
        ra = float(np.nanmean(
            [np.corrcoef(Xc[:, c], Ytt[:, c])[0, 1] for c in range(Xc.shape[1])]
        ))
        ru = float(np.nanmean(
            [np.corrcoef(Xc[:, c], Yun[:, c])[0, 1] for c in range(Xc.shape[1])]
        ))
        rows.append(dict(
            test_subject=args.test_subject, trial=t["trial"],
            trial_name=t["trial_name"], rho_att=ra, rho_una=ru,
            correct_trial=int(ra > ru), az=t["az"], snr=t["snr"],
            attended=t["attended"],
        ))

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"  wrote {out} ({len(df)} rows)  acc={df['correct_trial'].mean():.3f}")
    print(f"  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
