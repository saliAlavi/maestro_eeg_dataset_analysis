"""Iter-4 · 4-class speaker-identity EEG decoder (CCA mel-28 backbone).

For each trial the decoder must pick one of 4 attended speakers
(labels 1..4 → Device-1 L/R, Device-2 L/R). We train CCA between EEG lags
and each candidate envelope, then classify a test trial by which candidate's
canonical correlation with the learned EEG projection is highest.

Per-trial the output records:
    correct_trial      1 if top-1 candidate = true attended
    correct_hemisphere 1 if top-1 is in the same hemisphere as attended
    correct_device_pair 1 if same device (≡ hemisphere)
    rho_<k>            CC for candidate k
    pred_label         argmax_k rho_k

Evaluation windows {1, 2, 4, 8, 16, 30}s are reported for the 4-class metric.
"""
from __future__ import annotations
import argparse, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import KFold

from aad_utils import (
    EEG_CHANNELS, RESULTS_DIR, load_trials_csv,
    load_eeg_trial, load_eeg_time, load_gaze_trial_2d, load_audio_timestamps,
    align_modalities_to_trial, eeg_raw_to_mne, preprocess_eeg,
    gammatone_envelope, load_audio_file, trial_name,
)
from aad_utils.config import ATTENDED_HEMISPHERE

SR_OUT = 64.0
LAGS_MS = np.arange(-200, 501, 25)
N_FOLDS = 5
WINDOWS_S = [1, 2, 4, 8, 16, 30]


def design_lags(X, lags):
    T, C = X.shape
    out = np.zeros((T, C * len(lags)))
    for i, lag in enumerate(lags):
        if lag >= 0:
            out[:, i*C:(i+1)*C] = np.vstack([np.zeros((lag, C)), X[:T-lag]])
        else:
            out[:, i*C:(i+1)*C] = np.vstack([X[-lag:], np.zeros((-lag, C))])
    return out


def load_trial_for_4class(subject, k):
    """Load EEG + the 4 candidate envelopes (one per possible attended speaker)."""
    try:
        eeg, ts = load_eeg_trial(subject, k); em = load_eeg_time(subject, k)
        g2 = load_gaze_trial_2d(subject, k); at = load_audio_timestamps(subject, k)
    except (FileNotFoundError, EOFError, ValueError, OSError):
        return None
    try:
        ali = align_modalities_to_trial(eeg=eeg, eeg_ts=ts, eeg_time_meta=em,
                                        gaze2d=g2, audio_timestamps=at)
        raw = eeg_raw_to_mne(ali["eeg"])
        raw = preprocess_eeg(raw, l_freq=1.0, h_freq=9.0, notch=60.0,
                              reference="auto")
        raw.resample(SR_OUT, verbose="ERROR")
    except Exception:
        return None
    E = raw.get_data().T

    tno = trial_name(k, "main")
    tr = load_trials_csv()
    row = tr[tr["Trial No."] == tno]
    if not len(row): return None
    row = row.iloc[0]
    att = int(row["Attended Speaker"])

    # Device-1 stereo file and Device-2 stereo file each supply two candidate
    # envelopes (the device's L channel vs the device's R channel). Since the
    # stored file is mono per-channel identical speech (just weighted), we
    # approximate the per-speaker attended-envelope by weighting the device
    # file with the respective channel power.
    pL_d1 = float(row["Device-1 Left Power"]); pR_d1 = float(row["Device-1 Right Power"])
    pL_d2 = float(row["Device-2 Left Power"]); pR_d2 = float(row["Device-2 Right Power"])

    a1, sr = load_audio_file(row["Device-1"])
    a2, _  = load_audio_file(row["Device-2"])
    # Per-speaker mel-28 envelope with the channel power applied.
    envs = {
        1: gammatone_envelope(a1 * pL_d1, sr, sr_out=SR_OUT),  # D1-L, az=-67.5
        2: gammatone_envelope(a1 * pR_d1, sr, sr_out=SR_OUT),  # D1-R, az=-22.5
        3: gammatone_envelope(a2 * pL_d2, sr, sr_out=SR_OUT),  # D2-L, az=+22.5
        4: gammatone_envelope(a2 * pR_d2, sr, sr_out=SR_OUT),  # D2-R, az=+67.5
    }
    L = min(E.shape[0], *(len(e) for e in envs.values()))
    envs = {k: v[:L] for k, v in envs.items()}
    return dict(subject=subject, trial=k, trial_name=tno,
                eeg=E[:L], envs=envs, attended=att, snr=float(row["SNR"]))


def fit_4class(trials):
    """Train CCA on true-attended EEG↔envelope; test each trial by picking the
    envelope (among 4 candidates) with the highest CC to the learned EEG
    projection.
    """
    lags = [int(round(ms * SR_OUT / 1000)) for ms in LAGS_MS]
    kf = KFold(N_FOLDS, shuffle=True, random_state=0)
    rows = []
    for fi, (tr_i, te_i) in enumerate(kf.split(trials)):
        Xs, Ys = [], []
        for i in tr_i:
            Xs.append(design_lags(trials[i]["eeg"], lags))
            Ys.append(trials[i]["envs"][trials[i]["attended"]])  # (L, 28) 2D
        X = np.vstack(Xs); Y = np.vstack(Ys)  # Y = (n*L, 28)
        n_eff = max(1, min(3, X.shape[1], Y.shape[1], len(X)-1))
        try:
            cca = CCA(n_components=n_eff, max_iter=200).fit(X, Y)
        except Exception:
            continue
        for i in te_i:
            t = trials[i]
            Xt = design_lags(t["eeg"], lags)
            Xc, _ = cca.transform(Xt, t["envs"][t["attended"]])
            if Xc.ndim == 1: Xc = Xc[:, None]
            rho = {}
            for k in (1, 2, 3, 4):
                _, Yk = cca.transform(Xt, t["envs"][k])
                if Yk.ndim == 1: Yk = Yk[:, None]
                rho[k] = float(np.nanmean([np.corrcoef(Xc[:, c], Yk[:, c])[0, 1]
                                            for c in range(Xc.shape[1])]))
            pred = max(rho, key=rho.get)
            hemi_pred = ATTENDED_HEMISPHERE[pred]
            hemi_true = ATTENDED_HEMISPHERE[t["attended"]]
            inner_pred = pred in (2, 3); inner_true = t["attended"] in (2, 3)
            rows.append(dict(
                subject=t["subject"], fold=fi, trial=t["trial"],
                trial_name=t["trial_name"], attended=t["attended"],
                pred_label=pred, snr=t["snr"],
                rho_1=rho[1], rho_2=rho[2], rho_3=rho[3], rho_4=rho[4],
                correct_trial=int(pred == t["attended"]),
                correct_hemisphere=int(hemi_pred == hemi_true),
                correct_inner_outer=int(inner_pred == inner_true),
                window_s="full",
            ))
            # Window sweep on 4-class decision.
            T = Xt.shape[0]
            for w in WINDOWS_S:
                n = int(w * SR_OUT)
                if n > T: continue
                wcorr, whem, wio = [], [], []
                for s0 in range(0, T-n+1, n):
                    Xw = Xc[s0:s0+n]
                    rw = {}
                    for k in (1, 2, 3, 4):
                        _, Yk = cca.transform(Xt[s0:s0+n], t["envs"][k][s0:s0+n])
                        if Yk.ndim == 1: Yk = Yk[:, None]
                        rw[k] = float(np.nanmean([np.corrcoef(Xw[:, c], Yk[:, c])[0, 1]
                                                   for c in range(Xw.shape[1])]))
                    pk = max(rw, key=rw.get)
                    wcorr.append(int(pk == t["attended"]))
                    whem.append(int(ATTENDED_HEMISPHERE[pk] == hemi_true))
                    wio.append(int((pk in (2, 3)) == inner_true))
                if wcorr:
                    rows.append(dict(
                        subject=t["subject"], fold=fi, trial=t["trial"],
                        trial_name=t["trial_name"], attended=t["attended"],
                        pred_label=-1, snr=t["snr"],
                        rho_1=np.nan, rho_2=np.nan, rho_3=np.nan, rho_4=np.nan,
                        correct_trial=float(np.mean(wcorr)),
                        correct_hemisphere=float(np.mean(whem)),
                        correct_inner_outer=float(np.mean(wio)),
                        window_s=str(w),
                    ))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    t0 = time.time()
    print(f"[S{a.subject}] loading 100 main trials", flush=True)
    trials = []
    for k in range(1, 101):
        p = load_trial_for_4class(a.subject, k)
        if p is not None:
            trials.append(p)
    print(f"[S{a.subject}] {len(trials)} trials loaded in {time.time()-t0:.0f}s", flush=True)
    if len(trials) < 10: return
    df = fit_4class(trials)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out)
    full = df[df["window_s"] == "full"]
    print(f"[S{a.subject}] 4-class acc = {full['correct_trial'].mean():.3f}  "
          f"hemisphere = {full['correct_hemisphere'].mean():.3f}  "
          f"inner/outer = {full['correct_inner_outer'].mean():.3f}  "
          f"({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
