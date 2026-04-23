"""Iteration 3 AAD pipeline — proper α-CV, wider lags, multi-band options.

Changes over iter-2:
    - Nested 5-fold inner α-CV on a wider grid.
    - Lags -200..+500 ms (was -100..+400).
    - `--features`: one of {broadband, split_delta_theta, mel28, cca_mel}.
        broadband         : Hilbert-derivative at 1-9 Hz (iter-2 baseline).
        split_delta_theta : concat 1-4Hz EEG lags + 4-8Hz EEG lags as regressors.
        mel28             : 28-band log-mel envelope; forward model averaging
                            backward reconstruction across bands.
        cca_mel           : CCA on (EEG lags) x (mel envelope).
    - Per-trial target can be raw or derivative via --target.

CLI:
    python aad_v3.py --subject 3 --features broadband --out results/aad_v3/s3_broadband.parquet
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
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from aad_utils import (
    EEG_CHANNELS, RESULTS_DIR, load_trials_csv,
    load_eeg_trial, load_eeg_time, load_gaze_trial_2d, load_audio_timestamps,
    align_modalities_to_trial, eeg_raw_to_mne, preprocess_eeg, audio_envelope,
    gammatone_envelope, load_audio_file, trial_name,
)
from aad_utils.config import ATTENDED_SPEAKER_MAP

SR_OUT = 64.0
LAGS_MS = np.arange(-200, 501, 25)
RIDGE_GRID = np.logspace(-1, 6, 12)
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


def envelope_target(env: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return env
    if mode == "derivative":
        d = np.diff(env, prepend=env[0])
        return np.maximum(d, 0)
    raise ValueError(mode)


def load_eeg_band(subject, k, l_freq, h_freq, apply_ica=False):
    eeg, ts = load_eeg_trial(subject, k); em = load_eeg_time(subject, k)
    g2 = load_gaze_trial_2d(subject, k); at = load_audio_timestamps(subject, k)
    ali = align_modalities_to_trial(eeg=eeg, eeg_ts=ts, eeg_time_meta=em,
                                    gaze2d=g2, audio_timestamps=at)
    raw = eeg_raw_to_mne(ali["eeg"])
    raw = preprocess_eeg(raw, l_freq=l_freq, h_freq=h_freq, notch=60.0,
                         reference="auto", apply_ica=apply_ica)
    raw.resample(SR_OUT, verbose="ERROR")
    return raw.get_data().T  # (T, 32)


def load_pair(subject, k, *, features, target, apply_ica=False):
    try:
        if features == "split_delta_theta":
            eeg_delta = load_eeg_band(subject, k, 1.0, 4.0, apply_ica=apply_ica)
            eeg_theta = load_eeg_band(subject, k, 4.0, 8.0, apply_ica=apply_ica)
            L = min(len(eeg_delta), len(eeg_theta))
            eeg = np.concatenate([eeg_delta[:L], eeg_theta[:L]], axis=1)
        else:
            eeg = load_eeg_band(subject, k, 1.0, 9.0, apply_ica=apply_ica)
    except (FileNotFoundError, EOFError, ValueError, OSError, Exception):
        return None

    tno = trial_name(k, "main")
    tr = load_trials_csv()
    row = tr[tr["Trial No."] == tno]
    if not len(row): return None
    row = row.iloc[0]
    att_dev = "Device-1" if int(row["Attended Speaker"]) in (1, 2) else "Device-2"
    una_dev = "Device-2" if att_dev == "Device-1" else "Device-1"
    a_att, sr = load_audio_file(row[att_dev])
    a_una, _ = load_audio_file(row[una_dev])

    if features in {"mel28", "cca_mel"}:
        mel_att = gammatone_envelope(a_att, sr, sr_out=SR_OUT)
        mel_una = gammatone_envelope(a_una, sr, sr_out=SR_OUT)
        L = min(eeg.shape[0], len(mel_att), len(mel_una))
        return dict(subject=subject, trial=k, trial_name=tno,
                    eeg=eeg[:L], att=mel_att[:L], una=mel_una[:L],
                    attended=int(row["Attended Speaker"]),
                    az=ATTENDED_SPEAKER_MAP[int(row["Attended Speaker"])][2],
                    snr=float(row["SNR"]), features=features)

    env_att = envelope_target(audio_envelope(a_att, sr, sr_out=SR_OUT), target)
    env_una = envelope_target(audio_envelope(a_una, sr, sr_out=SR_OUT), target)
    L = min(eeg.shape[0], len(env_att), len(env_una))
    return dict(subject=subject, trial=k, trial_name=tno,
                eeg=eeg[:L], att=env_att[:L], una=env_una[:L],
                attended=int(row["Attended Speaker"]),
                az=ATTENDED_SPEAKER_MAP[int(row["Attended Speaker"])][2],
                snr=float(row["SNR"]), features=features)


def nested_alpha(trials, lags, alphas=RIDGE_GRID):
    # Inner 5-fold CV on the training set to pick α by mean Δρ on validation.
    best_a, best_score = alphas[0], -np.inf
    kf = KFold(5, shuffle=True, random_state=1)
    for a in alphas:
        scores = []
        for tr_i, va_i in kf.split(trials):
            Xs, ys = [], []
            for i in tr_i:
                Xs.append(design_lags(trials[i]["eeg"], lags))
                ys.append(trials[i]["att"].mean(axis=1) if trials[i]["att"].ndim == 2
                          else trials[i]["att"])
            X = np.vstack(Xs); y = np.concatenate(ys)
            m = Ridge(alpha=a).fit(X, y)
            for i in va_i:
                t = trials[i]
                pred = m.predict(design_lags(t["eeg"], lags))
                att = t["att"].mean(axis=1) if t["att"].ndim == 2 else t["att"]
                una = t["una"].mean(axis=1) if t["una"].ndim == 2 else t["una"]
                scores.append(np.corrcoef(pred, att)[0,1] - np.corrcoef(pred, una)[0,1])
        s = float(np.nanmean(scores))
        if s > best_score:
            best_score = s; best_a = a
    return best_a, best_score


def fit_backward_ridge(trials, lags, alpha):
    kf = KFold(N_FOLDS, shuffle=True, random_state=0)
    rows = []
    for fi, (tr_i, te_i) in enumerate(kf.split(trials)):
        Xs, ys = [], []
        for i in tr_i:
            Xs.append(design_lags(trials[i]["eeg"], lags))
            # For multi-band mel target, reconstruct the mean band envelope.
            y = trials[i]["att"].mean(axis=1) if trials[i]["att"].ndim == 2 else trials[i]["att"]
            ys.append(y)
        X = np.vstack(Xs); y = np.concatenate(ys)
        model = Ridge(alpha=alpha).fit(X, y)
        for i in te_i:
            t = trials[i]
            pred = model.predict(design_lags(t["eeg"], lags))
            att = t["att"].mean(axis=1) if t["att"].ndim == 2 else t["att"]
            una = t["una"].mean(axis=1) if t["una"].ndim == 2 else t["una"]
            ra = np.corrcoef(pred, att)[0,1]; ru = np.corrcoef(pred, una)[0,1]
            rows.append(dict(subject=t["subject"], fold=fi, trial=t["trial"],
                             trial_name=t["trial_name"], rho_att=ra, rho_una=ru,
                             correct_trial=int(ra > ru), az=t["az"], snr=t["snr"],
                             window_s="full"))
            # Window sweep
            T = len(pred)
            for w in WINDOWS_S:
                n = int(w * SR_OUT)
                if n > T: continue
                wins = []
                for s0 in range(0, T - n + 1, n):
                    wa = np.corrcoef(pred[s0:s0+n], att[s0:s0+n])[0,1]
                    wu = np.corrcoef(pred[s0:s0+n], una[s0:s0+n])[0,1]
                    wins.append(int(wa > wu))
                if wins:
                    rows.append(dict(subject=t["subject"], fold=fi, trial=t["trial"],
                                     trial_name=t["trial_name"], rho_att=np.nan, rho_una=np.nan,
                                     correct_trial=float(np.mean(wins)), az=t["az"],
                                     snr=t["snr"], window_s=str(w)))
    return pd.DataFrame(rows)


def fit_cca(trials, lags, n_components=5):
    # Canonical CCA: EEG(lags) ↔ mel28 envelope. Classification per trial
    # by whether attended envelope yields higher CC sum vs unattended.
    kf = KFold(N_FOLDS, shuffle=True, random_state=0)
    rows = []
    for fi, (tr_i, te_i) in enumerate(kf.split(trials)):
        Xs, Ys = [], []
        for i in tr_i:
            Xs.append(design_lags(trials[i]["eeg"], lags))
            Ys.append(trials[i]["att"])
        X = np.vstack(Xs); Y = np.vstack(Ys)
        n_eff = max(1, min(n_components, X.shape[1], Y.shape[1], len(X)-1))
        try:
            cca = CCA(n_components=n_eff, max_iter=200).fit(X, Y)
        except Exception:
            continue
        for i in te_i:
            t = trials[i]
            Xt = design_lags(t["eeg"], lags)
            Xc, Ytt = cca.transform(Xt, t["att"])
            _, Yun = cca.transform(Xt, t["una"])
            if Xc.ndim == 1: Xc = Xc[:, None]
            if Ytt.ndim == 1: Ytt = Ytt[:, None]
            if Yun.ndim == 1: Yun = Yun[:, None]
            ra = float(np.nanmean([np.corrcoef(Xc[:,c], Ytt[:,c])[0,1] for c in range(Xc.shape[1])]))
            ru = float(np.nanmean([np.corrcoef(Xc[:,c], Yun[:,c])[0,1] for c in range(Xc.shape[1])]))
            rows.append(dict(subject=t["subject"], fold=fi, trial=t["trial"],
                             trial_name=t["trial_name"], rho_att=ra, rho_una=ru,
                             correct_trial=int(ra > ru), az=t["az"], snr=t["snr"],
                             window_s="full"))
    return pd.DataFrame(rows)


def run_subject(subject, *, features, target, apply_ica=False, out_path=None):
    t0 = time.time()
    print(f"[S{subject}] features={features} target={target} ica={apply_ica}", flush=True)
    trials = []
    for k in range(1, 101):
        p = load_pair(subject, k, features=features, target=target, apply_ica=apply_ica)
        if p is not None: trials.append(p)
    print(f"[S{subject}] loaded {len(trials)}/100 trials in {time.time()-t0:.0f}s", flush=True)
    if len(trials) < 10:
        print(f"[S{subject}] too few trials"); return pd.DataFrame()

    lags = [int(round(ms * SR_OUT / 1000)) for ms in LAGS_MS]

    if features == "cca_mel":
        df = fit_cca(trials, lags)
    else:
        alpha, val = nested_alpha(trials, lags)
        print(f"[S{subject}] selected α={alpha:.1e} (inner CV Δρ={val:+.4f})", flush=True)
        df = fit_backward_ridge(trials, lags, alpha)
        df["alpha"] = alpha

    df["features"] = features
    df["target"] = target
    df["apply_ica"] = int(apply_ica)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path)
        print(f"[S{subject}] wrote {len(df)} rows -> {out_path}", flush=True)
    print(f"[S{subject}] done in {time.time()-t0:.0f}s", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--features", choices=["broadband", "split_delta_theta", "mel28", "cca_mel"],
                    default="broadband")
    ap.add_argument("--target", choices=["raw", "derivative"], default="derivative")
    ap.add_argument("--ica", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    run_subject(a.subject, features=a.features, target=a.target,
                apply_ica=a.ica, out_path=a.out)


if __name__ == "__main__":
    main()
