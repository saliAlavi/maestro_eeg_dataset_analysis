"""Gaze-residualised EEG AAD.

For each trial, channel-wise ridge-regress gaze (scene-projected gaze2d_x/y,
gaze3d_x/y/z, and IMU gyro/accel magnitudes — plus velocities) out of the EEG
*before* running the CCA mel-28 decoder. If AAD accuracy drops dramatically,
most of the apparent EEG signal was actually an oculomotor artefact; if it
stays the same, EEG carries independent information about the attended
stream.

We save two per-subject result rows:
    - baseline (no gaze regression)
    - residualised (gaze regressed out)
Same backbone (CCA mel-28), same inner-CV / ridge α.

CLI:
    python aad_gaze_residualized.py --subject 3 --out results/gaze_resid/s3.parquet
"""
from __future__ import annotations
import argparse, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import KFold

from aad_utils import (
    EEG_CHANNELS, RESULTS_DIR, load_trials_csv,
    load_eeg_trial, load_eeg_time, load_gaze_trial_2d, load_audio_timestamps,
    load_raw_gaze, load_raw_imu,
    align_modalities_to_trial, eeg_raw_to_mne, preprocess_eeg,
    gammatone_envelope, load_audio_file, trial_name, regress_out_gaze,
)
from aad_utils.config import ATTENDED_HEMISPHERE

SR_OUT = 64.0
LAGS_MS = np.arange(-200, 501, 25)
N_FOLDS = 5


def resample_to(t, x, out_t):
    mask = np.isfinite(t) & np.isfinite(x)
    if mask.sum() < 3:
        return np.zeros(len(out_t))
    y = interp1d(t[mask], x[mask], bounds_error=False, fill_value=0.0)(out_t)
    y[~np.isfinite(y)] = 0.0
    return y


def build_gaze_regressors(ali, n_times, sfreq=SR_OUT):
    """Produce a (n_times, k) regressor matrix of Tobii gaze + IMU features,
    temporally interpolated to the EEG time grid."""
    out_t = np.linspace(0, n_times/sfreq, n_times)
    rg = ali.get("raw_gaze", pd.DataFrame())
    ri = ali.get("raw_imu",  pd.DataFrame())
    cols = []
    t0 = ali["window"].t0
    if len(rg):
        tr = rg["t_unix"].values - t0
        for c in ["gaze2d_x", "gaze2d_y", "gaze3d_x", "gaze3d_y", "gaze3d_z",
                  "L_dx", "L_dy", "R_dx", "R_dy", "L_pupil", "R_pupil"]:
            cols.append(resample_to(tr, rg[c].astype(float).values, out_t))
        # Velocities
        cols.append(np.gradient(cols[0]))  # gaze2d_x velocity
        cols.append(np.gradient(cols[1]))  # gaze2d_y velocity
    if len(ri):
        ti = ri["t_unix"].values - t0
        cols.append(resample_to(ti,
                    np.linalg.norm(ri[["ax","ay","az"]].values, axis=1) - 9.81, out_t))
        cols.append(resample_to(ti,
                    np.linalg.norm(ri[["gx","gy","gz"]].values, axis=1), out_t))
    if not cols:
        return np.zeros((n_times, 1))
    return np.stack(cols, axis=1)


def design_lags(X, lags):
    T, C = X.shape
    out = np.zeros((T, C * len(lags)))
    for i, lag in enumerate(lags):
        if lag >= 0:
            out[:, i*C:(i+1)*C] = np.vstack([np.zeros((lag, C)), X[:T-lag]])
        else:
            out[:, i*C:(i+1)*C] = np.vstack([X[-lag:], np.zeros((-lag, C))])
    return out


def load_trial_pair(subject, k, residualise):
    try:
        eeg, ts = load_eeg_trial(subject, k); em = load_eeg_time(subject, k)
        g2 = load_gaze_trial_2d(subject, k); at = load_audio_timestamps(subject, k)
        rg = load_raw_gaze(subject, k); ri = load_raw_imu(subject, k)
    except Exception:
        return None
    try:
        ali = align_modalities_to_trial(eeg=eeg, eeg_ts=ts, eeg_time_meta=em,
                                        gaze2d=g2, audio_timestamps=at,
                                        raw_gaze=rg, raw_imu=ri)
        raw = eeg_raw_to_mne(ali["eeg"])
        raw = preprocess_eeg(raw, l_freq=1.0, h_freq=9.0, notch=60.0,
                              reference="auto")
        raw.resample(SR_OUT, verbose="ERROR")
    except Exception:
        return None
    E = raw.get_data().T  # (T, 32)
    if residualise:
        R = build_gaze_regressors(ali, E.shape[0])
        E = regress_out_gaze(E, R, ridge=1e-3)

    tno = trial_name(k, "main")
    tr = load_trials_csv()
    row = tr[tr["Trial No."] == tno]
    if not len(row): return None
    row = row.iloc[0]
    att = int(row["Attended Speaker"])
    att_dev = "Device-1" if att in (1, 2) else "Device-2"
    una_dev = "Device-2" if att_dev == "Device-1" else "Device-1"
    a_att, sr = load_audio_file(row[att_dev])
    a_una, _  = load_audio_file(row[una_dev])
    mel_att = gammatone_envelope(a_att, sr, sr_out=SR_OUT)
    mel_una = gammatone_envelope(a_una, sr, sr_out=SR_OUT)
    L = min(E.shape[0], len(mel_att), len(mel_una))
    return dict(subject=subject, trial=k, trial_name=tno,
                eeg=E[:L], att=mel_att[:L], una=mel_una[:L],
                attended=att, snr=float(row["SNR"]))


def fit_cca_aad(trials, n_components=3):
    lags = [int(round(ms * SR_OUT / 1000)) for ms in LAGS_MS]
    rows = []
    kf = KFold(N_FOLDS, shuffle=True, random_state=0)
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
            Xc, Ya = cca.transform(Xt, t["att"])
            _, Yu = cca.transform(Xt, t["una"])
            if Xc.ndim == 1: Xc = Xc[:, None]
            if Ya.ndim == 1: Ya = Ya[:, None]
            if Yu.ndim == 1: Yu = Yu[:, None]
            ra = float(np.nanmean([np.corrcoef(Xc[:, c], Ya[:, c])[0, 1] for c in range(Xc.shape[1])]))
            ru = float(np.nanmean([np.corrcoef(Xc[:, c], Yu[:, c])[0, 1] for c in range(Xc.shape[1])]))
            rows.append(dict(subject=t["subject"], fold=fi, trial=t["trial"],
                             rho_att=ra, rho_una=ru, correct_trial=int(ra > ru),
                             attended=t["attended"], snr=t["snr"]))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    t0 = time.time()
    results = []
    for residualise, label in [(False, "baseline"), (True, "gaze_residualised")]:
        print(f"[S{a.subject}] loading trials ({label})", flush=True)
        trials = []
        for k in range(1, 101):
            p = load_trial_pair(a.subject, k, residualise=residualise)
            if p is not None: trials.append(p)
        if len(trials) < 10:
            print(f"[S{a.subject}] too few trials ({label})"); continue
        df = fit_cca_aad(trials)
        df["condition"] = label
        results.append(df)
        print(f"[S{a.subject}] {label}: per-trial acc = {df['correct_trial'].mean():.3f}  ({time.time()-t0:.0f}s)", flush=True)
    if results:
        out = pd.concat(results, ignore_index=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(a.out)


if __name__ == "__main__":
    main()
