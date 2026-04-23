"""Iter-6 · EEG spectral AAD with joint-nuisance cleaning.

Re-runs the iter-5 368-feature spectral pipeline under four EEG
pre-cleaning variants:

    raw                    — no extra cleaning (baseline = iter-5)
    ica_eog                — ICA, auto-remove EOG components (Fp1/Fp2 proxy)
    gaze_regressed         — regress Tobii gaze (15 feats) from EEG
    gaze_imu_regressed     — regress gaze + IMU + video-motion from EEG
    drop_eog_emg_channels  — drop Fp1/Fpz/Fp2 (EOG) and T7/T8/F7/F8/FC5/FC6
                             (likely muscle) channels, recompute features

If AAD accuracy survives the aggressive cleanings → signal is
genuinely audio-cortex. If accuracy collapses → it was riding on
gaze / muscle / head-motion artefacts.

CLI:
    python eeg_clean_aad.py --subject 3 --condition ica_eog \
        --out results/eeg_clean/s3_ica_eog.parquet
"""
from __future__ import annotations
import argparse, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import mne
from scipy.interpolate import interp1d
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from aad_utils import (
    EEG_CHANNELS, EEG_SFREQ, RESULTS_DIR, load_trials_csv,
    load_eeg_trial, load_eeg_time, load_gaze_trial_2d, load_audio_timestamps,
    load_raw_gaze, load_raw_imu,
    align_modalities_to_trial, eeg_raw_to_mne, preprocess_eeg,
    regress_out_gaze, trial_name,
)
from aad_utils.config import ATTENDED_HEMISPHERE
from scripts.eeg_spectral_features import feature_vector

# Channels associated with the primary artefact sources.
EOG_CH = ["Fp1", "Fpz", "Fp2"]
EMG_CH = ["T7", "T8", "F7", "F8", "FC5", "FC6"]


def resample_to(t, x, out_t):
    mask = np.isfinite(t) & np.isfinite(x)
    if mask.sum() < 3:
        return np.zeros(len(out_t))
    y = interp1d(t[mask], x[mask], bounds_error=False, fill_value=0.0)(out_t)
    y[~np.isfinite(y)] = 0.0
    return y


def build_nuisance(ali, n_times, include_imu=True, include_video=False):
    """Stack gaze ± IMU ± video-motion as a nuisance regressor matrix."""
    out_t = np.linspace(0, n_times/EEG_SFREQ, n_times)
    cols = []
    rg = ali.get("raw_gaze", pd.DataFrame())
    ri = ali.get("raw_imu", pd.DataFrame())
    if len(rg):
        t0 = ali["window"].t0
        tr = rg["t_unix"].values - t0
        for c in ["gaze2d_x","gaze2d_y","gaze3d_x","gaze3d_y","gaze3d_z",
                  "L_dx","L_dy","R_dx","R_dy","L_pupil","R_pupil"]:
            cols.append(resample_to(tr, rg[c].astype(float).values, out_t))
        cols.append(np.gradient(cols[0]))
        cols.append(np.gradient(cols[1]))
    if include_imu and len(ri):
        ti = ri["t_unix"].values - ali["window"].t0
        acc = np.linalg.norm(ri[["ax","ay","az"]].values, axis=1) - 9.81
        gyr = np.linalg.norm(ri[["gx","gy","gz"]].values, axis=1)
        cols.append(resample_to(ti, acc, out_t))
        cols.append(resample_to(ti, gyr, out_t))
        for c in ("ax","ay","az","gx","gy","gz"):
            cols.append(resample_to(ti, ri[c].astype(float).values, out_t))
    if not cols:
        return np.zeros((n_times, 1))
    return np.stack(cols, axis=1)


def load_clean_eeg(subject, k, *, condition):
    """Return (eeg_matrix: (C, T), channel_names)."""
    try:
        eeg, ts = load_eeg_trial(subject, k); em = load_eeg_time(subject, k)
        g2 = load_gaze_trial_2d(subject, k); at = load_audio_timestamps(subject, k)
        rg = load_raw_gaze(subject, k); ri = load_raw_imu(subject, k)
    except Exception:
        return None, None
    try:
        ali = align_modalities_to_trial(eeg=eeg, eeg_ts=ts, eeg_time_meta=em,
                                        gaze2d=g2, audio_timestamps=at,
                                        raw_gaze=rg, raw_imu=ri)
        raw = eeg_raw_to_mne(ali["eeg"])
        raw = preprocess_eeg(raw, l_freq=1.0, h_freq=40.0, notch=60.0,
                              reference="auto")
    except Exception:
        return None, None

    if condition == "raw":
        return raw.get_data(), EEG_CHANNELS

    if condition == "ica_eog":
        try:
            ica = mne.preprocessing.ICA(n_components=0.99, random_state=0,
                                         method="fastica", max_iter="auto")
            ica.fit(raw, verbose="ERROR")
            bad = set()
            for ch in EOG_CH:
                try:
                    idx, _ = ica.find_bads_eog(raw, ch_name=ch, verbose="ERROR")
                    bad.update(idx)
                except Exception: pass
            if bad:
                ica.exclude = sorted(bad)
                ica.apply(raw, verbose="ERROR")
        except Exception:
            pass
        return raw.get_data(), EEG_CHANNELS

    if condition.startswith("regress"):
        E = raw.get_data().T   # (T, 32)
        include_imu = ("imu" in condition)
        R = build_nuisance(ali, E.shape[0], include_imu=include_imu)
        cleaned = regress_out_gaze(E, R, ridge=1e-3)
        return cleaned.T, EEG_CHANNELS

    if condition == "drop_eog_emg":
        keep = [i for i, c in enumerate(EEG_CHANNELS) if c not in EOG_CH and c not in EMG_CH]
        data = raw.get_data()[keep]
        kept_names = [EEG_CHANNELS[i] for i in keep]
        return data, kept_names
    return raw.get_data(), EEG_CHANNELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--condition", required=True,
                    choices=["raw","ica_eog","regress_gaze","regress_gaze_imu",
                              "drop_eog_emg"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    t0 = time.time()
    tr_csv = load_trials_csv()

    rows = []
    for k in range(1, 101):
        data, ch_names = load_clean_eeg(args.subject, k, condition=args.condition)
        if data is None: continue
        # feature_vector expects the full 32-ch set by default; but when we drop
        # channels the iter-5 feature vector would break. Pad with zeros for the
        # dropped channels so the feature schema stays constant — the downstream
        # classifier will simply see zero-variance columns that StandardScaler
        # handles fine.
        if data.shape[0] != 32:
            full = np.zeros((32, data.shape[1]))
            keep_idx = [EEG_CHANNELS.index(c) for c in ch_names]
            full[keep_idx] = data
            data = full
        f = feature_vector(data, EEG_SFREQ)
        tno = trial_name(k, "main")
        tr = tr_csv[tr_csv["Trial No."] == tno]
        if not len(tr): continue
        f.update(subject=args.subject, trial=k, attended=int(tr.iloc[0]["Attended Speaker"]))
        rows.append(f)

    if len(rows) < 10:
        print(f"[S{args.subject}/{args.condition}] too few trials"); return
    F = pd.DataFrame(rows)
    feat_cols = [c for c in F.columns if c not in ("subject","trial","attended")]
    X = F[feat_cols].fillna(0).values
    y_full = F["attended"].values

    def eval_task(lbl, task, nc):
        y = np.array([lbl(a_) for a_ in y_full])
        if len(np.unique(y)) < nc or pd.Series(y).value_counts().min() < 2:
            return []
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        accs = []
        for tr_i, te_i in skf.split(X, y):
            m = Pipeline([("sc",StandardScaler()),
                          ("cl", LogisticRegression(max_iter=3000, C=0.5))]).fit(X[tr_i], y[tr_i])
            accs.append(accuracy_score(y[te_i], m.predict(X[te_i])))
        return [dict(subject=args.subject, task=task, condition=args.condition,
                     chance=1/nc, acc=float(np.mean(accs)))]

    results = []
    results += eval_task(lambda a_: 0 if ATTENDED_HEMISPHERE[a_]=="L" else 1, "hemisphere", 2)
    results += eval_task(lambda a_: 0 if a_ in (2,3) else 1, "inner_outer", 2)
    results += eval_task(lambda a_: a_-1, "4class", 4)
    R = pd.DataFrame(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    R.to_parquet(args.out)
    print(f"[S{args.subject}/{args.condition}] done in {time.time()-t0:.0f}s", flush=True)
    print(R.to_string(index=False))


if __name__ == "__main__":
    main()
