"""Iter-5 · classical spectral / connectivity / complexity EEG features per trial.

For every main trial of a subject we extract a single feature vector and then
train per-subject classifiers for the hemisphere / inner-outer / 4-class tasks.
No deep learning. The goal is to establish what is \emph{linearly} learnable
from well-studied EEG features before reaching for deep models.

Feature groups (≈300 features per trial):
    1. Band powers (Welch, 2-s segments) in δ(1-4), θ(4-8), α(8-13), β(13-30),
       lo-γ(30-40), 32 channels × 5 bands = 160.
    2. Log band-power ratios per channel: α/θ, α/β, θ/β = 96.
    3. Alpha lateralisation index (log R/L) for 4 parietal/temporal groupings = 4.
    4. Hjorth mobility & complexity per channel = 64.
    5. Spectral entropy per channel (Shannon) = 32.
    6. 1/f aperiodic slope per channel group (frontal, central, parietal,
       occipital) = 4.
    7. Alpha-band connectivity summaries (MSC & wPLI): within-left,
       within-right, between-hemisphere = 6.
    8. Global field power mean & std = 2.

Per-subject evaluation on three tasks:
    A) hemisphere    (L {1,2} vs R {3,4})
    B) inner-outer   ({2,3} vs {1,4})
    C) 4-class identity
Classifiers: L2-logistic regression + LightGBM, 5-fold stratified CV per
subject. Reports accuracy per task per classifier.

CLI:
    python eeg_spectral_features.py --subject 3 \
        --out results/eeg_spectral/s3.parquet
"""
from __future__ import annotations
import argparse, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.signal import welch, butter, filtfilt, hilbert, coherence
from scipy.stats import linregress
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb

from aad_utils import (
    EEG_CHANNELS, EEG_SFREQ, RESULTS_DIR, load_trials_csv,
    load_eeg_trial, load_eeg_time, load_gaze_trial_2d, load_audio_timestamps,
    align_modalities_to_trial, eeg_raw_to_mne, preprocess_eeg, trial_name,
)
from aad_utils.config import ATTENDED_HEMISPHERE

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "lo_gamma": (30, 40),
}
GROUPS = dict(
    frontal  = ["Fp1","Fpz","Fp2","F7","F3","Fz","F4","F8"],
    central  = ["FC5","FC1","FC2","FC6","C3","Cz","C4"],
    parietal = ["CP5","CP1","CP2","CP6","P3","Pz","P4","P7","P8"],
    occipital= ["POz","O1","Oz","O2"],
)
LEFT  = ["Fp1","F7","F3","FC5","FC1","T7","C3","CP5","CP1","P7","P3","O1"]
RIGHT = ["Fp2","F8","F4","FC6","FC2","T8","C4","CP6","CP2","P8","P4","O2"]
LAT_PAIRS = {
    "parietal":  (["P3","P7","CP1","CP5"], ["P4","P8","CP2","CP6"]),
    "central":   (["C3","FC1","FC5"],      ["C4","FC2","FC6"]),
    "occipital": (["O1"],                  ["O2"]),
    "temporal":  (["T7"],                  ["T8"]),
}


def band_powers(data, sf):
    """Return (n_ch, n_bands) band-power matrix from Welch."""
    f, P = welch(data, fs=sf, nperseg=int(sf*2))
    out = np.zeros((data.shape[0], len(BANDS)))
    for i, (_, (lo, hi)) in enumerate(BANDS.items()):
        mask = (f >= lo) & (f <= hi)
        out[:, i] = P[:, mask].mean(axis=1)
    return out, f, P


def hjorth(data):
    """Return (n_ch, 2) — mobility and complexity per channel."""
    d1 = np.diff(data, axis=1)
    d2 = np.diff(d1, axis=1)
    s0 = data.var(axis=1)
    s1 = d1.var(axis=1)
    s2 = d2.var(axis=1)
    mobility = np.sqrt(np.where(s0 > 0, s1 / s0, 0))
    complexity = np.sqrt(np.where(s1 > 0, s2 / s1, 0)) / np.where(mobility > 0, mobility, 1)
    return mobility, complexity


def spectral_entropy(P, f, fmin=1, fmax=40):
    mask = (f >= fmin) & (f <= fmax)
    Ps = P[:, mask]
    Ps = Ps / (Ps.sum(axis=1, keepdims=True) + 1e-30)
    H = -np.sum(Ps * np.log(Ps + 1e-30), axis=1)
    return H / np.log(mask.sum())


def one_over_f_slope(P, f, fmin=3, fmax=30):
    mask = (f >= fmin) & (f <= fmax)
    lf = np.log(f[mask])
    out = np.zeros(P.shape[0])
    for i in range(P.shape[0]):
        try:
            out[i] = linregress(lf, np.log(P[i, mask] + 1e-30)).slope
        except Exception:
            out[i] = 0
    return out


def bandpass(data, sf, band):
    b, a = butter(4, [band[0]/(sf/2), band[1]/(sf/2)], btype="band")
    return filtfilt(b, a, data, axis=-1)


def pairwise_connectivity_summary(data, sf, band):
    """Return (msc_left, msc_right, msc_between, wpli_left, wpli_right, wpli_between)."""
    bp = bandpass(data, sf, band)
    analytic = hilbert(bp, axis=-1)
    # cross-spectrum
    def pair_stats(idxA, idxB):
        # mean magnitude-squared coherence approximation & wPLI over pairs
        msc_list, wpli_list = [], []
        for i in idxA:
            for j in idxB:
                if i == j: continue
                x, y = analytic[i], analytic[j]
                csd = x * np.conj(y)
                # wPLI
                num = np.abs(np.mean(np.imag(csd)))
                den = np.mean(np.abs(np.imag(csd))) + 1e-30
                wpli_list.append(num / den)
                # magnitude-squared coherence (approx): |<xy*>|^2 / (<|x|^2><|y|^2>)
                cx = np.mean(np.abs(x)**2); cy = np.mean(np.abs(y)**2)
                num2 = np.abs(np.mean(csd))**2
                msc_list.append(num2 / (cx * cy + 1e-30))
        return float(np.mean(msc_list) if msc_list else 0), float(np.mean(wpli_list) if wpli_list else 0)
    L = [EEG_CHANNELS.index(c) for c in LEFT]
    R = [EEG_CHANNELS.index(c) for c in RIGHT]
    ll_msc, ll_wpli = pair_stats(L, L)
    rr_msc, rr_wpli = pair_stats(R, R)
    lr_msc, lr_wpli = pair_stats(L, R)
    return {
        "msc_LL": ll_msc, "msc_RR": rr_msc, "msc_LR": lr_msc,
        "wpli_LL": ll_wpli, "wpli_RR": rr_wpli, "wpli_LR": lr_wpli,
    }


def feature_vector(data, sf):
    """data: (n_ch, n_times). Return dict of named features."""
    bp, f, P = band_powers(data, sf)
    feats = {}
    # 1. band powers per channel
    for bi, bname in enumerate(BANDS):
        for ci, ch in enumerate(EEG_CHANNELS):
            feats[f"bp_{bname}_{ch}"] = float(np.log(bp[ci, bi] + 1e-30))
    # 2. ratios
    a = bp[:, 2]; t = bp[:, 1]; b = bp[:, 3]
    for ci, ch in enumerate(EEG_CHANNELS):
        feats[f"ratio_a_t_{ch}"] = float(np.log((a[ci] + 1e-30) / (t[ci] + 1e-30)))
        feats[f"ratio_a_b_{ch}"] = float(np.log((a[ci] + 1e-30) / (b[ci] + 1e-30)))
        feats[f"ratio_t_b_{ch}"] = float(np.log((t[ci] + 1e-30) / (b[ci] + 1e-30)))
    # 3. alpha lateralisation
    for name, (Lch, Rch) in LAT_PAIRS.items():
        lp = bp[[EEG_CHANNELS.index(c) for c in Lch], 2].mean()
        rp = bp[[EEG_CHANNELS.index(c) for c in Rch], 2].mean()
        feats[f"alpha_lat_{name}"] = float(np.log((rp + 1e-30) / (lp + 1e-30)))
    # 4. Hjorth
    mob, comp = hjorth(data)
    for ci, ch in enumerate(EEG_CHANNELS):
        feats[f"hjorth_mob_{ch}"] = float(mob[ci])
        feats[f"hjorth_comp_{ch}"] = float(comp[ci])
    # 5. spectral entropy
    se = spectral_entropy(P, f)
    for ci, ch in enumerate(EEG_CHANNELS):
        feats[f"spec_ent_{ch}"] = float(se[ci])
    # 6. 1/f slope per group
    slopes = one_over_f_slope(P, f)
    for gname, chs in GROUPS.items():
        idx = [EEG_CHANNELS.index(c) for c in chs]
        feats[f"slope_{gname}"] = float(np.mean(slopes[idx]))
    # 7. connectivity (alpha only — keep feature count low)
    conn = pairwise_connectivity_summary(data, sf, BANDS["alpha"])
    for k, v in conn.items():
        feats[f"conn_alpha_{k}"] = v
    # 8. GFP
    gfp = data.std(axis=0)
    feats["gfp_mean"] = float(np.mean(gfp))
    feats["gfp_std"] = float(np.std(gfp))
    return feats


def load_preprocessed(subject, k):
    try:
        eeg, ts = load_eeg_trial(subject, k); em = load_eeg_time(subject, k)
        g2 = load_gaze_trial_2d(subject, k); at = load_audio_timestamps(subject, k)
    except Exception:
        return None
    try:
        ali = align_modalities_to_trial(eeg=eeg, eeg_ts=ts, eeg_time_meta=em,
                                        gaze2d=g2, audio_timestamps=at)
        raw = eeg_raw_to_mne(ali["eeg"])
        raw = preprocess_eeg(raw, l_freq=1.0, h_freq=40.0, notch=60.0,
                              reference="auto")
    except Exception:
        return None
    return raw.get_data()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    t0 = time.time()
    tr_csv = load_trials_csv()

    rows = []
    print(f"[S{a.subject}] extracting features", flush=True)
    for k in range(1, 101):
        data = load_preprocessed(a.subject, k)
        if data is None: continue
        f = feature_vector(data, EEG_SFREQ)
        tno = trial_name(k, "main")
        tr = tr_csv[tr_csv["Trial No."] == tno]
        if not len(tr): continue
        f.update(subject=a.subject, trial=k, attended=int(tr.iloc[0]["Attended Speaker"]),
                 snr=float(tr.iloc[0]["SNR"]))
        rows.append(f)
    if len(rows) < 10:
        print(f"[S{a.subject}] too few trials ({len(rows)})"); return
    F = pd.DataFrame(rows)
    feat_cols = [c for c in F.columns if c not in ("subject","trial","attended","snr")]
    print(f"[S{a.subject}] {len(rows)} trials × {len(feat_cols)} features  ({time.time()-t0:.0f}s)", flush=True)

    X = F[feat_cols].fillna(0).values
    y_full = F["attended"].values

    def eval_task(label_fn, task_name, n_classes):
        y = np.array([label_fn(a_) for a_ in y_full])
        if len(np.unique(y)) < n_classes or pd.Series(y).value_counts().min() < 2:
            return []
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        out = []
        # L2 logistic (C chosen moderate).
        lr_accs, gb_accs = [], []
        for tr_i, te_i in skf.split(X, y):
            lr = Pipeline([("sc", StandardScaler()),
                           ("cl", LogisticRegression(max_iter=3000, C=0.5, solver="lbfgs"))])
            lr.fit(X[tr_i], y[tr_i])
            lr_accs.append(accuracy_score(y[te_i], lr.predict(X[te_i])))
            gb = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, verbosity=-1)
            gb.fit(X[tr_i], y[tr_i])
            gb_accs.append(accuracy_score(y[te_i], gb.predict(X[te_i])))
        out.append(dict(subject=a.subject, task=task_name, classifier="logreg",
                        chance=1.0/n_classes, acc=float(np.mean(lr_accs))))
        out.append(dict(subject=a.subject, task=task_name, classifier="lightgbm",
                        chance=1.0/n_classes, acc=float(np.mean(gb_accs))))
        return out

    results = []
    results += eval_task(lambda a_: 0 if ATTENDED_HEMISPHERE[a_]=="L" else 1, "hemisphere", 2)
    results += eval_task(lambda a_: 0 if a_ in (2,3) else 1, "inner_outer", 2)
    results += eval_task(lambda a_: a_-1, "4class", 4)
    R = pd.DataFrame(results)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    # Also save full feature matrix (handy for iter-6 fusion re-runs).
    F.to_parquet(a.out.with_suffix(".features.parquet"))
    R.to_parquet(a.out)
    print(f"[S{a.subject}] done in {time.time()-t0:.0f}s", flush=True)
    print(R.to_string(index=False))


if __name__ == "__main__":
    main()
