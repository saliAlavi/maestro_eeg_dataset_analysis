"""Iter-7 · What can EEG spectral features decode about the OTHER modalities?

For every trial we already have:
    - EEG spectral features (368)        — iter-5
    - Gaze features (23)                 — aad_fusion
    - IMU features (35)                  — iter-6
    - Video features (15)                — iter-6

This script adds:
    - Audio features per trial (~25)     — computed fresh here from the
      attended + unattended stereo files

Then, per subject, it runs ridge regression from EEG features to every
individual feature of every other modality, reports per-target held-out
Pearson $r$ (mean across 5 folds), and pools across subjects with
bootstrap 95 % CI. The output is a sortable matrix of ``how much EEG
knows about each external-modality feature'' under a fixed linear
hypothesis class.

CLI:
    python eeg_decodes_what.py                    # runs full cohort, saves parquet
    python eeg_decodes_what.py --subjects 1,2,3   # subset
"""
from __future__ import annotations
import argparse, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

from aad_utils import (
    RESULTS_DIR, list_subjects, load_trials_csv, load_audio_file,
    audio_envelope, bootstrap_ci, trial_name,
)

# --------------------------------------------------------------------------- #
# Audio feature extractor — computed per trial from the attended + unattended
# device files (matches whatever the subject was listening to).
# --------------------------------------------------------------------------- #
def audio_feats(row) -> dict:
    att_dev = "Device-1" if int(row["Attended Speaker"]) in (1, 2) else "Device-2"
    una_dev = "Device-2" if att_dev == "Device-1" else "Device-1"
    a_att, sr = load_audio_file(row[att_dev])
    a_una, _  = load_audio_file(row[una_dev])

    def summary(x, sr, tag):
        env = audio_envelope(x, sr, sr_out=64.0)
        d = np.diff(env, prepend=env[0])
        f, P = welch(x.astype(float), fs=sr, nperseg=min(2048, len(x)))
        mask = (f >= 50) & (f <= 8000)
        if not mask.sum():
            return {}
        p = P[mask]; f_mask = f[mask]
        p_norm = p / (p.sum() + 1e-30)
        centroid = float((p_norm * f_mask).sum())
        spread = float(np.sqrt(((f_mask - centroid) ** 2 * p_norm).sum()))
        flat = float(np.exp(np.mean(np.log(p + 1e-30))) / (p.mean() + 1e-30))
        zcr = float(np.mean(np.abs(np.diff(np.sign(x))) > 0))
        return {
            f"{tag}_rms":         float(np.sqrt(np.mean(x**2))),
            f"{tag}_env_mean":    float(env.mean()),
            f"{tag}_env_std":     float(env.std()),
            f"{tag}_env_der_pos": float(np.maximum(d, 0).mean()),
            f"{tag}_centroid":    centroid,
            f"{tag}_spread":      spread,
            f"{tag}_flatness":    flat,
            f"{tag}_zcr":         zcr,
        }

    f = {}
    f.update(summary(a_att, sr, "att"))
    f.update(summary(a_una, sr, "una"))
    f["snr_col"]       = float(row["SNR"])
    f["att_una_ratio"] = f["att_rms"] / (f["una_rms"] + 1e-30)
    f["attended_dev"]  = 0 if att_dev == "Device-1" else 1
    return f


def build_audio_table():
    cache = RESULTS_DIR / "audio_features_per_trial.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    TR = load_trials_csv()
    rows = []
    for _, r in TR.iterrows():
        if not r["Trial No."].startswith("Trial-"): continue
        try:
            f = audio_feats(r)
        except Exception as e:
            print(f"  skipped {r['Trial No.']}: {e}")
            continue
        f["trial"] = int(r["Trial No."].split("-")[1])
        rows.append(f)
    A = pd.DataFrame(rows)
    A.to_parquet(cache)
    return A


def ridge_predict_per_feature(X, Y, alpha=10.0, kfold=5):
    """Return dict target_col -> held-out Pearson r (mean across folds)."""
    kf = KFold(kfold, shuffle=True, random_state=0)
    preds = np.zeros_like(Y)
    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=alpha).fit(sc.transform(X[tr]), Y[tr])
        preds[te] = m.predict(sc.transform(X[te]))
    r = {}
    for j in range(Y.shape[1]):
        if np.std(Y[:, j]) < 1e-10:
            r[j] = np.nan; continue
        r[j] = float(np.corrcoef(preds[:, j], Y[:, j])[0, 1])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default=None)
    args = ap.parse_args()
    subs = [int(x) for x in args.subjects.split(",")] if args.subjects else list_subjects()

    print("Loading per-subject feature tables …")
    # EEG spectral features (iter-5) — one parquet per subject, *.features.parquet
    EEG = pd.concat([pd.read_parquet(RESULTS_DIR / "eeg_spectral" / f"s{s}.features.parquet")
                     for s in subs if (RESULTS_DIR/"eeg_spectral"/f"s{s}.features.parquet").exists()],
                     ignore_index=True)
    GZ  = pd.read_parquet(RESULTS_DIR / "fusion_gaze_features.parquet")
    IMU = pd.concat([pd.read_parquet(RESULTS_DIR/"imu_aad"/f"s{s}.features.parquet")
                     for s in subs if (RESULTS_DIR/"imu_aad"/f"s{s}.features.parquet").exists()],
                     ignore_index=True)
    VID = pd.concat([pd.read_parquet(RESULTS_DIR/"video_aad"/f"s{s}.features.parquet")
                     for s in subs if (RESULTS_DIR/"video_aad"/f"s{s}.features.parquet").exists()],
                     ignore_index=True)
    print("Building per-trial audio features …")
    AUD = build_audio_table()
    print(f"  EEG:{EEG.shape} GZ:{GZ.shape} IMU:{IMU.shape} VID:{VID.shape} AUD:{AUD.shape}")

    EEG_FEAT_COLS = [c for c in EEG.columns if c not in ("subject","trial","attended","snr")]
    target_groups = {
        "gaze":  [c for c in GZ.columns  if c not in ("subject","trial","attended","group","snr")],
        "imu":   [c for c in IMU.columns if c not in ("subject","trial","attended","snr")],
        "video": [c for c in VID.columns if c not in ("subject","trial","attended","snr","fps","n_frames")],
        "audio": [c for c in AUD.columns if c not in ("trial","attended_dev","snr_col")],
    }

    rows = []
    for s in subs:
        t0 = time.time()
        Es = EEG[EEG["subject"] == s]
        if len(Es) < 20: continue
        for mod, (DF, key_on_subject) in [("gaze",(GZ,True)),("imu",(IMU,True)),
                                            ("video",(VID,True)),("audio",(AUD,False))]:
            if key_on_subject:
                Ms = DF[DF["subject"] == s]
                merge_cols = ["trial"]
            else:
                Ms = DF
                merge_cols = ["trial"]
            m = Es.merge(Ms, on=merge_cols, suffixes=("_eeg","_m"))
            if len(m) < 20: continue
            X = m[EEG_FEAT_COLS].fillna(0).values
            tgt_cols = target_groups[mod]
            Y = m[tgt_cols].fillna(0).values
            rvec = ridge_predict_per_feature(X, Y, alpha=10.0, kfold=5)
            for j, col in enumerate(tgt_cols):
                rows.append(dict(subject=s, modality=mod, feature=col,
                                 r=rvec.get(j, np.nan), n_trials=len(m)))
        print(f"  S{s} done in {time.time()-t0:.0f}s", flush=True)

    R = pd.DataFrame(rows)
    R.to_parquet(RESULTS_DIR / "eeg_decodes_what.parquet")

    print("\n=== Pooled EEG→feature held-out Pearson r (bootstrap 95% CI) ===")
    for mod in ("gaze","imu","video","audio"):
        sub = R[R["modality"] == mod]
        if sub.empty: continue
        summary = []
        for feat, g in sub.groupby("feature"):
            vals = g["r"].dropna().values
            if len(vals) < 3: continue
            m, lo, hi = bootstrap_ci(vals)
            summary.append((feat, m, lo, hi, len(vals)))
        summary.sort(key=lambda x: -abs(x[1]))
        print(f"\n--- {mod} ---")
        for feat, m, lo, hi, n in summary[:12]:
            print(f"  {feat:<22}  r={m:+.3f}  [{lo:+.3f}, {hi:+.3f}]  n={n}")
    print("\nSaved results/eeg_decodes_what.parquet")


if __name__ == "__main__":
    main()
