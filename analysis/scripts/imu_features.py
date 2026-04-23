"""Iter-6 · IMU feature extraction + AAD control.

Per trial, summarise the 30-s Tobii IMU stream (accelerometer + gyroscope,
recorded-relative clock, ~100 Hz) into a fixed feature vector, then run
the same 3-task AAD (hemisphere / inner-outer / 4-class) as iter-5 —
so we can ask whether head motion ALONE decodes the attended speaker.

If it does, then head movement is a confound; any EEG-AAD signal in
channels that carry muscle artefact (temporal, peripheral) could be
explained by this pathway. If it doesn't, IMU is clean and EEG can't be
accused of capturing only head-motion side-effects.

Per-trial features (35):
    - Per-axis mean / std / peak of accelerometer (3×3 = 9)
    - Per-axis mean / std / peak of gyroscope (3×3 = 9)
    - Magnitude stats (|accel|-g, |gyro|): mean, std, peak, 95th pct (8)
    - Total head rotation (integrated |gyro|) (1)
    - Dominant motion frequency (0.1–4 Hz peak) in accel + gyro magnitude (2)
    - Spectral power ratios (low<1Hz : 1–5Hz : >5Hz) for |accel|-g & |gyro| (6)

CLI:
    python imu_features.py --subject 3 --out results/imu_aad/s3.parquet
"""
from __future__ import annotations
import argparse, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb

from aad_utils import (
    RESULTS_DIR, load_trials_csv, load_raw_imu, trial_name,
)
from aad_utils.config import ATTENDED_HEMISPHERE


def imu_feature_vector(ri: pd.DataFrame) -> dict:
    if len(ri) < 20:
        return None
    t = ri["t"].astype(float).values
    ax = ri["ax"].astype(float).values
    ay = ri["ay"].astype(float).values
    az = ri["az"].astype(float).values
    gx = ri["gx"].astype(float).values
    gy = ri["gy"].astype(float).values
    gz = ri["gz"].astype(float).values
    acc_mag = np.sqrt(ax*ax + ay*ay + az*az) - 9.81
    gyr_mag = np.sqrt(gx*gx + gy*gy + gz*gz)

    f = {}
    for n, v in [("ax",ax),("ay",ay),("az",az),("gx",gx),("gy",gy),("gz",gz)]:
        f[f"{n}_mean"] = float(np.nanmean(v))
        f[f"{n}_std"]  = float(np.nanstd(v))
        f[f"{n}_peak"] = float(np.nanmax(np.abs(v)))
    for n, v in [("acc_mag", acc_mag), ("gyr_mag", gyr_mag)]:
        f[f"{n}_mean"] = float(np.nanmean(v))
        f[f"{n}_std"]  = float(np.nanstd(v))
        f[f"{n}_peak"] = float(np.nanmax(v))
        f[f"{n}_p95"]  = float(np.nanpercentile(v, 95))

    # Total head rotation: integrated |gyro|.
    dt = np.mean(np.diff(t)) if len(t) > 1 else 0.01
    f["total_rotation"] = float(np.nansum(gyr_mag) * dt)

    # Spectra on magnitudes — resample to uniform grid if non-uniform.
    if len(t) > 50:
        fs = 1.0 / dt
        for n, v in [("acc_mag", acc_mag), ("gyr_mag", gyr_mag)]:
            x = np.nan_to_num(v, nan=0.0)
            ff, P = welch(x, fs=fs, nperseg=min(128, len(x)//2))
            if len(P):
                f[f"{n}_dom_freq"] = float(ff[np.argmax(P)])
                # Band ratios.
                def band_frac(lo, hi):
                    m = (ff >= lo) & (ff < hi)
                    return float(P[m].sum() / (P.sum() + 1e-30))
                f[f"{n}_low"]  = band_frac(0, 1)
                f[f"{n}_mid"]  = band_frac(1, 5)
                f[f"{n}_high"] = band_frac(5, 50)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    t0 = time.time()
    tr_csv = load_trials_csv()
    rows = []
    for k in range(1, 101):
        try:
            ri = load_raw_imu(a.subject, k)
        except Exception: continue
        if ri is None or len(ri) < 20: continue
        f = imu_feature_vector(ri)
        if f is None: continue
        tno = trial_name(k, "main")
        tr = tr_csv[tr_csv["Trial No."] == tno]
        if not len(tr): continue
        f.update(subject=a.subject, trial=k, attended=int(tr.iloc[0]["Attended Speaker"]),
                 snr=float(tr.iloc[0]["SNR"]))
        rows.append(f)
    if len(rows) < 10:
        print(f"[S{a.subject}] too few trials"); return
    F = pd.DataFrame(rows)
    feat_cols = [c for c in F.columns if c not in ("subject","trial","attended","snr")]
    X = F[feat_cols].fillna(0).values
    y_full = F["attended"].values
    print(f"[S{a.subject}] {len(F)} trials × {len(feat_cols)} IMU features", flush=True)

    def eval_task(lbl, task, nc):
        y = np.array([lbl(a_) for a_ in y_full])
        if len(np.unique(y)) < nc or pd.Series(y).value_counts().min() < 2:
            return []
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        lr_accs, gb_accs = [], []
        for tr_i, te_i in skf.split(X, y):
            lr = Pipeline([("sc", StandardScaler()),
                           ("cl", LogisticRegression(max_iter=3000, C=0.5))]).fit(X[tr_i], y[tr_i])
            lr_accs.append(accuracy_score(y[te_i], lr.predict(X[te_i])))
            gb = lgb.LGBMClassifier(n_estimators=200, verbosity=-1).fit(X[tr_i], y[tr_i])
            gb_accs.append(accuracy_score(y[te_i], gb.predict(X[te_i])))
        return [
            dict(subject=a.subject, task=task, classifier="logreg",
                 chance=1/nc, acc=float(np.mean(lr_accs))),
            dict(subject=a.subject, task=task, classifier="lightgbm",
                 chance=1/nc, acc=float(np.mean(gb_accs))),
        ]

    results = []
    results += eval_task(lambda a_: 0 if ATTENDED_HEMISPHERE[a_]=="L" else 1, "hemisphere", 2)
    results += eval_task(lambda a_: 0 if a_ in (2,3) else 1, "inner_outer", 2)
    results += eval_task(lambda a_: a_-1, "4class", 4)
    R = pd.DataFrame(results)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    F.to_parquet(a.out.with_suffix(".features.parquet"))
    R.to_parquet(a.out)
    print(f"[S{a.subject}] done in {time.time()-t0:.0f}s", flush=True)
    print(R.to_string(index=False))


if __name__ == "__main__":
    main()
