"""Per-subject individual-differences regression.

Dependent variable: per-subject decoder accuracy for each of
{gaze-hemisphere, eeg-spectral-hemisphere, fusion-early-hemisphere,
 cca-4class-identity}.

Predictors (all per-subject):
    - comprehension: behavioural question accuracy (02_per_subject_acc)
    - bad_channel_rate: cohort rate across 100 main trials
    - alpha_lat_mag: RMS ALI on Trials 6-15 (audit_alpha_lateralization)
    - pupil_mean: per-subject mean pupil diameter (07_gaze_features)
    - head_motion: per-subject mean gyro magnitude (imu_aad features)
    - gaze_valid: per-subject mean L_valid + R_valid (fusion_gaze_features)

Writes:
    results/individual_differences_covariates.parquet
    results/individual_differences_regression.parquet
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def load_covariates() -> pd.DataFrame:
    # 1) Behavioural comprehension accuracy
    comp = pd.read_parquet(RESULTS / "02_per_subject_acc.parquet")[
        ["subject", "mean"]
    ].rename(columns={"mean": "comprehension"})

    # 2) Bad-channel rate: fraction of main trials with bad_fraction > 0.0625
    bc = pd.read_parquet(RESULTS / "bad_channels_manifest.parquet")
    bc = bc[bc["is_training"] == 0]
    bc_rate = bc.groupby("subject")["n_bad"].mean().rename(
        "bad_channel_rate").reset_index()

    # 3) Alpha lateralisation magnitude: RMS ALI across trials & az
    ali = pd.read_parquet(RESULTS / "audit_alpha_lateralization.parquet")
    ali_rms = ali.groupby("subject")["ALI"].apply(
        lambda x: float(np.sqrt(np.mean(x ** 2)))
    ).rename("alpha_lat_rms").reset_index()

    # 4) Pupil mean (from fusion gaze features — 1598 rows × 28 cols)
    gf = pd.read_parquet(RESULTS / "fusion_gaze_features.parquet")
    pupil = gf.groupby("subject")[["L_pup", "R_pup"]].mean()
    pupil["pupil_mean"] = pupil[["L_pup", "R_pup"]].mean(axis=1)
    pupil = pupil.reset_index()[["subject", "pupil_mean"]]

    # 5) Head motion burden: mean gyro magnitude from IMU features
    imu_rows = []
    for fp in sorted(glob.glob(str(RESULTS / "imu_aad/s*.features.parquet"))):
        df = pd.read_parquet(fp)
        imu_rows.append(df[["subject", "gyr_mag_mean"]])
    imu = pd.concat(imu_rows, ignore_index=True)
    head_motion = imu.groupby("subject")["gyr_mag_mean"].mean().rename(
        "head_motion").reset_index()

    # 6) Gaze validity
    valid = gf.groupby("subject")[["L_valid", "R_valid"]].mean()
    valid["gaze_valid"] = valid[["L_valid", "R_valid"]].mean(axis=1)
    valid = valid.reset_index()[["subject", "gaze_valid"]]

    cov = comp.merge(bc_rate, on="subject", how="outer") \
              .merge(ali_rms, on="subject", how="outer") \
              .merge(pupil, on="subject", how="outer") \
              .merge(head_motion, on="subject", how="outer") \
              .merge(valid, on="subject", how="outer")
    cov["subject"] = cov["subject"].astype(int)
    return cov.sort_values("subject").reset_index(drop=True)


def load_targets() -> pd.DataFrame:
    # per-subject hemisphere accuracy — from fusion_cca_mel.parquet
    fuse = pd.read_parquet(RESULTS / "fusion_cca_mel.parquet")
    fuse = fuse[["subject", "acc_gaze", "acc_eeg_lr", "acc_early"]]
    fuse = fuse.rename(columns={
        "acc_gaze": "gaze_hemi",
        "acc_eeg_lr": "eeg_hemi",
        "acc_early": "fusion_hemi",
    })
    # EEG spectral hemisphere
    spec_rows = []
    for fp in sorted(glob.glob(str(RESULTS / "eeg_spectral/s*.parquet"))):
        if ".features" in fp:
            continue
        df = pd.read_parquet(fp)
        r = df[(df["task"] == "hemisphere") & (df["classifier"] == "logreg")]
        if not r.empty:
            spec_rows.append(
                {"subject": int(r["subject"].iloc[0]),
                 "eeg_spectral_hemi": float(r["acc"].iloc[0])}
            )
    spec = pd.DataFrame(spec_rows)
    # 4-class CCA accuracy
    v4 = pd.read_parquet(RESULTS / "iter4_4class_per_subject.parquet")
    v4 = v4.reset_index()[["subject", "acc_4class"]].rename(
        columns={"acc_4class": "cca_4class"})
    v4["subject"] = v4["subject"].astype(int)

    t = fuse.merge(spec, on="subject", how="left").merge(
        v4, on="subject", how="left")
    return t.sort_values("subject").reset_index(drop=True)


def regress(cov: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    merged = cov.merge(targets, on="subject")
    pred_cols = ["comprehension", "bad_channel_rate", "alpha_lat_rms",
                 "pupil_mean", "head_motion", "gaze_valid"]
    out_cols = ["gaze_hemi", "eeg_hemi", "fusion_hemi",
                "eeg_spectral_hemi", "cca_4class"]

    rows: list[dict] = []
    # per-target: standardised coefficients + per-predictor Pearson r
    for y_name in out_cols:
        y = merged[y_name].to_numpy()
        mask = ~np.isnan(y)
        y = y[mask]
        Xp = merged.loc[mask, pred_cols].to_numpy()
        # drop any col with nans
        valid_cols = [i for i in range(Xp.shape[1]) if not np.isnan(Xp[:, i]).any()]
        Xp = Xp[:, valid_cols]
        if len(y) < 5:
            continue
        sc_X = StandardScaler().fit_transform(Xp)
        sc_y = (y - y.mean()) / (y.std() if y.std() > 0 else 1.0)
        lr = LinearRegression().fit(sc_X, sc_y)
        r2 = lr.score(sc_X, sc_y)
        for j, coln_idx in enumerate(valid_cols):
            coln = pred_cols[coln_idx]
            r = float(np.corrcoef(Xp[:, j], y)[0, 1])
            rows.append({
                "target": y_name, "predictor": coln,
                "std_coef": float(lr.coef_[j]),
                "pearson_r": r,
                "r2_joint": float(r2),
                "n": int(len(y)),
            })
    return pd.DataFrame(rows)


def main() -> int:
    cov = load_covariates()
    tgt = load_targets()
    reg = regress(cov, tgt)

    cov.to_parquet(RESULTS / "individual_differences_covariates.parquet",
                   index=False)
    print("Covariates per subject:")
    print(cov.to_string(index=False))

    tgt.to_parquet(RESULTS / "individual_differences_targets.parquet",
                   index=False)
    print("\nTargets per subject:")
    print(tgt.to_string(index=False))

    reg.to_parquet(RESULTS / "individual_differences_regression.parquet",
                   index=False)
    print("\nRegression (std-coef + univariate Pearson r):")
    print(reg.to_string(index=False))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
