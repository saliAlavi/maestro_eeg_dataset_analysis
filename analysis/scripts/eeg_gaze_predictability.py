"""Cross-modal predictability: EEG ↔ gaze, gaze→attended, EEG→attended.

Three analyses per subject (all 100 main trials, 5-fold within-subject CV):

    1) EEG → gaze  : ridge regression from EEG lags to scene-projected
       gaze2d_x, gaze2d_y, gaze3d_x (horizontal direction). Reports the
       held-out Pearson r per target.

    2) Gaze → attended : logistic/softmax classification from Tobii-gaze
       summary features to attended-speaker label, for each of:
         A) hemisphere (L vs R)
         B) inner vs outer
         C) 4-class speaker identity

    3) CCA EEG ↔ gaze : first canonical correlation between EEG lags and
       (gaze2d_x, gaze2d_y, gaze3d_x/y/z) — single scalar per subject,
       reports in-sample and held-out CC.

CLI:
    python eeg_gaze_predictability.py --subject 3 --out results/pred/s3.parquet
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
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score

from aad_utils import (
    EEG_CHANNELS, RESULTS_DIR, load_trials_csv,
    load_eeg_trial, load_eeg_time, load_gaze_trial_2d, load_audio_timestamps,
    load_raw_gaze, load_raw_imu, detect_saccades_ivt,
    align_modalities_to_trial, eeg_raw_to_mne, preprocess_eeg, trial_name,
)
from aad_utils.config import ATTENDED_HEMISPHERE

SR_OUT = 64.0
LAGS_MS = np.arange(-200, 301, 25)


def resample_to(t, x, out_t):
    mask = np.isfinite(t) & np.isfinite(x)
    if mask.sum() < 3:
        return np.full(len(out_t), np.nan)
    return interp1d(t[mask], x[mask], bounds_error=False, fill_value=np.nan)(out_t)


def load_multimodal_trial(subject, k):
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
        raw = preprocess_eeg(raw, l_freq=1.0, h_freq=40.0, notch=60.0,
                              reference="auto")
        raw.resample(SR_OUT, verbose="ERROR")
    except Exception:
        return None
    E = raw.get_data().T
    T = E.shape[0]
    out_t = np.linspace(0, T / SR_OUT, T)
    rg_a = ali.get("raw_gaze", pd.DataFrame())
    if not len(rg_a):
        return None
    t_rel = rg_a["t_unix"].values - ali["window"].t0
    gaze2d_x = resample_to(t_rel, rg_a["gaze2d_x"].astype(float).values, out_t)
    gaze2d_y = resample_to(t_rel, rg_a["gaze2d_y"].astype(float).values, out_t)
    gaze3d_x = resample_to(t_rel, rg_a["gaze3d_x"].astype(float).values, out_t)
    gaze3d_y = resample_to(t_rel, rg_a["gaze3d_y"].astype(float).values, out_t)
    gaze3d_z = resample_to(t_rel, rg_a["gaze3d_z"].astype(float).values, out_t)
    tno = trial_name(k, "main")
    tr = load_trials_csv()
    row = tr[tr["Trial No."] == tno]
    att = int(row.iloc[0]["Attended Speaker"]) if len(row) else -1
    return dict(subject=subject, trial=k, trial_name=tno, eeg=E,
                gaze2d_x=gaze2d_x, gaze2d_y=gaze2d_y,
                gaze3d_x=gaze3d_x, gaze3d_y=gaze3d_y, gaze3d_z=gaze3d_z,
                attended=att)


def design_lags(X, lags):
    T, C = X.shape
    out = np.zeros((T, C * len(lags)))
    for i, lag in enumerate(lags):
        if lag >= 0:
            out[:, i*C:(i+1)*C] = np.vstack([np.zeros((lag, C)), X[:T-lag]])
        else:
            out[:, i*C:(i+1)*C] = np.vstack([X[-lag:], np.zeros((-lag, C))])
    return out


def eeg_to_gaze(trials, alpha=1e3):
    """Ridge regression EEG-lags → 5 gaze targets. Returns per-target held-out
    Pearson r (averaged over 5 folds, over trials)."""
    lags = [int(round(ms * SR_OUT / 1000)) for ms in LAGS_MS]
    targets = ["gaze2d_x", "gaze2d_y", "gaze3d_x", "gaze3d_y", "gaze3d_z"]
    kf = KFold(5, shuffle=True, random_state=0)
    r_all = {t: [] for t in targets}
    for tr_i, te_i in kf.split(trials):
        Xs, Ys = [], []
        for i in tr_i:
            tr = trials[i]
            Ylist = np.stack([tr[t_] for t_ in targets], axis=1)
            mask = np.all(np.isfinite(Ylist), axis=1)
            if mask.sum() < 50: continue
            Xs.append(design_lags(tr["eeg"], lags)[mask])
            Ys.append(Ylist[mask])
        if not Xs: continue
        X = np.vstack(Xs); Y = np.vstack(Ys)
        m = Ridge(alpha=alpha).fit(X, Y)
        for i in te_i:
            t = trials[i]
            Ylist = np.stack([t[t_] for t_ in targets], axis=1)
            mask = np.all(np.isfinite(Ylist), axis=1)
            if mask.sum() < 50: continue
            pred = m.predict(design_lags(t["eeg"], lags)[mask])
            for j, tg in enumerate(targets):
                r = np.corrcoef(pred[:, j], Ylist[mask, j])[0, 1]
                r_all[tg].append(r)
    return {t: float(np.nanmean(r_all[t])) if r_all[t] else np.nan for t in targets}


def cca_eeg_gaze(trials):
    """First canonical correlation between EEG lags and the 5 gaze targets."""
    lags = [int(round(ms * SR_OUT / 1000)) for ms in LAGS_MS]
    targets = ["gaze2d_x", "gaze2d_y", "gaze3d_x", "gaze3d_y", "gaze3d_z"]
    Xs, Ys = [], []
    for t in trials:
        Y = np.stack([t[tg] for tg in targets], axis=1)
        mask = np.all(np.isfinite(Y), axis=1)
        if mask.sum() < 50: continue
        Xs.append(design_lags(t["eeg"], lags)[mask])
        Ys.append(Y[mask])
    if not Xs: return dict(cca_rho_in=np.nan, cca_rho_out=np.nan)
    X = np.vstack(Xs); Y = np.vstack(Ys)
    # Held-out via KFold on rows.
    n = len(X); kf = KFold(5, shuffle=True, random_state=0)
    in_r, out_r = [], []
    for tr_i, te_i in kf.split(np.arange(n)):
        try:
            cca = CCA(n_components=1, max_iter=100).fit(X[tr_i], Y[tr_i])
        except Exception:
            continue
        Xc_tr, Yc_tr = cca.transform(X[tr_i], Y[tr_i])
        Xc_te, Yc_te = cca.transform(X[te_i], Y[te_i])
        in_r.append(np.corrcoef(Xc_tr.ravel(), Yc_tr.ravel())[0, 1])
        out_r.append(np.corrcoef(Xc_te.ravel(), Yc_te.ravel())[0, 1])
    return dict(cca_rho_in=float(np.mean(in_r)),
                cca_rho_out=float(np.mean(out_r)))


def gaze_summary_feats(subject, k):
    try:
        rg = load_raw_gaze(subject, k); ri = load_raw_imu(subject, k)
    except Exception: return None
    if len(rg) < 20: return None
    gx = rg["gaze2d_x"].astype(float).values
    gy = rg["gaze2d_y"].astype(float).values
    t_ = rg["t"].astype(float).values
    sacc = detect_saccades_ivt(t_, np.where(np.isfinite(gx), gx, 0.5),
                                    np.where(np.isfinite(gy), gy, 0.5))
    f = dict(
        gx_mean=float(np.nanmean(gx)), gx_std=float(np.nanstd(gx)),
        gy_mean=float(np.nanmean(gy)), gy_std=float(np.nanstd(gy)),
        sacc_rate=len(sacc.onsets)/max(1, t_[-1]-t_[0]),
        sacc_amp_med=float(np.nanmedian(sacc.amplitudes)) if len(sacc.amplitudes) else 0.0,
        g3d_x_mean=float(np.nanmean(rg["gaze3d_x"])),
        g3d_y_mean=float(np.nanmean(rg["gaze3d_y"])),
        g3d_z_mean=float(np.nanmean(rg["gaze3d_z"])),
    )
    for side in ("L", "R"):
        dx = rg[f"{side}_dx"].values; dy = rg[f"{side}_dy"].values; dz = rg[f"{side}_dz"].values
        az = np.degrees(np.arctan2(dx, dz)); el = np.degrees(np.arctan2(dy, dz))
        f[f"{side}_az_mean"] = float(np.nanmean(az)); f[f"{side}_az_std"] = float(np.nanstd(az))
        f[f"{side}_el_mean"] = float(np.nanmean(el))
        f[f"{side}_pup"] = float(np.nanmean(rg[f"{side}_pupil"]))
    if len(ri):
        f["gyro_mag"] = float(np.linalg.norm(ri[["gx","gy","gz"]].values, axis=1).mean())
    return f


def gaze_to_attended(subject, trials):
    """Per-subject, 5-fold CV logistic regression: gaze → attended label,
    for 3 framings (hemisphere, inner/outer, 4-class)."""
    rows = []
    tr_csv = load_trials_csv()
    feats = []
    for t in trials:
        gf = gaze_summary_feats(subject, t["trial"])
        if gf is None: continue
        gf.update(subject=subject, trial=t["trial"], attended=t["attended"])
        feats.append(gf)
    if not feats: return rows
    G = pd.DataFrame(feats)
    feat_cols = [c for c in G.columns if c not in ("subject","trial","attended")]
    X = G[feat_cols].fillna(0).values
    for task, lbl_fn, nc in [
        ("hemisphere", lambda a: 0 if ATTENDED_HEMISPHERE[a]=="L" else 1, 2),
        ("inner_outer", lambda a: 0 if a in (2,3) else 1, 2),
        ("4class", lambda a: a-1, 4)
    ]:
        y = G["attended"].apply(lbl_fn).values
        if len(np.unique(y)) < nc: continue
        if pd.Series(y).value_counts().min() < 2: continue
        acc = []
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        for tr_i, te_i in skf.split(X, y):
            m = Pipeline([("sc", StandardScaler()),
                          ("cl", LogisticRegression(max_iter=2000, multi_class="auto"))])
            m.fit(X[tr_i], y[tr_i])
            acc.append(accuracy_score(y[te_i], m.predict(X[te_i])))
        rows.append(dict(subject=subject, task=task, chance=1.0/nc, acc=float(np.mean(acc))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    t0 = time.time()
    print(f"[S{a.subject}] loading multimodal trials", flush=True)
    trials = []
    for k in range(1, 101):
        p = load_multimodal_trial(a.subject, k)
        if p is not None: trials.append(p)
    print(f"[S{a.subject}] {len(trials)}/100 trials loaded in {time.time()-t0:.0f}s", flush=True)
    if len(trials) < 10:
        print("too few trials"); return

    eeg2gaze = eeg_to_gaze(trials)
    cca_pair = cca_eeg_gaze(trials)
    g2a = gaze_to_attended(a.subject, trials)
    rows = []
    for tgt, r in eeg2gaze.items():
        rows.append(dict(subject=a.subject, analysis="eeg_to_gaze", target=tgt, value=r))
    rows.append(dict(subject=a.subject, analysis="cca_eeg_gaze", target="in_sample", value=cca_pair["cca_rho_in"]))
    rows.append(dict(subject=a.subject, analysis="cca_eeg_gaze", target="held_out", value=cca_pair["cca_rho_out"]))
    for r in g2a:
        rows.append(dict(subject=a.subject, analysis="gaze_to_attended", target=r["task"],
                         value=r["acc"], chance=r["chance"]))
    df = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out)
    print(f"[S{a.subject}] done in {time.time()-t0:.0f}s", flush=True)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
