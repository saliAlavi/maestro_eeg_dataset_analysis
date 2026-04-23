"""Multimodal EEG+gaze fusion for binary AAD (Device-1 vs Device-2 group).

Uses iter-3 EEG out-of-fold predictions (from results/aad_v3/*) and builds
gaze features per trial to train a late/stacked fusion classifier.

Binary task: attended speaker group ∈ {1 (Device-1), 2 (Device-2)} collapsing
the 4 original speakers (1,2 → group 1; 3,4 → group 2). Chance = 0.5.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb

from aad_utils import (
    RESULTS_DIR, FIGURES_DIR, load_trials_csv, load_gaze_trial_2d,
    load_raw_gaze, load_raw_imu, detect_saccades_ivt, list_subjects,
    set_pub_style, save_fig, COLORS, bootstrap_ci, trial_name,
)
from aad_utils.config import ATTENDED_SPEAKER_MAP
set_pub_style()


def speaker_to_group(att: int) -> int:
    # 1,2 (Device-1 L/R) -> group 0; 3,4 (Device-2 L/R) -> group 1.
    return 0 if att in (1, 2) else 1


def gaze_feats(s, k):
    """Per-trial gaze features — sourced ONLY from the Tobii video-folder
    stream (``gazedata.gz``). The experiment_data 2-D gaze is not used here
    per the authoritative mapping decision.
    """
    try:
        rg = load_raw_gaze(s, k)   # Tobii per-eye + scene-projected gaze
        ri = load_raw_imu(s, k)    # head IMU
    except Exception:
        return None
    if len(rg) < 20: return None
    # Scene-projected gaze (equivalent to the old gaze_x/gaze_y).
    gx = rg['gaze2d_x'].astype(float).values
    gy = rg['gaze2d_y'].astype(float).values
    t = rg['t'].astype(float).values
    sacc = detect_saccades_ivt(t, np.where(np.isfinite(gx), gx, 0.5),
                                  np.where(np.isfinite(gy), gy, 0.5))
    f = dict(
        gx_mean=float(np.nanmean(gx)), gx_std=float(np.nanstd(gx)),
        gy_mean=float(np.nanmean(gy)), gy_std=float(np.nanstd(gy)),
        gx_valid=float(np.mean(np.isfinite(gx))),
        sacc_rate=len(sacc.onsets) / max(1, t[-1] - t[0]),
        sacc_amp_med=float(np.nanmedian(sacc.amplitudes)) if len(sacc.amplitudes) else 0.0,
        # 3-D gaze centroid (listener-referenced, mm) — powerful spatial cue.
        g3d_x_mean=float(np.nanmean(rg['gaze3d_x'])),
        g3d_y_mean=float(np.nanmean(rg['gaze3d_y'])),
        g3d_z_mean=float(np.nanmean(rg['gaze3d_z'])),
        g3d_x_std=float(np.nanstd(rg['gaze3d_x'])),
    )
    for side in ('L', 'R'):
        dx = rg[f'{side}_dx'].values; dy = rg[f'{side}_dy'].values; dz = rg[f'{side}_dz'].values
        az = np.degrees(np.arctan2(dx, dz))
        el = np.degrees(np.arctan2(dy, dz))
        f[f'{side}_az_mean'] = float(np.nanmean(az)); f[f'{side}_az_std'] = float(np.nanstd(az))
        f[f'{side}_el_mean'] = float(np.nanmean(el))
        f[f'{side}_pup'] = float(np.nanmean(rg[f'{side}_pupil']))
        f[f'{side}_valid'] = float(np.mean(np.isfinite(dx)))
    if len(ri):
        f['gyro_mag'] = float(np.linalg.norm(ri[['gx','gy','gz']].values, axis=1).mean())
        f['acc_std'] = float(np.linalg.norm(ri[['ax','ay','az']].values, axis=1).std())
    return f


def load_eeg_oof(v3_dir: Path, features: str = "broadband"):
    # Aggregate per-trial rho_att, rho_una, and the delta as an "EEG probability".
    # Backfill the `attended` column from trials.csv since aad_v3.py only stored `az`.
    tr = load_trials_csv()
    csv_att = {row["Trial No."]: int(row["Attended Speaker"]) for _, row in tr.iterrows()}
    rows = []
    for p in sorted(v3_dir.glob(f"s*_{features}_*.parquet")):
        d = pd.read_parquet(p)
        d = d[d['window_s'] == 'full'].copy()
        d['attended'] = d['trial_name'].map(csv_att)
        d['eeg_prob_g1'] = 1 / (1 + np.exp(-5 * (d['rho_att'] - d['rho_una'])))
        # Aggregate folds (per-subject × trial — same trial appears in multiple folds, keep
        # one row per subject-trial).
        g = d.groupby(['subject', 'trial', 'attended']).agg(
            rho_att=('rho_att', 'mean'),
            rho_una=('rho_una', 'mean'),
            eeg_prob_g1=('eeg_prob_g1', 'mean'),
        ).reset_index()
        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_gaze_table(subjects=None):
    if subjects is None:
        subjects = list_subjects()
    tr_csv = load_trials_csv()
    rows = []
    for s in subjects:
        for k in range(1, 101):
            f = gaze_feats(s, k)
            if f is None: continue
            row = tr_csv[tr_csv['Trial No.'] == f'Trial-{k}']
            if not len(row): continue
            att = int(row.iloc[0]['Attended Speaker'])
            f.update(subject=s, trial=k, attended=att, group=speaker_to_group(att),
                     snr=float(row.iloc[0]['SNR']))
            rows.append(f)
    return pd.DataFrame(rows)


def fit_cv(X, y, clf_ctor, cv):
    oof_prob = np.zeros(len(y))
    for tr, te in cv.split(X, y):
        m = clf_ctor().fit(X[tr], y[tr])
        try:
            oof_prob[te] = m.predict_proba(X[te])[:, 1]
        except Exception:
            oof_prob[te] = (m.decision_function(X[te]) > 0).astype(float)
    return oof_prob


def evaluate_fusion(G: pd.DataFrame, E: pd.DataFrame) -> pd.DataFrame:
    merged = G.merge(E, on=['subject','trial','attended'])
    merged['group'] = merged['attended'].apply(speaker_to_group)
    feat_cols = [c for c in G.columns if c not in ('subject','trial','attended','group','snr')]

    rows = []
    for s, g in merged.groupby('subject'):
        if len(g) < 30: continue
        y = g['group'].values
        Xg = g[feat_cols].fillna(0).values
        Xe = g[['rho_att','rho_una']].values
        Xe_prob = g[['eeg_prob_g1']].values

        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        # Single modalities.
        oof_gaze = fit_cv(Xg, y, lambda: Pipeline([('sc', StandardScaler()),
                                                     ('c', LogisticRegression(max_iter=2000))]),
                          skf)
        oof_eeg_prob = Xe_prob[:, 0]  # already a single number per trial
        oof_eeg_lr  = fit_cv(Xe, y, lambda: Pipeline([('sc', StandardScaler()),
                                                        ('c', LogisticRegression(max_iter=2000))]),
                              skf)
        # Fusion variants.
        oof_late = 0.5 * (oof_gaze + oof_eeg_prob)
        X_stack = np.stack([oof_gaze, oof_eeg_prob], axis=1)
        oof_stack = fit_cv(X_stack, y, lambda: lgb.LGBMClassifier(n_estimators=100, verbosity=-1),
                           skf)
        X_early = np.concatenate([Xg, Xe], axis=1)
        oof_early = fit_cv(X_early, y, lambda: lgb.LGBMClassifier(n_estimators=200, verbosity=-1),
                           skf)
        rows.append(dict(
            subject=s, n=len(g),
            acc_gaze=accuracy_score(y, oof_gaze > 0.5),
            acc_eeg_raw=accuracy_score(y, oof_eeg_prob > 0.5),
            acc_eeg_lr=accuracy_score(y, oof_eeg_lr > 0.5),
            acc_late=accuracy_score(y, oof_late > 0.5),
            acc_stack=accuracy_score(y, oof_stack > 0.5),
            acc_early=accuracy_score(y, oof_early > 0.5),
        ))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eeg-features', default='broadband',
                    choices=['broadband', 'split_delta_theta', 'mel28', 'cca_mel'])
    ap.add_argument('--out', type=Path, default=RESULTS_DIR / 'fusion_summary.parquet')
    args = ap.parse_args()

    v3 = RESULTS_DIR / 'aad_v3'
    E = load_eeg_oof(v3, features=args.eeg_features)
    if not len(E):
        print(f'No iter-3 results found for features={args.eeg_features} in {v3}')
        return
    print(f'Loaded {len(E)} EEG OOF rows for {E["subject"].nunique()} subjects')

    gaze_path = RESULTS_DIR / 'fusion_gaze_features.parquet'
    if gaze_path.exists():
        G = pd.read_parquet(gaze_path)
    else:
        G = build_gaze_table()
        G.to_parquet(gaze_path)
    print(f'Gaze features: {G.shape}')

    results = evaluate_fusion(G, E)
    print('\n=== Per-subject binary AAD accuracy (chance = 0.5) ===')
    print(results.round(3).to_string(index=False))
    print('\n=== Pooled means ===')
    for col in [c for c in results.columns if c.startswith('acc_')]:
        m, lo, hi = bootstrap_ci(results[col].values)
        print(f'  {col:<12}  {m:.3f}  [{lo:.3f}, {hi:.3f}]')

    results.to_parquet(args.out)
    print(f'\nSaved -> {args.out}')

    # Figure.
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    cols = ['acc_gaze','acc_eeg_raw','acc_late','acc_stack','acc_early']
    means = [results[c].mean() for c in cols]
    stds = [results[c].std() for c in cols]
    ax.bar(cols, means, yerr=stds, capsize=3, color=[COLORS['gaze'], COLORS['eeg'],
                                                      COLORS['attended'], COLORS['audio'],
                                                      COLORS['video']])
    ax.axhline(0.5, color=COLORS['chance'], ls='--')
    ax.set_ylim(0, 1); ax.set_ylabel('pooled binary AAD accuracy')
    ax.set_title(f'Multimodal fusion — EEG({args.eeg_features}) × gaze')
    ax.tick_params(axis='x', rotation=20)
    save_fig(fig, f'fusion_{args.eeg_features}', FIGURES_DIR)
    plt.close(fig)


if __name__ == "__main__":
    main()
