"""Build 07_gaze_AAD.ipynb — gaze-only AAD baseline."""
from _build_notebook import build

CELLS = [
("md", """\
# 07 · Gaze-only AAD baseline

The task is spatial, so an uncalibrated but precise gaze stream may already
contain strong attended-speaker information. Here we benchmark a pure-gaze
AAD classifier and compare it against EEG-only.

Features per trial (computed on `gaze2d_x/y`, plus Tobii raw per-eye when
available):

- mean / median / std of horizontal and vertical gaze
- per-eye 3-D gaze-direction angles (azimuth θ, elevation φ) aggregated over
  the trial
- saccade rate, mean saccade amplitude, dominant saccade direction
- pupil baseline and slope
- head motion (IMU) summary statistics

Classifiers: logistic regression (with class weights), gradient boosting
(LightGBM), and a small 1-D TCN (implemented, not run unless GPU).
"""),

("code", """\
import sys, os, warnings; sys.path.insert(0, os.path.abspath('.'))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb
from aad_utils import (list_subjects, load_trials_csv, load_gaze_trial_2d,
                       load_raw_gaze, load_raw_imu, detect_saccades_ivt,
                       FIGURES_DIR, RESULTS_DIR, set_pub_style, save_fig, COLORS)
from aad_utils.config import ATTENDED_SPEAKER_MAP
set_pub_style()
SUBJECTS = list_subjects(); TRIALS = load_trials_csv()
"""),

("md", "## 1 · Feature extraction"),
("code", """\
def gaze_features(subject, k):
    try:
        g2 = load_gaze_trial_2d(subject, k)
    except Exception: return None
    if len(g2) < 20: return None
    sacc = detect_saccades_ivt(g2['gaze_ts'].values, g2['gaze_x'].fillna(0.5).values, g2['gaze_y'].fillna(0.5).values)
    rg = load_raw_gaze(subject, k); ri = load_raw_imu(subject, k)
    f = dict(
        gx_mean=np.nanmean(g2['gaze_x']), gx_med=np.nanmedian(g2['gaze_x']), gx_std=np.nanstd(g2['gaze_x']),
        gy_mean=np.nanmean(g2['gaze_y']), gy_med=np.nanmedian(g2['gaze_y']), gy_std=np.nanstd(g2['gaze_y']),
        sacc_rate=len(sacc.onsets)/max(1, g2['gaze_ts'].values[-1]-g2['gaze_ts'].values[0]),
        sacc_amp_med=np.nanmedian(sacc.amplitudes) if len(sacc.amplitudes) else 0.0,
        sacc_amp_max=np.nanmax(sacc.amplitudes) if len(sacc.amplitudes) else 0.0,
    )
    if len(rg):
        for side in ('L','R'):
            az = np.degrees(np.arctan2(rg[f'{side}_dx'], rg[f'{side}_dz']))
            el = np.degrees(np.arctan2(rg[f'{side}_dy'], rg[f'{side}_dz']))
            f[f'{side}_az_mean'] = np.nanmean(az); f[f'{side}_az_std'] = np.nanstd(az)
            f[f'{side}_el_mean'] = np.nanmean(el); f[f'{side}_el_std'] = np.nanstd(el)
            f[f'{side}_pupil_mean'] = np.nanmean(rg[f'{side}_pupil'])
            f[f'{side}_pupil_slope'] = np.polyfit(np.arange(len(rg)), rg[f'{side}_pupil'].fillna(rg[f'{side}_pupil'].median()), 1)[0] if len(rg) > 10 else 0.0
    if len(ri):
        f['gyro_mag_mean'] = float(np.linalg.norm(ri[['gx','gy','gz']].values, axis=1).mean())
        f['acc_mag_std'] = float(np.linalg.norm(ri[['ax','ay','az']].values, axis=1).std())
    return f

rows = []
for s in SUBJECTS:
    for k in range(1, 101):
        f = gaze_features(s, k)
        if f is None: continue
        tno = f'Trial-{k}'
        tr = TRIALS[TRIALS['Trial No.']==tno]
        if not len(tr): continue
        att = int(tr.iloc[0]['Attended Speaker'])
        az = ATTENDED_SPEAKER_MAP[att][2]
        f.update(subject=s, trial=k, attended=att, attended_az=az,
                 snr=float(tr.iloc[0]['SNR']))
        rows.append(f)
G = pd.DataFrame(rows)
G.to_parquet(RESULTS_DIR / '07_gaze_features.parquet')
print(G.shape, G.head())
"""),

("md", "## 2 · Within-subject 4-way classifier (speaker 1-4)"),
("code", """\
def eval_cv(clf, X, y, groups=None, loso=False):
    if loso:
        cv = LeaveOneGroupOut().split(X, y, groups)
    else:
        cv = StratifiedKFold(5, shuffle=True, random_state=0).split(X, y)
    acc = []; auc = []
    for tr, te in cv:
        clf.fit(X[tr], y[tr])
        p = clf.predict(X[te]); acc.append(accuracy_score(y[te], p))
        try:
            pp = clf.predict_proba(X[te])
            auc.append(roc_auc_score(y[te], pp, multi_class='ovr', average='macro'))
        except Exception: pass
    return acc, auc

feat_cols = [c for c in G.columns if c not in ('subject','trial','attended','attended_az','snr')]
X = G[feat_cols].fillna(0).values; y = G['attended'].values
scores = {}
for s in G['subject'].unique():
    idx = G['subject']==s
    if idx.sum() < 40: continue
    acc, auc = eval_cv(Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(max_iter=2000))]),
                        X[idx], y[idx])
    scores[s] = dict(acc_mean=np.mean(acc), acc_std=np.std(acc), auc_mean=np.mean(auc) if auc else np.nan)
within = pd.DataFrame(scores).T
print('Within-subject 4-way gaze AAD accuracy:')
print(within)
within.to_parquet(RESULTS_DIR / '07_within_subject_gaze.parquet')
"""),

("code", """\
# LightGBM within-subject
lgb_scores = {}
for s in G['subject'].unique():
    idx = G['subject']==s
    if idx.sum() < 40: continue
    Xs, ys = X[idx], y[idx]
    accs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xs, ys):
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, verbosity=-1)
        m.fit(Xs[tr], ys[tr])
        accs.append(accuracy_score(ys[te], m.predict(Xs[te])))
    lgb_scores[s] = np.mean(accs)
print('LightGBM within-subject:', lgb_scores)
"""),

("md", "## 3 · Leave-one-subject-out"),
("code", """\
clf = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(max_iter=2000))])
acc, auc = eval_cv(clf, X, y, groups=G['subject'].values, loso=True)
print(f'LOSO LogReg accuracy: {np.mean(acc):.3f} ± {np.std(acc):.3f}')
loso = pd.DataFrame(dict(subject=sorted(G['subject'].unique()), acc=acc))
loso.to_parquet(RESULTS_DIR / '07_loso_gaze.parquet')
fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(loso['subject'].astype(str), loso['acc'], color=COLORS['gaze'])
ax.axhline(0.25, color=COLORS['chance'], ls='--', label='chance (4-AFC)')
ax.set_xlabel('held-out subject'); ax.set_ylabel('accuracy')
ax.set_title('Gaze-only LOSO AAD'); ax.legend()
save_fig(fig, '07_loso_gaze', FIGURES_DIR); plt.show()
"""),

("md", "## 4 · Feature importance (SHAP on LightGBM pooled)"),
("code", """\
import shap
m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, verbosity=-1).fit(X, y)
expl = shap.TreeExplainer(m)
sv = expl.shap_values(X)
# Reduce to a per-feature importance vector. SHAP's return shape depends on
# both the library version and the classifier:
#   list of (n_samples, n_features) arrays, one per class  [older SHAP]
#   (n_samples, n_features, n_classes) 3-D array           [newer SHAP]
#   (n_samples, n_features) 2-D array                      [binary / regression]
if isinstance(sv, list):
    imp = np.mean([np.abs(s).mean(0) for s in sv], axis=0)
else:
    sv_arr = np.asarray(sv)
    if sv_arr.ndim == 3:
        imp = np.abs(sv_arr).mean(axis=(0, 2))  # mean over samples and classes
    else:
        imp = np.abs(sv_arr).mean(0)
imp = np.asarray(imp).ravel()
order = np.argsort(imp)[::-1]
fig, ax = plt.subplots(figsize=(6, 4))
ax.barh(range(15), imp[order[:15]][::-1], color=COLORS['gaze'])
ax.set_yticks(range(15)); ax.set_yticklabels([feat_cols[i] for i in order[:15]][::-1])
ax.set_xlabel('mean |SHAP|'); ax.set_title('Top gaze features for attended-speaker')
save_fig(fig, '07_shap_gaze', FIGURES_DIR); plt.show()
"""),

("md", "## 5 · Deep gaze-TCN (implemented, not run)"),
("code", """\
RUN_DEEP = False
import torch, torch.nn as nn

class GazeTCN(nn.Module):
    def __init__(self, in_dim=6, n_classes=4, channels=(32, 64, 128)):
        super().__init__()
        layers = []
        d = in_dim
        for c in channels:
            layers += [nn.Conv1d(d, c, 5, padding=2), nn.BatchNorm1d(c), nn.ReLU(), nn.Dropout(0.2)]
            d = c
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(channels[-1], n_classes)
    def forward(self, x):  # (B, C, T)
        return self.head(self.trunk(x).mean(dim=2))

print('GazeTCN defined.', sum(p.numel() for p in GazeTCN().parameters()), 'params.')
if RUN_DEEP:
    print('(would train here)')
"""),
]
build('/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/07_gaze_AAD.ipynb', CELLS)
print('Wrote 07_gaze_AAD.ipynb')
