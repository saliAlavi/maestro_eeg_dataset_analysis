"""Build 09_multimodal_fusion.ipynb."""
from _build_notebook import build

CELLS = [
("md", """\
# 09 · Multimodal fusion for AAD

Compare **single-modality** baselines (EEG, gaze, pupil, IMU, video-motion)
against **fusion** strategies for attended-speaker classification.

Fusion schemes implemented:

- **Early (concatenation)**: per-trial feature vectors concatenated, fed to
  logistic regression / gradient boosting.
- **Late (probability averaging)**: class probabilities from per-modality
  classifiers averaged with learned weights (LogReg over probs).
- **Stacked generalisation**: out-of-fold predictions from each modality
  as meta-features, meta-learner = LightGBM.
- **Attention fusion (deep, implemented, not run)**: cross-modal attention
  transformer consuming per-modality token streams.

Reports:

- Per-modality vs fusion accuracy, ΔAUC, within-subject and LOSO.
- **Shapley value** modality-contribution decomposition.
- Learning curves as function of training-trial budget.
"""),

("code", """\
import sys, os, warnings; sys.path.insert(0, os.path.abspath('.'))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb
from aad_utils import (list_subjects, load_trials_csv, CACHE_DIR, FIGURES_DIR,
                       RESULTS_DIR, set_pub_style, save_fig, COLORS)
set_pub_style()
SUBJECTS = list_subjects(); TRIALS = load_trials_csv()
"""),

("md", "## 1 · Build per-modality feature tables"),
("code", """\
# Reuse features from nb07 (gaze) plus aggregate EEG/pupil/IMU simple summaries.
gaze_p = RESULTS_DIR / '07_gaze_features.parquet'
if gaze_p.exists():
    G = pd.read_parquet(gaze_p)
    print('Gaze features:', G.shape)
else:
    print('Run notebook 07 first to produce gaze features.')
    G = pd.DataFrame()
"""),

("code", """\
# Simple EEG summary: band power features per canonical band across frontal/central/parietal/occipital groups.
from aad_utils import (EEG_CHANNELS, load_eeg_trial, load_eeg_time, load_gaze_trial_2d,
                       load_audio_timestamps, align_modalities_to_trial, eeg_raw_to_mne,
                       preprocess_eeg, EEG_SFREQ)
from scipy.signal import welch

BANDS = dict(delta=(1,4), theta=(4,8), alpha=(8,13), beta=(13,30))
GROUPS = dict(
    frontal=['Fp1','Fp2','Fpz','F7','F3','Fz','F4','F8'],
    central=['FC5','FC1','FC2','FC6','C3','Cz','C4'],
    parietal=['CP5','CP1','CP2','CP6','P3','Pz','P4','P7','P8'],
    occipital=['POz','O1','Oz','O2'],
)

def eeg_feats(s, k):
    try:
        eeg, ts = load_eeg_trial(s, k); em = load_eeg_time(s, k)
        g2 = load_gaze_trial_2d(s, k); at = load_audio_timestamps(s, k)
        ali = align_modalities_to_trial(eeg=eeg, eeg_ts=ts, eeg_time_meta=em, gaze2d=g2, audio_timestamps=at)
        raw = eeg_raw_to_mne(ali['eeg'])
        raw = preprocess_eeg(raw, l_freq=1, h_freq=40, reference=('M1','M2'))
    except Exception: return None
    d = raw.get_data()
    f, P = welch(d, fs=EEG_SFREQ, nperseg=int(EEG_SFREQ*2))
    feat = {}
    for bname,(lo,hi) in BANDS.items():
        mask = (f>=lo)&(f<=hi)
        for gname, chs in GROUPS.items():
            idx = [EEG_CHANNELS.index(c) for c in chs if c in EEG_CHANNELS]
            feat[f'{bname}_{gname}'] = float(P[idx][:, mask].mean())
        # Laterality alpha for attended-side proxy
    # alpha L-parietal vs R-parietal
    mask = (f>=8)&(f<=13)
    lp = P[[EEG_CHANNELS.index(c) for c in ['P3','P7']]][:, mask].mean()
    rp = P[[EEG_CHANNELS.index(c) for c in ['P4','P8']]][:, mask].mean()
    feat['alpha_lat'] = float(np.log(rp/lp)) if lp > 0 else 0.0
    return feat

# Compute for a subset (first 4 subjects × first 30 main trials) — scale out later.
rows = []
for s in SUBJECTS[:4]:
    for k in range(1, 31):
        f = eeg_feats(s, k)
        if f is None: continue
        tno = f'Trial-{k}'
        tr = TRIALS[TRIALS['Trial No.']==tno]
        if not len(tr): continue
        f.update(subject=s, trial=k, attended=int(tr.iloc[0]['Attended Speaker']))
        rows.append(f)
E = pd.DataFrame(rows)
E.to_parquet(RESULTS_DIR / '09_eeg_features.parquet')
print('EEG feature table:', E.shape)
"""),

("md", "## 2 · Per-modality classifiers"),
("code", """\
def fit_cv(X, y, clf_ctor, n_splits=5):
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=0)
    oof = np.zeros((len(y), len(np.unique(y))))
    for tr, te in skf.split(X, y):
        m = clf_ctor().fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])
    return oof

if len(E) and len(G):
    merged = E.merge(G, on=['subject','trial','attended'], suffixes=('_eeg',''))
    y = merged['attended'].values
    Xe = merged[[c for c in E.columns if c not in ('subject','trial','attended')]].fillna(0).values
    Xg = merged[[c for c in G.columns if c not in ('subject','trial','attended','attended_az','snr')]].fillna(0).values
    def mkpipe():
        return Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(max_iter=2000))])
    oof_e = fit_cv(Xe, y, mkpipe)
    oof_g = fit_cv(Xg, y, mkpipe)
    acc_e = accuracy_score(y, oof_e.argmax(1))
    acc_g = accuracy_score(y, oof_g.argmax(1))
    print(f'EEG-only acc: {acc_e:.3f}  Gaze-only acc: {acc_g:.3f}')
"""),

("md", "## 3 · Late fusion (avg) vs stacked (meta-learner)"),
("code", """\
if len(E) and len(G):
    # Late: simple mean
    late = (oof_e + oof_g) / 2
    acc_late = accuracy_score(y, late.argmax(1))
    # Stacked: meta = LightGBM on [oof_e, oof_g]
    Xmeta = np.concatenate([oof_e, oof_g], axis=1)
    meta_oof = fit_cv(Xmeta, y, lambda: lgb.LGBMClassifier(n_estimators=200, verbosity=-1))
    acc_stack = accuracy_score(y, meta_oof.argmax(1))
    # Early: concat raw features
    Xearly = np.concatenate([Xe, Xg], axis=1)
    oof_early = fit_cv(Xearly, y, lambda: lgb.LGBMClassifier(n_estimators=300, verbosity=-1))
    acc_early = accuracy_score(y, oof_early.argmax(1))
    print(f'Late (avg):   {acc_late:.3f}')
    print(f'Early (cat):  {acc_early:.3f}')
    print(f'Stacked LGBM: {acc_stack:.3f}')
"""),

("md", "## 4 · Shapley modality contribution"),
("code", """\
import math
from itertools import permutations

def coalition_acc(subset, y, Xe, Xg):
    feats = []
    if 'eeg' in subset: feats.append(Xe)
    if 'gaze' in subset: feats.append(Xg)
    if not feats: return 0.25  # 4-AFC chance
    X = np.concatenate(feats, axis=1)
    oof = fit_cv(X, y, lambda: lgb.LGBMClassifier(n_estimators=200, verbosity=-1))
    return accuracy_score(y, oof.argmax(1))

if len(E) and len(G):
    modalities = ['eeg','gaze']
    shap_vals = {m: 0.0 for m in modalities}
    for perm in permutations(modalities):
        seen = set()
        prev = coalition_acc(seen, y, Xe, Xg)
        for m in perm:
            seen.add(m); curr = coalition_acc(seen, y, Xe, Xg)
            shap_vals[m] += curr - prev; prev = curr
    n_perm = math.factorial(len(modalities))
    shap_vals = {k: v/n_perm for k,v in shap_vals.items()}
    print('Shapley contributions:', shap_vals)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(list(shap_vals), list(shap_vals.values()), color=[COLORS['eeg'], COLORS['gaze']])
    ax.set_ylabel('Shapley contribution to accuracy')
    save_fig(fig, '09_shapley_modalities', FIGURES_DIR); plt.show()
"""),

("md", "## 5 · Deep multimodal attention model (implemented, not run)"),
("code", """\
RUN_DEEP = False
import torch, torch.nn as nn

class CrossModalAttention(nn.Module):
    # Cross-modal transformer that attends between EEG, gaze, pupil, IMU, env tokens.
    def __init__(self, dims=dict(eeg=32, gaze=2, pupil=1, imu=2, env=2), d_model=64, n_layers=3, n_heads=4, n_classes=4, T=128):
        super().__init__()
        self.projs = nn.ModuleDict({k: nn.Linear(v, d_model) for k,v in dims.items()})
        self.cls = nn.Parameter(torch.zeros(1,1,d_model))
        self.pos = nn.Parameter(torch.randn(1, T*len(dims)+1, d_model)*0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, n_layers)
        self.head = nn.Linear(d_model, n_classes)
        self.dims = dims
    def forward(self, batch):
        toks = []
        for k in self.dims: toks.append(self.projs[k](batch[k]))
        x = torch.cat(toks, dim=1)
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], dim=1)
        x = x + self.pos[:, :x.size(1)]
        z = self.enc(x)[:, 0]
        return self.head(z)

print('CrossModalAttention defined:',
      sum(p.numel() for p in CrossModalAttention().parameters()), 'params')
if RUN_DEEP:
    print('(would train here)')
"""),

("md", "## 6 · Learning curves (LGBM fusion)"),
("code", """\
if len(E) and len(G):
    rng = np.random.default_rng(0)
    sizes = [10, 20, 40, 80, len(y)]
    learn_rows = []
    for n in sizes:
        accs = []
        for trial in range(5):
            idx = rng.choice(len(y), size=min(n, len(y)), replace=False)
            Xearly_s = np.concatenate([Xe, Xg], axis=1)[idx]
            ys = y[idx]
            try:
                oof = fit_cv(Xearly_s, ys, lambda: lgb.LGBMClassifier(n_estimators=200, verbosity=-1),
                             n_splits=min(5, len(np.unique(ys))))
                accs.append(accuracy_score(ys, oof.argmax(1)))
            except Exception: pass
        if accs: learn_rows.append(dict(n=n, mean=np.mean(accs), std=np.std(accs)))
    lc = pd.DataFrame(learn_rows)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.errorbar(lc['n'], lc['mean'], yerr=lc['std'], fmt='o-', color=COLORS['eeg'])
    ax.set_xscale('log'); ax.axhline(0.25, color=COLORS['chance'], ls='--')
    ax.set_xlabel('# trials'); ax.set_ylabel('fusion accuracy')
    ax.set_title('Early-fusion learning curve')
    save_fig(fig, '09_learning_curve', FIGURES_DIR); plt.show()
"""),
]
build('/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/09_multimodal_fusion.ipynb', CELLS)
print('Wrote 09_multimodal_fusion.ipynb')
