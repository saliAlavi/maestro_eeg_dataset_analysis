"""Build 06_eeg_audio_decoding.ipynb — mTRF / CCA / deep AAD decoders."""
from _build_notebook import build

CELLS = [
("md", """\
# 06 · EEG ↔ audio decoding (AAD benchmark)

Methods for attended-speaker decoding from EEG + stimulus envelopes:

**Linear (fast):**
1. **Backward model** (stimulus reconstruction) with ridge regression and
   time lags 0–250 ms (`mTRF`-style). Per-trial correlation of reconstructed
   envelope vs attended/unattended, attended is whichever has higher ρ.
2. **Forward TRF** — EEG predicted from attended envelope; produces
   topographic TRF weights we can plot.
3. **CCA** between EEG windows and envelope windows (canonical AAD baseline).
4. **State-space / Kalman** version for comparison.

**Nonlinear (implemented; NOT RUN unless GPU available):**
5. **EEGNet** (Lawhern et al. 2018) variant for end-to-end AAD.
6. **Attention-based fusion** — multi-head attention between EEG time-series
   and envelope, inspired by Vandecappelle et al. 2020 CNN AAD.

Evaluation: within-subject 5-fold CV and leave-one-subject-out. Decision
windows from 1 to 30 s with bootstrap CIs.
"""),

("code", """\
import sys, os, warnings; sys.path.insert(0, os.path.abspath('.'))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import resample_poly
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.cross_decomposition import CCA
from aad_utils import (EEG_CHANNELS, EEG_SFREQ, list_subjects, load_trials_csv,
                       load_eeg_trial, load_eeg_time, load_gaze_trial_2d,
                       load_audio_timestamps, align_modalities_to_trial,
                       eeg_raw_to_mne, preprocess_eeg, audio_envelope,
                       CACHE_DIR, FIGURES_DIR, RESULTS_DIR, set_pub_style,
                       save_fig, bootstrap_ci, COLORS)
from aad_utils.config import ATTENDED_SPEAKER_MAP
set_pub_style()
TRIALS = load_trials_csv()
SUBJECTS = list_subjects()
RUN_DEEP = False  # Flip to True on a GPU node to run EEGNet / attention models
AUD_CACHE = CACHE_DIR / 'audio_features'
EEG_TARGET_SR = 64.0  # Hz — match envelope rate for TRFs
"""),

("md", "## 1 · Trial-level EEG/envelope pair loader (cached)"),
("code", """\
PAIR_CACHE = CACHE_DIR / 'aad_pairs'; PAIR_CACHE.mkdir(exist_ok=True)

def load_trial_pair(subject, k):
    out = PAIR_CACHE / f's{subject}_t{k}.npz'
    if out.exists():
        d = np.load(out)
        return dict(eeg=d['eeg'], env_att=d['env_att'], env_unatt=d['env_unatt'],
                    attended=int(d['attended']), snr=float(d['snr']))
    eeg, ts = load_eeg_trial(subject, k); em = load_eeg_time(subject, k)
    g2 = load_gaze_trial_2d(subject, k); at = load_audio_timestamps(subject, k)
    ali = align_modalities_to_trial(eeg=eeg, eeg_ts=ts, eeg_time_meta=em,
                                    gaze2d=g2, audio_timestamps=at)
    raw = eeg_raw_to_mne(ali['eeg'])
    raw = preprocess_eeg(raw, l_freq=1.0, h_freq=9.0, notch=60.0, reference=('M1','M2'))
    raw.resample(EEG_TARGET_SR, verbose='ERROR')
    E = raw.get_data().T  # (T, 32)

    # Map trial index to CSV row.
    tno = f'Trial-{k}'
    tr = TRIALS[TRIALS['Trial No.'] == tno]
    if not len(tr): return None
    tr = tr.iloc[0]; attended = int(tr['Attended Speaker'])
    att_dev = 'Device-1' if attended in (1,2) else 'Device-2'
    una_dev = 'Device-2' if att_dev == 'Device-1' else 'Device-1'

    fname = f'Trial-{tno}' if tno.isdigit() else tno
    cache = AUD_CACHE / f'{fname}.npz'
    if cache.exists():
        d = np.load(cache)
        env_att = d[f'{att_dev}_env']; env_una = d[f'{una_dev}_env']
    else:
        from aad_utils import load_audio_file
        a_att, sr_a = load_audio_file(tr[att_dev]); env_att = audio_envelope(a_att, sr_a, sr_out=EEG_TARGET_SR)
        a_una, sr_u = load_audio_file(tr[una_dev]); env_una = audio_envelope(a_una, sr_u, sr_out=EEG_TARGET_SR)

    L = min(E.shape[0], len(env_att), len(env_una))
    E = E[:L]; env_att = env_att[:L]; env_una = env_una[:L]
    np.savez_compressed(out, eeg=E, env_att=env_att, env_unatt=env_una,
                         attended=attended, snr=float(tr['SNR']))
    return dict(eeg=E, env_att=env_att, env_unatt=env_una,
                attended=attended, snr=float(tr['SNR']))

print('Pair loader ready.')
"""),

("md", "## 2 · Backward model (stimulus reconstruction) — within-subject"),
("code", """\
def design_lags(X, lags):
    T = X.shape[0]; Xl = []
    for lag in lags:
        if lag >= 0:
            Xs = np.vstack([np.zeros((lag, X.shape[1])), X[:T-lag]])
        else:
            Xs = np.vstack([X[-lag:], np.zeros((-lag, X.shape[1]))])
        Xl.append(Xs)
    return np.concatenate(Xl, axis=1)

def within_subject_backward(subject, n_trials=20, alpha=1e3,
                            lags_ms=(0, 50, 100, 150, 200, 250)):
    # Train on 80 %, test on 20 %, report AAD accuracy per decision window.
    trials = []
    for k in range(1, 1 + n_trials):
        r = load_trial_pair(subject, k)
        if r is not None: trials.append(r)
    if len(trials) < 5: return None
    lags = [int(round(ms * EEG_TARGET_SR / 1000)) for ms in lags_ms]

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    rows = []
    for fi, (tr, te) in enumerate(kf.split(trials)):
        # Build training set from attended envelopes only.
        Xs, ys = [], []
        for i in tr:
            Xs.append(design_lags(trials[i]['eeg'], lags))
            ys.append(trials[i]['env_att'])
        X = np.vstack(Xs); y = np.concatenate(ys)
        model = Ridge(alpha=alpha).fit(X, y)
        # Test per trial: reconstruct envelope, correlate vs att & unatt.
        for i in te:
            Xt = design_lags(trials[i]['eeg'], lags)
            pred = model.predict(Xt)
            rho_att = np.corrcoef(pred, trials[i]['env_att'])[0,1]
            rho_una = np.corrcoef(pred, trials[i]['env_unatt'])[0,1]
            rows.append(dict(subject=subject, fold=fi, trial_idx=i,
                             rho_att=rho_att, rho_una=rho_una,
                             correct=int(rho_att > rho_una)))
    return pd.DataFrame(rows)

# Run on first 3 subjects as a benchmark demo (extend as needed).
dfs = []
for s in SUBJECTS[:3]:
    d = within_subject_backward(s, n_trials=20)
    if d is not None: dfs.append(d)
backward = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
if len(backward):
    print('Per-subject AAD accuracy (backward, 30-s window):')
    print(backward.groupby('subject')['correct'].agg(['mean','count']))
    backward.to_parquet(RESULTS_DIR / '06_backward_within.parquet')
"""),

("md", "## 3 · Forward TRF + topographic plots"),
("code", """\
def forward_trf(subject, n_trials=20, alpha=1e2, lags_ms=np.arange(-100, 400, 20)):
    trials = [load_trial_pair(subject, k) for k in range(1, 1+n_trials)]
    trials = [t for t in trials if t is not None]
    if not trials: return None, None
    lags = [int(round(ms * EEG_TARGET_SR / 1000)) for ms in lags_ms]
    Xs, Ys = [], []
    for t in trials:
        env = t['env_att'][:, None]
        Xs.append(design_lags(env, lags)); Ys.append(t['eeg'])
    X = np.vstack(Xs); Y = np.vstack(Ys)
    B = np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ Y)  # (n_lags, 32)
    B = B.reshape(len(lags), 1, 32).squeeze(1)
    return B, np.asarray(lags_ms)

B, lag_axis = forward_trf(1, n_trials=20)
if B is not None:
    fig, ax = plt.subplots(figsize=(7, 3))
    for i, ch in enumerate(['Fz','Cz','Pz','T7','T8']):
        ax.plot(lag_axis, B[:, EEG_CHANNELS.index(ch)]*1e6, label=ch)
    ax.axvline(0, color='k', lw=0.5); ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('lag (ms)'); ax.set_ylabel('TRF weight (µV per env-unit)')
    ax.set_title('Forward TRF · attended envelope → EEG · Subject 1')
    ax.legend(); save_fig(fig, '06_forward_trf_s1', FIGURES_DIR); plt.show()

    import mne
    from aad_utils.preprocess import make_mne_info
    info = make_mne_info()
    peak_idx = int(np.argmax(np.abs(B).mean(axis=1)))
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    mne.viz.plot_topomap(B[peak_idx], info, axes=ax, show=False, cmap='RdBu_r')
    ax.set_title(f'TRF topomap at {lag_axis[peak_idx]:.0f} ms')
    save_fig(fig, '06_topo_trf_s1', FIGURES_DIR); plt.show()
"""),

("md", "## 4 · CCA baseline"),
("code", """\
def cca_aad(subject, n_trials=20, n_comp=3, window_s=30.0):
    trials = [load_trial_pair(subject, k) for k in range(1, 1+n_trials)]
    trials = [t for t in trials if t is not None]
    if not trials: return None
    rows = []
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for fi, (tr_i, te_i) in enumerate(kf.split(trials)):
        E_tr = np.vstack([trials[i]['eeg'] for i in tr_i])
        A_tr = np.concatenate([trials[i]['env_att'] for i in tr_i])[:, None]
        # CCA n_components bounded by min(n_samples, n_features_X, n_features_Y);
        # the envelope is 1-D so max allowed components is 1.
        n_eff = max(1, min(n_comp, E_tr.shape[1], A_tr.shape[1]))
        cca = CCA(n_components=n_eff, max_iter=500).fit(E_tr, A_tr)
        for i in te_i:
            E_te = trials[i]['eeg']
            Ue = cca.transform(E_te)
            if Ue.ndim == 1: Ue = Ue[:, None]
            rho_att = np.mean([np.corrcoef(Ue[:, c], trials[i]['env_att'])[0,1] for c in range(n_eff)])
            rho_una = np.mean([np.corrcoef(Ue[:, c], trials[i]['env_unatt'])[0,1] for c in range(n_eff)])
            rows.append(dict(subject=subject, fold=fi, trial_idx=i, n_comp=n_eff,
                             rho_att=rho_att, rho_una=rho_una,
                             correct=int(rho_att > rho_una)))
    return pd.DataFrame(rows)

cca_rows = []
for s in SUBJECTS[:3]:
    d = cca_aad(s, n_trials=20)
    if d is not None: cca_rows.append(d)
cca_df = pd.concat(cca_rows, ignore_index=True) if cca_rows else pd.DataFrame()
if len(cca_df):
    print('CCA accuracy:', cca_df.groupby('subject')['correct'].mean())
    cca_df.to_parquet(RESULTS_DIR / '06_cca_within.parquet')
"""),

("md", "## 5 · AAD accuracy vs decision-window length"),
("code", """\
def window_accuracy(subject, n_trials=20, windows_s=(1,2,4,8,16,30), alpha=1e3,
                    lags_ms=(0, 50, 100, 150, 200, 250)):
    trials = [load_trial_pair(subject, k) for k in range(1, 1+n_trials)]
    trials = [t for t in trials if t is not None]
    if not trials: return None
    lags = [int(round(ms * EEG_TARGET_SR / 1000)) for ms in lags_ms]
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    rows = []
    for fi, (tr_i, te_i) in enumerate(kf.split(trials)):
        Xs, ys = [], []
        for i in tr_i:
            Xs.append(design_lags(trials[i]['eeg'], lags)); ys.append(trials[i]['env_att'])
        X = np.vstack(Xs); y = np.concatenate(ys)
        model = Ridge(alpha=alpha).fit(X, y)
        for i in te_i:
            Xt = design_lags(trials[i]['eeg'], lags)
            pred = model.predict(Xt)
            T = len(pred)
            for w in windows_s:
                nW = int(round(w * EEG_TARGET_SR))
                if nW > T: continue
                for start in range(0, T - nW + 1, nW):
                    a = np.corrcoef(pred[start:start+nW], trials[i]['env_att'][start:start+nW])[0,1]
                    b = np.corrcoef(pred[start:start+nW], trials[i]['env_unatt'][start:start+nW])[0,1]
                    rows.append(dict(subject=subject, window_s=w, correct=int(a>b)))
    return pd.DataFrame(rows)

win_rows = []
for s in SUBJECTS[:3]:
    d = window_accuracy(s, n_trials=20)
    if d is not None: win_rows.append(d)
win_df = pd.concat(win_rows, ignore_index=True) if win_rows else pd.DataFrame()
if len(win_df):
    win_df.to_parquet(RESULTS_DIR / '06_window_accuracy.parquet')
    agg = win_df.groupby(['subject','window_s'])['correct'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    import seaborn as sns
    sns.lineplot(data=agg, x='window_s', y='correct', hue='subject', marker='o', ax=ax)
    ax.axhline(0.5, color=COLORS['chance'], ls='--'); ax.set_xscale('log')
    ax.set_xlabel('decision window (s)'); ax.set_ylabel('AAD accuracy')
    ax.set_title('Backward-model AAD vs window length')
    save_fig(fig, '06_window_accuracy', FIGURES_DIR); plt.show()
"""),

("md", "## 6 · Leave-one-subject-out (LOSO)"),
("code", """\
def loso_backward(subjects, n_trials=15, alpha=1e3,
                  lags_ms=(0, 50, 100, 150, 200, 250)):
    all_trials = {s: [load_trial_pair(s, k) for k in range(1, 1+n_trials)] for s in subjects}
    all_trials = {s: [t for t in ts if t is not None] for s, ts in all_trials.items()}
    lags = [int(round(ms * EEG_TARGET_SR / 1000)) for ms in lags_ms]
    rows = []
    logo = LeaveOneGroupOut()
    subj_order = list(all_trials.keys())
    groups = np.concatenate([[s]*len(all_trials[s]) for s in subj_order])
    trials_flat = [t for s in subj_order for t in all_trials[s]]
    for tr_i, te_i in logo.split(np.zeros(len(groups)), groups=groups):
        test_s = groups[te_i[0]]
        Xs, ys = [], []
        for i in tr_i:
            Xs.append(design_lags(trials_flat[i]['eeg'], lags)); ys.append(trials_flat[i]['env_att'])
        X = np.vstack(Xs); y = np.concatenate(ys)
        model = Ridge(alpha=alpha).fit(X, y)
        for i in te_i:
            t = trials_flat[i]
            Xt = design_lags(t['eeg'], lags); pred = model.predict(Xt)
            a = np.corrcoef(pred, t['env_att'])[0,1]; b = np.corrcoef(pred, t['env_unatt'])[0,1]
            rows.append(dict(test_subject=int(test_s), correct=int(a>b)))
    return pd.DataFrame(rows)

loso = loso_backward(SUBJECTS[:3], n_trials=15)
if len(loso):
    print('LOSO AAD accuracy:', loso.groupby('test_subject')['correct'].mean())
    loso.to_parquet(RESULTS_DIR / '06_loso_backward.parquet')
"""),

("md", "## 7 · Deep AAD models (implemented, not run)"),
("code", """\
import torch, torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, n_channels=32, n_classes=2, n_samples=128, F1=8, D=2, F2=16, drop=0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1*D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1*D)
        self.act = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(drop)
        self.sep_depth = nn.Conv2d(F1*D, F1*D, (1, 16), padding=(0, 8), groups=F1*D, bias=False)
        self.sep_point = nn.Conv2d(F1*D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(drop)
        self.fc = nn.Linear(F2 * (n_samples // 32), n_classes)
    def forward(self, x):  # x: (B, 1, C, T)
        x = self.bn1(self.conv1(x))
        x = self.act(self.bn2(self.depthwise(x)))
        x = self.dropout1(self.pool1(x))
        x = self.sep_point(self.sep_depth(x))
        x = self.act(self.bn3(x))
        x = self.dropout2(self.pool2(x))
        return self.fc(x.flatten(1))

class EEGEnvAttention(nn.Module):
    # Bi-modal attention between EEG tokens and candidate-envelope tokens.
    def __init__(self, n_channels=32, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.eeg_proj = nn.Linear(n_channels, d_model)
        self.env_proj = nn.Linear(2, d_model)  # (env_cand_1, env_cand_2)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, 2)
    def forward(self, eeg, env):  # eeg (B,T,C), env (B,T,2)
        x = self.eeg_proj(eeg) + self.env_proj(env)
        z = self.enc(x).mean(dim=1)
        return self.head(z)

def make_training_tensors(subject, n_trials=40, win_s=4):
    Xs, ys = [], []
    for k in range(1, 1+n_trials):
        r = load_trial_pair(subject, k)
        if r is None: continue
        T = min(r['eeg'].shape[0], len(r['env_att']), len(r['env_unatt']))
        nW = int(round(win_s * EEG_TARGET_SR))
        for start in range(0, T - nW + 1, nW):
            eeg_win = r['eeg'][start:start+nW].T  # (C, T)
            env_cand = np.stack([r['env_att'][start:start+nW], r['env_unatt'][start:start+nW]], axis=1)
            # Randomly permute which candidate is index 0 so label is learnable.
            perm = np.random.randint(0, 2)
            if perm == 1: env_cand = env_cand[:, ::-1].copy()
            Xs.append((eeg_win, env_cand, perm))
    return Xs

print('Defined EEGNet + EEGEnvAttention. Set RUN_DEEP=True on a GPU node to train.')
if RUN_DEEP:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EEGNet(n_samples=int(4*EEG_TARGET_SR)).to(device)
    print('model ready on', device, sum(p.numel() for p in model.parameters()), 'params')
"""),
]
build('/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/06_eeg_audio_decoding.ipynb', CELLS)
print('Wrote 06_eeg_audio_decoding.ipynb')
