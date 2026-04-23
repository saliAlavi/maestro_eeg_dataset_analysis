"""Build 08_cross_modal_predictability.ipynb."""
from _build_notebook import build

CELLS = [
("md", """\
# 08 · Cross-modal predictability

Quantify, per trial and per subject, how much of one modality is predictable
from another. This is the core of the dataset paper's claim that "useful
information is extractable from EEG and correlated with each modality."

Measures computed:

1. **Canonical correlation** between pairs of modalities (EEG ↔ audio env /
   mel / gaze / pupil / IMU / video motion energy).
2. **Mutual information** (histogram-based and nearest-neighbour KSG
   estimator) between feature summaries.
3. **Granger causality** (linear VAR) between pairs of down-sampled
   time series.
4. **Transfer entropy** (binary partitioning, Schreiber estimator) —
   non-linear counterpart to Granger.
5. **Centered kernel alignment (CKA)** between deep-feature embeddings of
   each modality (learned separately by small auto-encoders — stubbed for
   GPU execution).
6. Heatmap / matrix of all pairwise predictability scores, suitable for a
   single figure in the paper.
"""),

("code", """\
import sys, os, warnings; sys.path.insert(0, os.path.abspath('.'))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.signal import resample_poly
from sklearn.cross_decomposition import CCA
from sklearn.feature_selection import mutual_info_regression
from aad_utils import (EEG_CHANNELS, EEG_SFREQ, list_subjects, load_trials_csv,
                       load_eeg_trial, load_eeg_time, load_gaze_trial_2d, load_audio_timestamps,
                       load_raw_gaze, load_raw_imu, align_modalities_to_trial,
                       eeg_raw_to_mne, preprocess_eeg, audio_envelope, load_audio_file,
                       CACHE_DIR, FIGURES_DIR, RESULTS_DIR, set_pub_style, save_fig, COLORS)
from aad_utils.config import ATTENDED_SPEAKER_MAP
set_pub_style()
TRIALS = load_trials_csv(); SUBJECTS = list_subjects()
SR = 64.0  # common rate
"""),

("md", "## 1 · Multi-modal tensor loader (64 Hz common rate)"),
("code", """\
from scipy.interpolate import interp1d

def resample_to(t, x, out_t):
    if len(t) < 2: return np.full(len(out_t), np.nan)
    f = interp1d(t, x, bounds_error=False, fill_value=np.nan)
    return f(out_t)

def load_multimodal(subject, k):
    eeg, ts = load_eeg_trial(subject, k); em = load_eeg_time(subject, k)
    g2 = load_gaze_trial_2d(subject, k); at = load_audio_timestamps(subject, k)
    rg = load_raw_gaze(subject, k); ri = load_raw_imu(subject, k)
    ali = align_modalities_to_trial(eeg=eeg, eeg_ts=ts, eeg_time_meta=em, gaze2d=g2,
                                     audio_timestamps=at, raw_gaze=rg, raw_imu=ri)
    raw = eeg_raw_to_mne(ali['eeg'])
    raw = preprocess_eeg(raw, l_freq=1, h_freq=30, reference=('M1','M2'))
    raw.resample(SR, verbose='ERROR')
    eeg_T = raw.get_data().T
    T = eeg_T.shape[0]
    out_t = np.linspace(0, T/SR, T)
    # gaze2d
    g = ali['gaze2d']
    gt = g['t_unix'].values - ali['window'].t0
    gx = resample_to(gt, g['gaze_x'].fillna(0.5).values, out_t)
    gy = resample_to(gt, g['gaze_y'].fillna(0.5).values, out_t)
    # pupil (from raw_gaze)
    rg = ali.get('raw_gaze', pd.DataFrame())
    if len(rg):
        rt = rg['t_unix'].values - ali['window'].t0
        pup = resample_to(rt, np.nanmean(rg[['L_pupil','R_pupil']].values, axis=1), out_t)
    else:
        pup = np.full(T, np.nan)
    # IMU
    ri = ali.get('raw_imu', pd.DataFrame())
    if len(ri):
        it = ri['t_unix'].values - ali['window'].t0
        gyro_mag = np.linalg.norm(ri[['gx','gy','gz']].values, axis=1)
        acc_mag = np.linalg.norm(ri[['ax','ay','az']].values, axis=1) - 9.81
        gyro = resample_to(it, gyro_mag, out_t); acc = resample_to(it, acc_mag, out_t)
    else:
        gyro = np.full(T, np.nan); acc = np.full(T, np.nan)
    # audio envelopes (attended + unattended)
    tno = f'Trial-{k}'
    tr = TRIALS[TRIALS['Trial No.']==tno]
    env_att = env_una = np.zeros(T)
    if len(tr):
        tr = tr.iloc[0]
        att_dev = 'Device-1' if int(tr['Attended Speaker']) in (1,2) else 'Device-2'
        una_dev = 'Device-2' if att_dev == 'Device-1' else 'Device-1'
        a_att, sr = load_audio_file(tr[att_dev])
        env_att = audio_envelope(a_att, sr, sr_out=SR)[:T]
        a_una, sr = load_audio_file(tr[una_dev])
        env_una = audio_envelope(a_una, sr, sr_out=SR)[:T]
        env_att = np.pad(env_att, (0, max(0, T-len(env_att))))[:T]
        env_una = np.pad(env_una, (0, max(0, T-len(env_una))))[:T]
    return dict(eeg=eeg_T, gx=gx, gy=gy, pupil=pup, gyro=gyro, acc=acc,
                env_att=env_att, env_una=env_una, sr=SR, subject=subject, trial=k)

demo = load_multimodal(1, 6)
print({k: (v.shape if hasattr(v,'shape') else v) for k,v in demo.items() if not isinstance(v, (int,float))})
"""),

("md", "## 2 · CCA between modalities"),
("code", """\
def cca_pair(X, Y, n_comp=3):
    X = np.atleast_2d(X); Y = np.atleast_2d(Y)
    if X.ndim == 1: X = X[:, None]
    if Y.ndim == 1: Y = Y[:, None]
    L = min(len(X), len(Y))
    X, Y = X[:L], Y[:L]
    mask = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(Y), axis=1)
    if mask.sum() < 200: return np.nan
    n_eff = max(1, min(n_comp, X.shape[1], Y.shape[1], mask.sum() - 1))
    cca = CCA(n_components=n_eff).fit(X[mask], Y[mask])
    Xc, Yc = cca.transform(X[mask], Y[mask])
    if Xc.ndim == 1: Xc = Xc[:, None]
    if Yc.ndim == 1: Yc = Yc[:, None]
    return float(np.nanmean([np.corrcoef(Xc[:, i], Yc[:, i])[0, 1] for i in range(Xc.shape[1])]))

def modality_dict(d):
    return dict(
        eeg=d['eeg'],
        gaze=np.stack([d['gx'], d['gy']], axis=1),
        pupil=d['pupil'][:, None],
        imu=np.stack([d['gyro'], d['acc']], axis=1),
        audio_att=d['env_att'][:, None],
        audio_una=d['env_una'][:, None],
    )

mods = modality_dict(demo)
keys = list(mods.keys())
mat = np.full((len(keys), len(keys)), np.nan)
for i, a in enumerate(keys):
    for j, b in enumerate(keys):
        if i >= j: continue
        mat[i, j] = mat[j, i] = cca_pair(mods[a], mods[b])
fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(mat, cmap='viridis', vmin=0, vmax=1)
ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, rotation=45)
ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys)
for i in range(len(keys)):
    for j in range(len(keys)):
        if i != j and np.isfinite(mat[i,j]):
            ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center', color='white' if mat[i,j]<0.5 else 'black', fontsize=7)
plt.colorbar(im, ax=ax, label='mean CC (first 3 comp)')
ax.set_title('Pairwise CCA (Subject 1 · Eval-6)')
save_fig(fig, '08_cca_matrix_s1e6', FIGURES_DIR); plt.show()
"""),

("md", "## 3 · Mutual information (summary statistics)"),
("code", """\
def mi_matrix(mods):
    # All arrays are sampled on the same time grid by load_multimodal, but may
    # differ in length when a short audio clip produces a shorter envelope.
    # Truncate every pair to the shared prefix before masking NaNs jointly.
    keys = list(mods.keys()); n = len(keys)
    out = np.full((n, n), np.nan)
    for i, a in enumerate(keys):
        Xa = np.atleast_2d(mods[a])
        if Xa.ndim == 1: Xa = Xa[:, None]
        for j, b in enumerate(keys):
            if i == j: continue
            Yb = np.atleast_2d(mods[b])
            if Yb.ndim == 1: Yb = Yb[:, None]
            L = min(len(Xa), len(Yb))
            Xs = Xa[:L]; Ys = Yb[:L]
            mask = np.all(np.isfinite(Xs), axis=1) & np.all(np.isfinite(Ys), axis=1)
            if mask.sum() < 200:
                continue
            feats = Xs[mask]
            target = Ys[mask, 0]  # predict first column of Y
            mi = mutual_info_regression(feats, target, random_state=0).mean()
            out[i, j] = float(mi)
    return out, keys

M, keys = mi_matrix(mods)
fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(M, cmap='magma')
ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, rotation=45)
ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys)
plt.colorbar(im, ax=ax, label='MI (nats, row → col[0])')
ax.set_title('Mutual information (Subject 1 · Eval-6)')
save_fig(fig, '08_mi_matrix_s1e6', FIGURES_DIR); plt.show()
"""),

("md", "## 4 · Granger causality (VAR)"),
("code", """\
from statsmodels.tsa.stattools import grangercausalitytests

def grangers(x, y, max_lag=5):
    # Test whether y Granger-causes x.
    df = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(df) < 50: return np.nan
    try:
        res = grangercausalitytests(df[['x','y']], maxlag=max_lag, verbose=False)
        # Smallest p-value over lags
        return float(min(res[l][0]['ssr_ftest'][1] for l in res))
    except Exception: return np.nan

targets = ['env_att','env_una','gx','pupil','gyro']
eeg_mean = demo['eeg'][:, EEG_CHANNELS.index('Cz')]
g_rows = []
for t in targets:
    p = grangers(eeg_mean, demo[t] if demo[t].ndim==1 else demo[t][:,0])
    g_rows.append(dict(source='Cz', target=t, p_value=p))
pd.DataFrame(g_rows)
"""),

("md", "## 5 · Binary transfer entropy"),
("code", """\
def binarize(x, median=None):
    x = np.asarray(x)
    mask = np.isfinite(x)
    if median is None: median = np.nanmedian(x[mask])
    return (x > median).astype(int), mask

def transfer_entropy(src, tgt, k=1):
    sx, m1 = binarize(src); tx, m2 = binarize(tgt)
    m = m1 & m2
    sx, tx = sx[m], tx[m]
    if len(sx) < k+2: return np.nan
    # States: (tx_{t}, tx_{t-1..t-k}, sx_{t-1..t-k})
    T = len(sx)
    joint = np.zeros((2, 2, 2))   # tx_next, tx_prev, sx_prev
    for i in range(k, T-1):
        joint[tx[i+1], tx[i], sx[i]] += 1
    joint /= joint.sum()
    # TE(S→T) = Σ p(t+,t,s) log[ p(t+|t,s)/p(t+|t) ]
    te = 0.0
    for a in range(2):
        for b in range(2):
            for c in range(2):
                p_abc = joint[a,b,c]
                if p_abc == 0: continue
                p_bc = joint[:,b,c].sum(); p_b = joint[:,b,:].sum()
                p_ab = joint[a,b,:].sum()
                num = p_abc / p_bc if p_bc > 0 else 0
                den = p_ab / p_b if p_b > 0 else 1e-12
                if num > 0 and den > 0:
                    te += p_abc * np.log(num / den)
    return float(te)

for t in targets:
    te = transfer_entropy(demo[t] if demo[t].ndim == 1 else demo[t][:,0], eeg_mean)
    print(f'TE({t} → Cz) = {te:.4f}')
"""),

("md", "## 6 · CKA on learned embeddings (stub)"),
("code", """\
def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    XtY = np.linalg.norm(X.T @ Y, 'fro') ** 2
    XtX = np.linalg.norm(X.T @ X, 'fro')
    YtY = np.linalg.norm(Y.T @ Y, 'fro')
    return float(XtY / (XtX * YtY + 1e-12))

# Use PCA-reduced projections of each modality as a cheap proxy for learned embeddings.
from sklearn.decomposition import PCA
def project(X, k=8):
    mask = np.all(np.isfinite(X), axis=1)
    if mask.sum() < k+5: return None
    X_ = X[mask]; k_eff = min(k, X_.shape[1])
    return PCA(n_components=k_eff).fit_transform(X_), mask

emb = {k: project(v) for k,v in mods.items()}
keys = [k for k,v in emb.items() if v is not None]
cka_mat = np.full((len(keys), len(keys)), np.nan)
for i, a in enumerate(keys):
    for j, b in enumerate(keys):
        Xa, ma = emb[a]; Xb, mb = emb[b]
        m = ma & mb
        if m.sum() < 50: continue
        cka_mat[i, j] = cka(Xa[m[ma]], Xb[m[mb]])
fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(cka_mat, cmap='Greens', vmin=0, vmax=1)
ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, rotation=45)
ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys)
plt.colorbar(im, ax=ax, label='linear CKA')
ax.set_title('CKA on 8-D PCA embeddings')
save_fig(fig, '08_cka_matrix', FIGURES_DIR); plt.show()
"""),

("md", "## 7 · Aggregate pairwise predictability (many trials)"),
("code", """\
agg_rows = []
for s in SUBJECTS[:3]:
    for k in range(1, 11):  # first 10 main trials
        try:
            d = load_multimodal(s, k); m = modality_dict(d)
            for a in m:
                for b in m:
                    if a >= b: continue
                    agg_rows.append(dict(subject=s, trial=k, a=a, b=b, cca=cca_pair(m[a], m[b])))
        except Exception as e:
            pass
AG = pd.DataFrame(agg_rows)
AG.to_parquet(RESULTS_DIR / '08_cca_per_trial.parquet')
summary = AG.groupby(['a','b'])['cca'].agg(['mean','std','count']).reset_index()
print(summary.sort_values('mean', ascending=False).head(20))
"""),
]
build('/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/08_cross_modal_predictability.ipynb', CELLS)
print('Wrote 08_cross_modal_predictability.ipynb')
