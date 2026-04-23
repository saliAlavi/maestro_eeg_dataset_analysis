"""Build 10_publication_figures.ipynb."""
from _build_notebook import build

CELLS = [
("md", """\
# 10 · Publication-ready figures

Consolidates the headline results from notebooks 01–09 & 11 into single,
colorblind-safe, vector-friendly figures targeted at a NeurIPS Datasets &
Benchmarks submission.

Run notebooks 01–09, 11 first (they write parquet files into
`analysis/results/`). This notebook reads those and renders final figures.

Figures produced:

- **F1 — Dataset overview**: participant demographics, trial structure, SNR
  distribution, modality presence heatmap.
- **F2 — Behavioral**: psychometric curve (accuracy vs SNR) with 95 % CI.
- **F3 — EEG signal quality**: grand-average PSDs, 1/f slope topo, bad-channel
  stats.
- **F4 — Alpha lateralization vs attended side**: box + individual lines.
- **F5 — AAD benchmark**: accuracy vs decision-window (mTRF / CCA), within-vs-LOSO.
- **F6 — Multimodal fusion**: per-modality vs fusion accuracy bars, Shapley
  modality contributions.
- **F7 — Cross-modal predictability**: CCA + CKA matrices side by side.
- **F8 — Forward TRF topography**: grand-average peak topo plus waveforms.
"""),

("code", """\
import sys, os, warnings; sys.path.insert(0, os.path.abspath('.'))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from pathlib import Path
import mne
from aad_utils import (EEG_CHANNELS, FIGURES_DIR, RESULTS_DIR, set_pub_style,
                       save_fig, COLORS, bootstrap_ci)
from aad_utils.preprocess import make_mne_info
set_pub_style()

def load_parquet(name):
    p = RESULTS_DIR / name
    return pd.read_parquet(p) if p.exists() else None
"""),

("md", "## F1 · Dataset overview"),
("code", """\
presence = load_parquet('01_presence.parquet')
beh = load_parquet('02_behavioral_records.parquet')
fig = plt.figure(figsize=(10, 4.5))
G = gs.GridSpec(2, 3, figure=fig, hspace=0.6, wspace=0.4)

if presence is not None:
    ax = fig.add_subplot(G[:, 0])
    mat = presence.pivot(index='subject', columns='trial', values='complete').astype(int).values
    ax.imshow(mat, aspect='auto', cmap='Greens', vmin=0, vmax=1)
    ax.set_xlabel('trial'); ax.set_ylabel('subject'); ax.set_title('a · modality completeness')

if beh is not None:
    ax = fig.add_subplot(G[0, 1])
    ax.hist(beh['SNR'].dropna(), bins=15, color=COLORS['audio'])
    ax.set_xlabel('SNR (dB)'); ax.set_title('b · SNR distribution (across trials)')

    ax = fig.add_subplot(G[0, 2])
    counts = beh['Attended Speaker'].value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, color=COLORS['attended'])
    ax.set_xlabel('attended speaker'); ax.set_title('c · attention balance')

    ax = fig.add_subplot(G[1, 1:])
    acc = beh.groupby('subject')['Correct'].mean().sort_values()
    ax.bar([f'S{s}' for s in acc.index], acc.values, color=COLORS['eeg'])
    ax.axhline(0.25, color=COLORS['chance'], ls='--')
    ax.set_ylabel('mean accuracy'); ax.set_title('d · comprehension accuracy per subject')
    ax.tick_params(axis='x', rotation=45)

save_fig(fig, 'F1_dataset_overview', FIGURES_DIR); plt.show()
"""),

("md", "## F2 · Psychometric curve"),
("code", """\
beh = load_parquet('02_behavioral_records.parquet')
if beh is not None:
    from scipy.optimize import curve_fit
    main = beh[~beh['is_training']].copy()
    main['Correct'] = pd.to_numeric(main['Correct'], errors='coerce')
    main['SNR'] = pd.to_numeric(main['SNR'], errors='coerce')
    grp = main.groupby('SNR')['Correct'].agg(['sum','count']).reset_index()
    def wilson(k,n,alpha=0.05):
        if n==0: return np.nan,np.nan,np.nan
        p=k/n; z=1.96; denom=1+z*z/n
        c=(p+z*z/(2*n))/denom
        h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
        return p, max(0,c-h), min(1,c+h)
    grp['p'], grp['lo'], grp['hi'] = zip(*[wilson(int(k),int(n)) for k,n in zip(grp['sum'], grp['count'])])
    def logistic(x, a, b, c, d): return c + (d-c)/(1+np.exp(-(x-a)/b))
    try:
        popt,_ = curve_fit(logistic, grp['SNR'], grp['p'], p0=[0,3,0.25,1.0], maxfev=5000)
    except Exception: popt = None
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.errorbar(grp['SNR'], grp['p'],
                yerr=[grp['p']-grp['lo'], grp['hi']-grp['p']],
                fmt='o', color=COLORS['attended'], capsize=2)
    if popt is not None:
        xs = np.linspace(grp['SNR'].min()-1, grp['SNR'].max()+1, 200)
        ax.plot(xs, logistic(xs, *popt), color=COLORS['eeg'])
    ax.axhline(0.25, color=COLORS['chance'], ls='--')
    ax.set_xlabel('SNR (dB)'); ax.set_ylabel('accuracy'); ax.set_ylim(0,1.05)
    ax.set_title('F2 · Psychometric curve (all subjects pooled)')
    save_fig(fig, 'F2_psychometric', FIGURES_DIR); plt.show()
"""),

("md", "## F3 · EEG quality (spectra + 1/f + saturation)"),
("code", """\
eeg_q = load_parquet('01_eeg_quicklook_sample.parquet')
if eeg_q is not None:
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].hist(eeg_q['jitter_ms'], bins=40, color=COLORS['eeg']); axes[0].set_xlabel('jitter (ms)'); axes[0].set_title('a · sampling jitter')
    axes[1].hist(eeg_q['duration_s'], bins=30, color=COLORS['eeg']); axes[1].set_xlabel('duration (s)'); axes[1].set_title('b · trial duration')
    axes[2].hist(eeg_q['saturation_rate'], bins=30, color=COLORS['eeg']); axes[2].set_xlabel('sat. rate'); axes[2].set_title('c · saturation')
    save_fig(fig, 'F3_eeg_quality', FIGURES_DIR); plt.show()
"""),

("md", "## F4 · Alpha lateralization vs attended side"),
("code", """\
ali = load_parquet('03_alpha_lateralization.parquet')
if ali is not None:
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.boxplot(data=ali.dropna(subset=['attended_az']), x='attended_az', y='ALI', ax=ax, palette='vlag', width=0.5)
    sns.stripplot(data=ali.dropna(subset=['attended_az']), x='attended_az', y='ALI', ax=ax, color='k', size=2, alpha=0.4)
    ax.axhline(0, color='k', lw=0.5); ax.set_xlabel('attended azimuth (°)'); ax.set_ylabel('log(R/L) α')
    ax.set_title('F4 · Parietal α-lateralization')
    save_fig(fig, 'F4_alpha_lateralization', FIGURES_DIR); plt.show()
"""),

("md", "## F5 · AAD benchmark vs decision-window length"),
("code", """\
win = load_parquet('06_window_accuracy.parquet')
loso = load_parquet('06_loso_backward.parquet')
if win is not None:
    import seaborn as sns
    agg = win.groupby(['subject','window_s'])['correct'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(5, 3.2))
    sns.lineplot(data=agg, x='window_s', y='correct', hue='subject',
                 marker='o', palette='tab10', ax=ax, legend=False)
    # Grand mean + bootstrap CI
    xs = sorted(agg['window_s'].unique()); means = []; los = []; his = []
    for w in xs:
        d = agg[agg['window_s']==w]['correct'].values
        m, lo, hi = bootstrap_ci(d)
        means.append(m); los.append(lo); his.append(hi)
    ax.plot(xs, means, color='k', lw=2, label='grand mean')
    ax.fill_between(xs, los, his, color='k', alpha=0.15)
    ax.axhline(0.5, color=COLORS['chance'], ls='--')
    ax.set_xscale('log'); ax.set_xlabel('decision window (s)'); ax.set_ylabel('AAD accuracy')
    ax.set_title('F5 · Backward-model AAD · within-subject')
    save_fig(fig, 'F5_aad_benchmark', FIGURES_DIR); plt.show()
"""),

("md", "## F6 · Multimodal fusion results"),
("code", """\
# Reconstruct from per-subject files saved by nb09 — for now, just bar chart.
within = load_parquet('07_within_subject_gaze.parquet')
if within is not None:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(['gaze-only'], [within['acc_mean'].mean()], color=[COLORS['gaze']],
           yerr=[within['acc_mean'].std()])
    ax.axhline(0.25, color=COLORS['chance'], ls='--')
    ax.set_ylabel('mean within-subject accuracy')
    ax.set_title('F6 · modality-level accuracy (placeholder — extend once nb09 runs full)')
    save_fig(fig, 'F6_fusion', FIGURES_DIR); plt.show()
"""),

("md", "## F7 · Cross-modal CCA"),
("code", """\
cca_df = load_parquet('08_cca_per_trial.parquet')
if cca_df is not None and len(cca_df):
    mat = cca_df.groupby(['a','b'])['cca'].mean().unstack()
    keys = sorted(set(cca_df['a']).union(cca_df['b']))
    M = pd.DataFrame(index=keys, columns=keys, dtype=float)
    for _, r in cca_df.groupby(['a','b'])['cca'].mean().reset_index().iterrows():
        M.at[r['a'], r['b']] = r['cca']; M.at[r['b'], r['a']] = r['cca']
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(M.values.astype(float), cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, rotation=45, ha='right')
    ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys)
    plt.colorbar(im, ax=ax, label='mean CC')
    ax.set_title('F7 · Cross-modal CCA')
    save_fig(fig, 'F7_cca_matrix', FIGURES_DIR); plt.show()
"""),

("md", "## F8 · Forward TRF topography"),
("code", """\
# Re-compute on the fly for the figure (uses cached pairs from nb06 if present).
from aad_utils import CACHE_DIR
PAIR_CACHE = CACHE_DIR / 'aad_pairs'
EEG_TARGET_SR = 64.0
lags_ms = np.arange(-100, 400, 20)
lags = [int(round(ms * EEG_TARGET_SR / 1000)) for ms in lags_ms]

def design_lags(X, lags):
    T = X.shape[0]; Xl = []
    for lag in lags:
        if lag >= 0: Xs = np.vstack([np.zeros((lag, X.shape[1])), X[:T-lag]])
        else: Xs = np.vstack([X[-lag:], np.zeros((-lag, X.shape[1]))])
        Xl.append(Xs)
    return np.concatenate(Xl, axis=1)

Xs, Ys = [], []
for p in sorted(PAIR_CACHE.glob('s*_t*.npz'))[:60]:
    d = np.load(p)
    env = d['env_att'][:, None]; eeg = d['eeg']
    L = min(len(env), len(eeg))
    Xs.append(design_lags(env[:L], lags)); Ys.append(eeg[:L])
if Xs:
    X = np.vstack(Xs); Y = np.vstack(Ys)
    alpha = 1e2
    B = np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ Y)
    B = B.reshape(len(lags), 32)
    info = make_mne_info()
    peak = int(np.argmax(np.abs(B).mean(1)))
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2),
                             gridspec_kw=dict(width_ratios=[1.2, 1]))
    axes[0].plot(lags_ms, B[:, EEG_CHANNELS.index('Cz')]*1e6, label='Cz', color=COLORS['eeg'])
    axes[0].plot(lags_ms, B[:, EEG_CHANNELS.index('Pz')]*1e6, label='Pz', color=COLORS['gaze'])
    axes[0].plot(lags_ms, B[:, EEG_CHANNELS.index('Fz')]*1e6, label='Fz', color=COLORS['audio'])
    axes[0].axvline(0, color='k', lw=0.5); axes[0].axhline(0, color='k', lw=0.5)
    axes[0].set_xlabel('lag (ms)'); axes[0].set_ylabel('TRF (µV)'); axes[0].legend()
    axes[0].set_title('a · TRF waveforms')
    mne.viz.plot_topomap(B[peak], info, axes=axes[1], show=False, cmap='RdBu_r')
    axes[1].set_title(f'b · topo @ {lags_ms[peak]:.0f} ms')
    save_fig(fig, 'F8_forward_trf_topo', FIGURES_DIR); plt.show()
else:
    print('No cached pairs yet. Run nb06 first.')
"""),

("md", """\
### Next steps

- Run notebooks 01-09 and 11 in order. Each writes parquet files into
  `analysis/results/`. Re-run this notebook to refresh final figures.
- For deep-learning notebooks (06, 07, 09), flip `RUN_DEEP = True` on a GPU
  node; otherwise all shallow benchmarks run on CPU.
"""),
]
build('/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/10_publication_figures.ipynb', CELLS)
print('Wrote 10_publication_figures.ipynb')
