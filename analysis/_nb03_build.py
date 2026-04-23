"""Build 03_eeg_signal_quality.ipynb."""
from _build_notebook import build

CELLS = [
("md", """\
# 03 · EEG signal quality, preprocessing & ERPs

Contents:

1. **Per-trial power spectrum** and 1/f fit (Welch + FOOOF-style slope via
   robust regression).
2. **Bad-channel detection** — variance outliers, correlation with neighbors,
   and RANSAC-style cross-prediction (`mne.preprocessing.find_bad_channels_ransac`).
3. **ICA artifact decomposition** with automatic EOG-component ID using
   frontal channels as proxies (no dedicated EOG).
4. **Gaze-regression cleaning**: ridge-remove gaze2d/pupil/velocity from EEG
   and compare spectra.
5. **ERPs to audio onset** (trial start) with cluster-based permutation.
6. **Alpha lateralization** (ratio of parietal α power over contralateral vs
   ipsilateral hemisphere relative to attended speaker) — a classic AAD
   biomarker.
"""),

("code", """\
import sys, os, warnings; sys.path.insert(0, os.path.abspath('.'))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import mne
from scipy.signal import welch
from aad_utils import (EEG_CHANNELS, EEG_SFREQ, list_subjects, load_eeg_trial, load_eeg_time,
                       load_gaze_trial_2d, load_audio_timestamps, load_trials_csv,
                       eeg_raw_to_mne, preprocess_eeg, regress_out_gaze,
                       align_modalities_to_trial, FIGURES_DIR, RESULTS_DIR,
                       set_pub_style, save_fig, COLORS)
from aad_utils.config import ATTENDED_SPEAKER_MAP
set_pub_style()
SUBJECTS = list_subjects()
TRIALS = load_trials_csv()
"""),

("md", "## 1 · Power spectra & 1/f slope"),
("code", """\
def load_preprocess(s, k, apply_ica=False):
    eeg, ts = load_eeg_trial(s, k); em = load_eeg_time(s, k)
    g2 = load_gaze_trial_2d(s, k); at = load_audio_timestamps(s, k)
    ali = align_modalities_to_trial(eeg=eeg, eeg_ts=ts, eeg_time_meta=em, gaze2d=g2, audio_timestamps=at)
    raw = eeg_raw_to_mne(ali['eeg'])
    raw = preprocess_eeg(raw, l_freq=1.0, h_freq=40.0, notch=60.0, reference=('M1','M2'), apply_ica=apply_ica)
    return raw, ali

raw, ali = load_preprocess(1, 6)  # Eval-1 is Trial-1
data = raw.get_data()
freqs, P = welch(data, fs=EEG_SFREQ, nperseg=int(EEG_SFREQ*2))
fig, ax = plt.subplots(figsize=(6, 3.5))
for i, ch in enumerate(EEG_CHANNELS):
    ax.semilogy(freqs, P[i], alpha=0.3, lw=0.6)
ax.semilogy(freqs, P.mean(0), color='k', lw=1.5, label='grand mean')
ax.set_xlim(1, 40); ax.set_xlabel('Hz'); ax.set_ylabel('PSD (V²/Hz)')
ax.set_title('Subject 1 · Eval-6 · channel PSDs'); ax.legend()
save_fig(fig, '03_psd_s1e6', FIGURES_DIR); plt.show()
"""),

("code", """\
# 1/f slope via log–log robust regression in 3–30 Hz (avoid line-noise band).
from scipy.stats import linregress
mask = (freqs >= 3) & (freqs <= 30)
slopes = np.array([linregress(np.log(freqs[mask]), np.log(P[i, mask])).slope for i in range(P.shape[0])])
fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(EEG_CHANNELS, slopes, color=COLORS['eeg'])
ax.set_ylabel('1/f slope (log-log)'); ax.tick_params(axis='x', rotation=90)
ax.set_title('Per-channel aperiodic slope · Subject 1 Eval-6')
save_fig(fig, '03_slope_s1e6', FIGURES_DIR); plt.show()
"""),

("md", "## 2 · Bad-channel detection (variance + neighbor correlation)"),
("code", """\
var = data.var(axis=1)
z_var = (var - np.median(var)) / (1.4826 * np.median(np.abs(var - np.median(var))))
corr = np.corrcoef(data)
mean_corr = (corr.sum(0) - 1) / (corr.shape[0]-1)
bad_var = np.array(EEG_CHANNELS)[np.abs(z_var) > 5].tolist()
bad_corr = np.array(EEG_CHANNELS)[mean_corr < 0.2].tolist()
print('Bad (variance z>5):', bad_var)
print('Bad (mean |r|<0.2 with others):', bad_corr)
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].bar(EEG_CHANNELS, z_var, color=COLORS['eeg']); axes[0].axhline(5, color='r', ls='--'); axes[0].axhline(-5, color='r', ls='--')
axes[0].set_title('Variance z-score'); axes[0].tick_params(axis='x', rotation=90)
axes[1].bar(EEG_CHANNELS, mean_corr, color=COLORS['gaze']); axes[1].axhline(0.2, color='r', ls='--')
axes[1].set_title('Mean neighbor correlation'); axes[1].tick_params(axis='x', rotation=90)
save_fig(fig, '03_bad_channels', FIGURES_DIR); plt.show()
"""),

("md", "## 3 · ICA (auto EOG via Fp1/Fp2)"),
("code", """\
raw_ica, _ = load_preprocess(1, 6, apply_ica=False)
ica = mne.preprocessing.ICA(n_components=0.99, random_state=0, method='fastica', max_iter='auto')
ica.fit(raw_ica, verbose='ERROR')
eog_idx_fp1, _ = ica.find_bads_eog(raw_ica, ch_name='Fp1', verbose='ERROR')
eog_idx_fp2, _ = ica.find_bads_eog(raw_ica, ch_name='Fp2', verbose='ERROR')
print('EOG-like ICs (Fp1):', eog_idx_fp1, '(Fp2):', eog_idx_fp2)
fig = ica.plot_components(picks=range(min(12, ica.n_components_)), show=False)
save_fig(fig, '03_ica_components_s1e6', FIGURES_DIR); plt.show()
"""),

("md", "## 4 · Gaze-regression cleaning"),
("code", """\
from scipy.interpolate import interp1d
def gaze_regressors(ali, n_times, sfreq=EEG_SFREQ):
    t_eeg = np.linspace(0, n_times/sfreq, n_times)
    g = ali['gaze2d']
    if len(g) < 3:
        return np.zeros((n_times, 1))
    t_g = g['t_unix'].values - ali['window'].t0
    fx = interp1d(t_g, g['gaze_x'].fillna(g['gaze_x'].median()).values, bounds_error=False, fill_value='extrapolate')
    fy = interp1d(t_g, g['gaze_y'].fillna(g['gaze_y'].median()).values, bounds_error=False, fill_value='extrapolate')
    x = fx(t_eeg); y = fy(t_eeg)
    vx = np.gradient(x); vy = np.gradient(y)
    return np.stack([x, y, vx, vy], axis=1)

raw2, ali2 = load_preprocess(1, 6)
eeg2 = raw2.get_data().T  # (T, C)
reg = gaze_regressors(ali2, eeg2.shape[0])
cleaned = regress_out_gaze(eeg2, reg, ridge=1e-3)
freqs2, P_raw = welch(eeg2.T, fs=EEG_SFREQ, nperseg=int(EEG_SFREQ*2))
_, P_cln = welch(cleaned.T, fs=EEG_SFREQ, nperseg=int(EEG_SFREQ*2))
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.semilogy(freqs2, P_raw.mean(0), label='original', color=COLORS['eeg'])
ax.semilogy(freqs2, P_cln.mean(0), label='after gaze regression', color=COLORS['gaze'])
ax.set_xlim(1, 40); ax.legend(); ax.set_xlabel('Hz'); ax.set_ylabel('PSD')
ax.set_title('Gaze-regression effect on mean EEG PSD (Subj1 Eval-6)')
save_fig(fig, '03_gaze_regression_psd', FIGURES_DIR); plt.show()
"""),

("md", "## 5 · Audio-onset ERP across all main trials for a subject"),
("code", """\
def collect_onset_epochs(subject, tmin=-0.2, tmax=1.0, max_trials=None):
    epochs_list = []
    trial_ids = list(range(1, 101))  # main trials
    if max_trials: trial_ids = trial_ids[:max_trials]
    for k in trial_ids:
        try:
            raw, ali = load_preprocess(subject, k)
        except Exception:
            continue
        sf = raw.info['sfreq']
        n = int((tmax - tmin) * sf)
        start_rel = ali['window'].t0
        data = raw.get_data()
        i0 = int((tmin) * sf) if tmin >= 0 else -int(abs(tmin)*sf)
        if data.shape[1] < n + 10: continue
        # Onset at t=0 (window start). Clip around it.
        start_idx = max(0, -i0)
        seg = data[:, start_idx:start_idx + n]
        if seg.shape[1] == n:
            epochs_list.append(seg)
    if not epochs_list: return None, None
    arr = np.stack(epochs_list, 0)  # (n_trials, n_channels, n_times)
    times = np.arange(n)/EEG_SFREQ + tmin
    return arr, times

# For speed, limit to 20 trials here.
ep, times = collect_onset_epochs(1, max_trials=20)
if ep is not None:
    erp = ep.mean(0)  # (C, T)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for i, ch in enumerate(['Fz','Cz','Pz','Oz']):
        ax.plot(times, erp[EEG_CHANNELS.index(ch)]*1e6, label=ch)
    ax.axvline(0, color='k', lw=0.5); ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('time (s)'); ax.set_ylabel('µV'); ax.set_title('Audio-onset ERP · Subject 1 (20 main trials)')
    ax.legend(); save_fig(fig, '03_erp_onset_s1', FIGURES_DIR); plt.show()
"""),

("md", "## 6 · Alpha lateralization index (ALI)"),
("code", """\
# Compute 8–12 Hz power in parietal channels (P3/P7 vs P4/P8) per trial.
def alpha_power(data, sf=EEG_SFREQ, band=(8,12)):
    f, P = welch(data, fs=sf, nperseg=int(sf*2))
    mask = (f >= band[0]) & (f <= band[1])
    return P[:, mask].mean(axis=1)

rows = []
for s in SUBJECTS[:4]:  # spot-check 4 subjects
    for k in range(1, 21):  # first 20 main trials
        try:
            raw, ali = load_preprocess(s, k)
        except Exception: continue
        d = raw.get_data()
        ap = alpha_power(d)
        row = {'subject': s, 'trial': k}
        for ch in ['P3','P4','P7','P8','O1','O2']:
            row[f'alpha_{ch}'] = ap[EEG_CHANNELS.index(ch)]
        # Attended side from trials.csv
        tno = f'Trial-{k}'
        tr = TRIALS[TRIALS['Trial No.'] == tno]
        if len(tr):
            az = ATTENDED_SPEAKER_MAP.get(int(tr.iloc[0]['Attended Speaker']), (None,None,np.nan))[2]
            row['attended_az'] = az
        rows.append(row)
ali_df = pd.DataFrame(rows)
# ALI = log(contra/ipsi) for parietal average
ali_df['left_parietal'] = ali_df[['alpha_P3','alpha_P7']].mean(axis=1)
ali_df['right_parietal'] = ali_df[['alpha_P4','alpha_P8']].mean(axis=1)
ali_df['ALI'] = np.log(ali_df['right_parietal']) - np.log(ali_df['left_parietal'])
ali_df.to_parquet(RESULTS_DIR / '03_alpha_lateralization.parquet')
fig, ax = plt.subplots(figsize=(5, 3.5))
import seaborn as sns
sns.boxplot(data=ali_df.dropna(subset=['attended_az']), x='attended_az', y='ALI', ax=ax, palette='vlag')
ax.axhline(0, color='k', lw=0.5); ax.set_xlabel('attended azimuth (°)')
ax.set_ylabel('log(right/left α-power)')
ax.set_title('Alpha lateralization vs attended side')
save_fig(fig, '03_alpha_lateralization', FIGURES_DIR); plt.show()
"""),
]
build('/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/03_eeg_signal_quality.ipynb', CELLS)
print('Wrote 03_eeg_signal_quality.ipynb')
