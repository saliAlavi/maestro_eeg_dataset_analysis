"""Build 04_gaze_analysis.ipynb."""
from _build_notebook import build

CELLS = [
("md", """\
# 04 · Gaze & pupil analysis

Uses Tobii Glasses 3 raw stream (`gazedata.gz`) for per-eye 3-D gaze direction,
binocular vergence, and pupil diameter, plus IMU for head motion.

Analyses:

1. **Fixation / saccade** detection (I-VT) on gaze2d.
2. **Vergence angle** between L/R gaze vectors — proxy for depth attention
   and can help detect calibration drift.
3. **Pupil dilation as listening effort**: trial-locked pupil time-courses,
   per-SNR comparison, dilation ~ comprehension-correct (LME).
4. **Head motion (IMU)**: angular velocity magnitude, coupling to saccades
   and to gaze velocity.
5. **Gaze vs attended speaker**: does gaze cluster toward the attended source
   direction?
"""),

("code", """\
import sys, os, warnings; sys.path.insert(0, os.path.abspath('.'))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from aad_utils import (list_subjects, load_trials_csv, load_raw_gaze, load_raw_imu,
                       load_gaze_trial_2d, load_audio_timestamps, load_answers,
                       align_modalities_to_trial, FIGURES_DIR, RESULTS_DIR,
                       detect_saccades_ivt, pupil_baseline_correct,
                       set_pub_style, save_fig, COLORS)
from aad_utils.config import ATTENDED_SPEAKER_MAP
from aad_utils.io import load_eeg_trial, load_eeg_time
set_pub_style()
TRIALS = load_trials_csv()
SUBJECTS = list_subjects()
"""),

("md", "## 1 · Fixations & saccades (example trial)"),
("code", """\
s, k = 1, 6
g2 = load_gaze_trial_2d(s, k)
sacc = detect_saccades_ivt(g2['gaze_ts'].values, g2['gaze_x'].fillna(0.5).values,
                            g2['gaze_y'].fillna(0.5).values, velocity_threshold_deg_s=30)
print(f'Detected {len(sacc.onsets)} saccade onsets; median amp = {np.nanmedian(sacc.amplitudes):.2f}°, median peak-v = {np.nanmedian(sacc.velocities):.1f}°/s')
fig, ax = plt.subplots(figsize=(6, 3))
t = g2['gaze_ts'].values - g2['gaze_ts'].values[0]
ax.plot(t, g2['gaze_x'], color=COLORS['gaze'], label='x')
ax.plot(t, g2['gaze_y'], color=COLORS['audio'], label='y')
for on in (sacc.onsets - g2['gaze_ts'].values[0]):
    ax.axvline(on, color='k', lw=0.3, alpha=0.3)
ax.set_xlabel('time (s)'); ax.set_ylabel('normalized gaze'); ax.legend()
ax.set_title('Subject 1 Eval-6 · gaze2d with saccade onsets')
save_fig(fig, '04_saccades_example', FIGURES_DIR); plt.show()
"""),

("md", "## 2 · Vergence angle (per-eye 3-D gaze)"),
("code", """\
def vergence_deg(df):
    L = df[['L_dx','L_dy','L_dz']].values; R = df[['R_dx','R_dy','R_dz']].values
    Ln = L / np.linalg.norm(L, axis=1, keepdims=True)
    Rn = R / np.linalg.norm(R, axis=1, keepdims=True)
    cosang = np.clip(np.sum(Ln*Rn, axis=1), -1, 1)
    return np.degrees(np.arccos(cosang))

rg = load_raw_gaze(s, k)
ver = vergence_deg(rg)
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(rg['t'], ver, color=COLORS['pupil'], lw=0.6)
ax.set_xlabel('time (s, recording-relative)'); ax.set_ylabel('vergence angle (°)')
ax.set_title('Binocular vergence · Subject 1 Eval-6')
save_fig(fig, '04_vergence', FIGURES_DIR); plt.show()
print('Median vergence:', np.nanmedian(ver), '°')
"""),

("md", "## 3 · Pupil dilation & listening effort"),
("code", """\
def trial_pupil(subject, trial, attended_only=True):
    rg = load_raw_gaze(subject, trial)
    if rg.empty: return None
    pup = np.nanmean(rg[['L_pupil','R_pupil']].values, axis=1)
    t = rg['t'].values
    pup_corr = pupil_baseline_correct(pup, t, baseline_window=(0.0, 0.5))
    return t - t[0], pup, pup_corr

# Aggregate across first 10 main trials × 4 subjects.
curves = []
for s in SUBJECTS[:4]:
    for k in range(1, 11):
        r = trial_pupil(s, k)
        if r is None: continue
        t, p, pc = r
        if len(t) < 50: continue
        curves.append((s, k, t, pc))
# Plot median ± IQR
fig, ax = plt.subplots(figsize=(6, 3.5))
tgrid = np.linspace(0, 25, 500)
mat = []
for s, k, t, pc in curves:
    if t[-1] < 20: continue
    pc_i = np.interp(tgrid, t, pc)
    mat.append(pc_i)
mat = np.array(mat)
med = np.nanmedian(mat, axis=0); q25, q75 = np.nanpercentile(mat, [25, 75], axis=0)
ax.plot(tgrid, med, color=COLORS['pupil'])
ax.fill_between(tgrid, q25, q75, color=COLORS['pupil'], alpha=0.3)
ax.axhline(0, color='k', lw=0.5); ax.set_xlabel('time since trial start (s)')
ax.set_ylabel('baseline-corrected pupil (mm)')
ax.set_title(f'Pupil trajectory (n={len(mat)} trials)')
save_fig(fig, '04_pupil_trajectory', FIGURES_DIR); plt.show()
"""),

("md", "## 4 · IMU head motion"),
("code", """\
from numpy.linalg import norm
ri = load_raw_imu(s, k)
acc_mag = norm(ri[['ax','ay','az']].values, axis=1) - 9.81
gyr_mag = norm(ri[['gx','gy','gz']].values, axis=1)
fig, axes = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
axes[0].plot(ri['t'], acc_mag, color=COLORS['imu']); axes[0].set_ylabel('|acc|-g (m/s²)')
axes[1].plot(ri['t'], gyr_mag, color=COLORS['imu']); axes[1].set_ylabel('|gyro| (rad/s)')
axes[1].set_xlabel('time (s)'); plt.suptitle('Head motion (IMU) · Subject 1 Eval-6')
save_fig(fig, '04_imu_trial', FIGURES_DIR); plt.show()
print('median gyro:', np.median(gyr_mag), 'max gyro:', np.max(gyr_mag))
"""),

("md", "## 5 · Gaze spatial distribution vs attended direction"),
("code", """\
# Relate mean gaze_x (scene-coord [0,1]) to attended azimuth across main trials.
rows = []
for s in SUBJECTS[:6]:
    for k in range(1, 101):
        try:
            g2 = load_gaze_trial_2d(s, k)
            tno = f'Trial-{k}'
            tr = TRIALS[TRIALS['Trial No.'] == tno]
            if not len(tr): continue
            az = ATTENDED_SPEAKER_MAP[int(tr.iloc[0]['Attended Speaker'])][2]
            rows.append(dict(subject=s, trial=k, attended_az=az,
                             gx=np.nanmean(g2['gaze_x']), gy=np.nanmean(g2['gaze_y'])))
        except Exception: continue
gaze_dir = pd.DataFrame(rows)
gaze_dir.to_parquet(RESULTS_DIR / '04_gaze_vs_attended.parquet')

fig, ax = plt.subplots(figsize=(6, 3.5))
import seaborn as sns
sns.stripplot(data=gaze_dir, x='attended_az', y='gx', ax=ax, hue='subject', palette='tab10', alpha=0.6)
ax.set_xlabel('attended azimuth (°)'); ax.set_ylabel('mean gaze_x in scene')
ax.set_title('Gaze horizontal centroid vs attended direction')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
save_fig(fig, '04_gaze_vs_attended', FIGURES_DIR); plt.show()
# Corr with attended azimuth
from scipy.stats import spearmanr
rho, p = spearmanr(gaze_dir['attended_az'], gaze_dir['gx'], nan_policy='omit')
print(f'Spearman rho(attended_az, gaze_x) = {rho:.3f} (p={p:.2e})')
"""),
]
build('/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/04_gaze_analysis.ipynb', CELLS)
print('Wrote 04_gaze_analysis.ipynb')
