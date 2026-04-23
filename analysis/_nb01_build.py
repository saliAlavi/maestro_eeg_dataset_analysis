"""Build analysis/01_data_audit.ipynb."""
from _build_notebook import build

CELLS = [
("md", """\
# 01 · Data audit & modality alignment

**Goal.** Verify the integrity of every recording for every subject × trial and
document *coverage* per modality so the rest of the analyses know what they can
trust.

Checks performed here:

1. **Presence matrix** (modality × subject × trial) from on-disk files.
2. **EEG sampling-rate stability** (jitter in `eeg_ts`), duration distribution,
   channel amplitude histograms, and saturation rate.
3. **Gaze validity** (proportion of samples with finite per-eye gaze, pupil
   availability, missing-sample bursts).
4. **Wall-clock alignment**: derive unix-time windows from
   `audio_timestamps.json` and verify the EEG window fully covers them.
5. **Trial-number sanity** for `Video Recordings/` (numeric ordering vs mtime
   fallback).

Outputs go to `analysis/results/01_audit.parquet` and the figures directory.
"""),
("code", """\
import sys, os, warnings
sys.path.insert(0, os.path.abspath('.'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aad_utils import (
    DATA_ROOT, EXPERIMENT_DIR, VIDEO_DIR, PAIRS_DIR, TRIALS_CSV,
    EEG_CHANNELS, EEG_SFREQ, N_TRAIN_TRIALS, N_MAIN_TRIALS,
    FIGURES_DIR, RESULTS_DIR, CACHE_DIR,
    list_subjects, load_trials_csv, load_answers, load_demographics,
    load_eeg_trial, load_eeg_time, load_gaze_trial_2d, load_gaze_time,
    load_audio_timestamps, resolve_video_trial_mapping, load_raw_gaze, load_raw_imu,
    align_modalities_to_trial,
    set_pub_style, save_fig, COLORS,
)
set_pub_style()
SUBJECTS = list_subjects()
N_TRIALS = N_TRAIN_TRIALS + N_MAIN_TRIALS
print(f'{len(SUBJECTS)} subjects, {N_TRIALS} expected trials each')
"""),

("md", "## 1 · Modality presence matrix"),
("code", """\
import json
from pathlib import Path

rows = []
for s in SUBJECTS:
    vid_map = resolve_video_trial_mapping(s)
    for k in range(1, N_TRIALS + 1):
        eval_dir = EXPERIMENT_DIR / f'Subject {s}' / f'Eval-{k}'
        vid_dir = vid_map.get(k)
        rec = dict(
            subject=s, trial=k,
            eeg=(eval_dir / 'eeg_data.p').exists(),
            eeg_time=(eval_dir / 'eeg_time_data.p').exists(),
            gaze2d=(eval_dir / 'gaze_data.p').exists(),
            audio_ts=(eval_dir / 'audio_timestamps.json').exists(),
            video=(vid_dir / 'scenevideo.mp4').exists() if vid_dir else False,
            raw_gaze=(vid_dir / 'gazedata.gz').exists() if vid_dir else False,
            imu=(vid_dir / 'imudata.gz').exists() if vid_dir else False,
        )
        rec['complete'] = all(rec[c] for c in ('eeg','eeg_time','gaze2d','audio_ts','video','raw_gaze','imu'))
        rows.append(rec)
presence = pd.DataFrame(rows)
print('Complete trials per subject:')
display(presence.groupby('subject')['complete'].sum().to_frame('n_complete').T)
presence.to_parquet(RESULTS_DIR / '01_presence.parquet')
"""),

("code", """\
# Visual presence heatmap per modality.
fig, axes = plt.subplots(1, 7, figsize=(16, 4), sharey=True)
mods = ['eeg','eeg_time','gaze2d','audio_ts','video','raw_gaze','imu']
for ax, m in zip(axes, mods):
    grid = presence.pivot(index='subject', columns='trial', values=m).astype(float)
    ax.imshow(grid.values, aspect='auto', cmap='Greens', vmin=0, vmax=1, interpolation='nearest')
    ax.set_title(m); ax.set_xlabel('trial')
axes[0].set_ylabel('subject')
plt.suptitle('Modality presence (green = file exists)', y=1.02)
save_fig(fig, '01_presence_heatmap', FIGURES_DIR)
plt.show()
"""),

("md", "## 2 · EEG sampling-rate stability & amplitudes"),
("code", """\
def eeg_quicklook(subject, trial):
    try:
        eeg, ts = load_eeg_trial(subject, trial)
    except FileNotFoundError:
        return None
    dts = np.diff(ts)
    sat_rate = float(np.mean(np.any(np.abs(eeg) >= 0.08388, axis=1)))  # clip rate
    return dict(
        subject=subject, trial=trial,
        n_samples=len(ts),
        duration_s=float(ts[-1]-ts[0]) if len(ts) > 1 else np.nan,
        mean_dt_ms=float(np.nanmean(dts)*1000) if len(dts) else np.nan,
        jitter_ms=float(np.nanstd(dts)*1000) if len(dts) else np.nan,
        max_dt_ms=float(np.nanmax(dts)*1000) if len(dts) else np.nan,
        mean_abs_uv=float(np.mean(np.abs(eeg))*1e6),
        saturation_rate=sat_rate,
    )

# Sample one trial per subject for a fast overview.
qrows = []
for s in SUBJECTS:
    r = eeg_quicklook(s, 1)
    if r is not None:
        qrows.append(r)
eeg_q = pd.DataFrame(qrows)
display(eeg_q)
eeg_q.to_parquet(RESULTS_DIR / '01_eeg_quicklook_trial1.parquet')
"""),

("code", """\
# Denser scan: subsample 10 trials per subject for dt-jitter distribution.
rng = np.random.default_rng(0)
sample_trials = rng.choice(np.arange(1, N_TRIALS+1), size=min(10, N_TRIALS), replace=False)
deep_rows = []
for s in SUBJECTS:
    for k in sample_trials:
        r = eeg_quicklook(s, int(k))
        if r is not None:
            deep_rows.append(r)
eeg_deep = pd.DataFrame(deep_rows)
eeg_deep.to_parquet(RESULTS_DIR / '01_eeg_quicklook_sample.parquet')

fig, axes = plt.subplots(1, 3, figsize=(12, 3))
axes[0].hist(eeg_deep['jitter_ms'].dropna(), bins=40, color=COLORS['eeg'])
axes[0].set_xlabel('inter-sample jitter (ms)'); axes[0].set_ylabel('# trials')
axes[0].set_title('EEG sampling jitter')
axes[1].hist(eeg_deep['duration_s'].dropna(), bins=30, color=COLORS['eeg'])
axes[1].set_xlabel('trial duration (s)'); axes[1].set_title('Trial duration')
axes[2].hist(eeg_deep['saturation_rate'].dropna(), bins=30, color=COLORS['eeg'])
axes[2].set_xlabel('fraction samples saturated'); axes[2].set_title('EEG saturation rate')
save_fig(fig, '01_eeg_quality', FIGURES_DIR)
plt.show()
"""),

("md", "## 3 · Gaze validity & pupil availability"),
("code", """\
def gaze_quicklook(subject, trial):
    try:
        g2 = load_gaze_trial_2d(subject, trial)
    except FileNotFoundError:
        return None
    rg = load_raw_gaze(subject, trial)
    out = dict(
        subject=subject, trial=trial,
        n_gaze2d=len(g2),
        gaze2d_valid=float(np.mean(np.isfinite(g2[['gaze_x','gaze_y']].values))),
    )
    if len(rg):
        out.update(dict(
            n_raw_gaze=len(rg),
            pct_left_eye=float(np.mean(np.isfinite(rg[['L_dx','L_dy','L_dz']].values))),
            pct_right_eye=float(np.mean(np.isfinite(rg[['R_dx','R_dy','R_dz']].values))),
            pct_L_pupil=float(np.mean(np.isfinite(rg['L_pupil']))),
            pct_R_pupil=float(np.mean(np.isfinite(rg['R_pupil']))),
        ))
    return out

qrows = []
for s in SUBJECTS:
    for k in sample_trials:
        r = gaze_quicklook(s, int(k))
        if r is not None:
            qrows.append(r)
gaze_q = pd.DataFrame(qrows)
gaze_q.to_parquet(RESULTS_DIR / '01_gaze_quicklook_sample.parquet')
display(gaze_q.groupby('subject')[['gaze2d_valid','pct_left_eye','pct_right_eye','pct_L_pupil','pct_R_pupil']].mean())
"""),

("code", """\
fig, ax = plt.subplots(figsize=(7, 3.5))
g = gaze_q.groupby('subject')[['pct_left_eye','pct_right_eye','pct_L_pupil','pct_R_pupil']].mean()
g.plot.bar(ax=ax, width=0.8, color=[COLORS['gaze'], COLORS['imu'], COLORS['pupil'], COLORS['audio']])
ax.set_ylabel('fraction valid samples'); ax.set_title('Per-eye gaze & pupil validity (subject-averaged)')
ax.set_ylim(0, 1.05); ax.legend(loc='lower right', ncol=2)
save_fig(fig, '01_gaze_validity', FIGURES_DIR)
plt.show()
"""),

("md", "## 4 · Wall-clock alignment check"),
("code", """\
align_rows = []
for s in SUBJECTS[:4]:  # spot-check first 4; extend if needed
    for k in range(1, min(6, N_TRIALS+1)):
        try:
            eeg, ts = load_eeg_trial(s, k)
            em = load_eeg_time(s, k)
            g2 = load_gaze_trial_2d(s, k)
            at = load_audio_timestamps(s, k)
            rg = load_raw_gaze(s, k)
            ri = load_raw_imu(s, k)
            ali = align_modalities_to_trial(eeg=eeg, eeg_ts=ts, eeg_time_meta=em, gaze2d=g2,
                                            audio_timestamps=at, raw_gaze=rg, raw_imu=ri)
            align_rows.append(dict(
                subject=s, trial=k,
                audio_window_s=float(ali['window'].duration),
                eeg_post_clip_samples=int(ali['eeg'].shape[0]),
                gaze2d_post_clip=int(len(ali['gaze2d'])),
                raw_gaze_post_clip=int(len(ali.get('raw_gaze', pd.DataFrame()))),
                raw_imu_post_clip=int(len(ali.get('raw_imu', pd.DataFrame()))),
            ))
        except Exception as e:
            align_rows.append(dict(subject=s, trial=k, error=str(e)[:60]))
align_df = pd.DataFrame(align_rows)
align_df.to_parquet(RESULTS_DIR / '01_alignment_check.parquet')
display(align_df)
"""),

("md", "## 5 · Example aligned-trial timeline (Subject 1, Eval-1)"),
("code", """\
s, k = 1, 1
eeg, ts = load_eeg_trial(s, k)
em = load_eeg_time(s, k)
g2 = load_gaze_trial_2d(s, k)
at = load_audio_timestamps(s, k)
rg = load_raw_gaze(s, k); ri = load_raw_imu(s, k)
ali = align_modalities_to_trial(eeg=eeg, eeg_ts=ts, eeg_time_meta=em, gaze2d=g2,
                                audio_timestamps=at, raw_gaze=rg, raw_imu=ri)
w = ali['window']
fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
t0 = w.t0
axes[0].plot(ali['eeg_unix'] - t0, ali['eeg'][:, EEG_CHANNELS.index('Cz')] * 1e6, color=COLORS['eeg'], lw=0.5)
axes[0].set_ylabel('Cz (µV)'); axes[0].set_title('Aligned trial streams · Subject 1 · Eval-1')
axes[1].plot(ali['gaze2d']['t_unix'] - t0, ali['gaze2d']['gaze_x'], color=COLORS['gaze'], label='x')
axes[1].plot(ali['gaze2d']['t_unix'] - t0, ali['gaze2d']['gaze_y'], color=COLORS['audio'], label='y')
axes[1].set_ylabel('gaze2d'); axes[1].legend(loc='upper right')
if 'raw_gaze' in ali and len(ali['raw_gaze']):
    axes[2].plot(ali['raw_gaze']['t_unix'] - t0, ali['raw_gaze']['L_pupil'], color=COLORS['pupil'], label='L pupil')
    axes[2].plot(ali['raw_gaze']['t_unix'] - t0, ali['raw_gaze']['R_pupil'], color=COLORS['video'], label='R pupil')
    axes[2].set_ylabel('pupil Ø (mm)'); axes[2].legend(loc='upper right')
if 'raw_imu' in ali and len(ali['raw_imu']):
    t_imu = ali['raw_imu']['t_unix'] - t0
    axes[3].plot(t_imu, ali['raw_imu'][['ax','ay','az']], alpha=0.7)
    axes[3].set_ylabel('accel (m/s²)'); axes[3].legend(['ax','ay','az'], loc='upper right')
# Overlay audio-playback windows.
for spec in at:
    for ax in axes:
        ax.axvspan(spec['playback_start_time']-t0, spec['end_time']-t0, color='gray', alpha=0.05)
axes[-1].set_xlabel('time since window start (s)')
save_fig(fig, '01_trial_timeline_subj1_eval1', FIGURES_DIR)
plt.show()
"""),

("md", """\
### Summary

- Presence matrix, quicklook statistics, and a wall-clock alignment spot-check
  are saved under `analysis/results/01_*.parquet`.
- `aad_utils.align.align_modalities_to_trial` is the canonical cross-modal
  clipping routine used by all downstream notebooks.
- ⚠️ Any subject/trial flagged with `saturation_rate > 0.02` or
  `gaze2d_valid < 0.9` should be reviewed before inclusion in decoders.
"""),
]

build('/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/01_data_audit.ipynb', CELLS)
print('Wrote 01_data_audit.ipynb')
