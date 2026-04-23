"""Build 11_scene_video.ipynb — egocentric scene-video analysis."""
from _build_notebook import build

CELLS = [
("md", """\
# 11 · Scene-video (egocentric) analysis

Even though the Tobii scene video is head-mounted and the experimenter room
is relatively static, the video still carries valuable information:

1. **Motion energy** per frame — correlates with head movement and with
   moments where the subject looks around / between speakers.
2. **Optical-flow statistics** (mean magnitude, direction histogram).
3. **Gaze-contingent scene patch**: 64×64 crop around the fixation, for
   downstream representation learning (we extract but do not train here).
4. **Face / object detection** using OpenCV's pretrained Haar cascade + YOLOv5
   (optional heavy; guarded).
5. Correlations between video motion energy and EEG / pupil / IMU / envelope.

Only a short slice of one trial is computed here by default (30 s video at
30 fps is ~900 frames); set `N_FRAMES = None` to process all frames.
"""),

("code", """\
import sys, os, warnings; sys.path.insert(0, os.path.abspath('.'))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import cv2
from pathlib import Path
from aad_utils import (list_subjects, load_trials_csv, video_trial_dir, load_raw_gaze,
                       load_raw_imu, load_eeg_trial, load_eeg_time, load_gaze_trial_2d,
                       load_audio_timestamps, align_modalities_to_trial, eeg_raw_to_mne,
                       preprocess_eeg, audio_envelope, load_audio_file,
                       FIGURES_DIR, RESULTS_DIR, CACHE_DIR, set_pub_style, save_fig, COLORS)
set_pub_style()
N_FRAMES = 300  # cap processed frames for quick demos
"""),

("md", "## 1 · Motion energy"),
("code", """\
def motion_energy(video_path, n_frames=None, stride=1):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    ok, prev = cap.read()
    if not ok: return None, None
    prev_g = cv2.cvtColor(cv2.resize(prev, (256, 144)), cv2.COLOR_BGR2GRAY)
    energies = []; i = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if stride > 1 and i % stride: i += 1; continue
        g = cv2.cvtColor(cv2.resize(frame, (256, 144)), cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(g, prev_g).astype(np.float32).mean()
        energies.append(diff)
        prev_g = g; i += 1
        if n_frames and i >= n_frames: break
    cap.release()
    return np.array(energies), fps

s, k = 1, 6
vdir = video_trial_dir(s, k); vp = vdir / 'scenevideo.mp4'
me, fps = motion_energy(vp, n_frames=N_FRAMES)
print(f'frames analysed: {len(me)}  fps: {fps:.1f}')
fig, ax = plt.subplots(figsize=(7, 2.5))
ax.plot(np.arange(len(me))/fps, me, color=COLORS['video'])
ax.set_xlabel('time (s)'); ax.set_ylabel('motion energy (|Δ pixels|)')
ax.set_title(f'Scene motion energy · Subject {s} Eval-{k}')
save_fig(fig, '11_motion_energy', FIGURES_DIR); plt.show()
"""),

("md", "## 2 · Dense optical flow statistics"),
("code", """\
def flow_stats(video_path, n_frames=120):
    cap = cv2.VideoCapture(str(video_path))
    ok, prev = cap.read()
    if not ok: return None
    prev_g = cv2.cvtColor(cv2.resize(prev, (320, 180)), cv2.COLOR_BGR2GRAY)
    rows = []; i = 0
    while i < n_frames:
        ok, frame = cap.read()
        if not ok: break
        g = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_g, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        rows.append(dict(frame=i, flow_mag_mean=float(mag.mean()),
                         flow_mag_std=float(mag.std()),
                         flow_dir_hist_0=float(np.mean(np.cos(ang))),
                         flow_dir_hist_90=float(np.mean(np.sin(ang)))))
        prev_g = g; i += 1
    cap.release()
    return pd.DataFrame(rows)

fs = flow_stats(vp, n_frames=N_FRAMES)
print(fs.describe() if fs is not None else 'No flow')
fs.to_parquet(RESULTS_DIR / '11_flow_stats_s1e6.parquet')
"""),

("md", "## 3 · Gaze-contingent patches"),
("code", """\
def gaze_patches(video_path, gaze_df, n_frames=50, patch=64):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    patches = []; i = 0
    while i < n_frames:
        ok, frame = cap.read()
        if not ok: break
        h, w = frame.shape[:2]
        t = i / fps
        row = gaze_df[np.abs(gaze_df['t'] - t) < 0.05]
        if len(row):
            cx = int(np.clip(row.iloc[0]['gaze2d_x'], 0, 1) * w)
            cy = int(np.clip(row.iloc[0]['gaze2d_y'], 0, 1) * h)
        else:
            cx, cy = w//2, h//2
        x0 = max(0, cx - patch//2); y0 = max(0, cy - patch//2)
        x1 = min(w, x0 + patch); y1 = min(h, y0 + patch)
        p = frame[y0:y1, x0:x1]
        if p.shape[:2] == (patch, patch): patches.append(p)
        i += 1
    cap.release()
    return np.stack(patches, 0) if patches else None

rg = load_raw_gaze(s, k)
patches = gaze_patches(vp, rg, n_frames=N_FRAMES)
if patches is not None:
    fig, axes = plt.subplots(4, 8, figsize=(10, 5))
    for ax, p in zip(axes.ravel(), patches[::max(1, len(patches)//32)]):
        ax.imshow(cv2.cvtColor(p, cv2.COLOR_BGR2RGB)); ax.axis('off')
    plt.suptitle('Gaze-contingent patches · Subject 1 Eval-6')
    save_fig(fig, '11_gaze_patches', FIGURES_DIR); plt.show()
"""),

("md", "## 4 · Face / object detection (lightweight, OpenCV Haar)"),
("code", """\
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cc = cv2.CascadeClassifier(cascade_path)

def count_faces(video_path, n_frames=100):
    cap = cv2.VideoCapture(str(video_path))
    counts = []; i = 0
    while i < n_frames:
        ok, frame = cap.read()
        if not ok: break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cc.detectMultiScale(g, 1.2, 4, minSize=(50, 50))
        counts.append(len(faces)); i += 1
    cap.release()
    return np.array(counts)

faces = count_faces(vp, n_frames=N_FRAMES)
print('Frames with detected faces:', int((faces > 0).sum()), '/', len(faces))
"""),

("md", "## 5 · Correlate motion energy with EEG band power & envelopes"),
("code", """\
# Compare motion-energy sampled at video fps vs EEG alpha power and audio envelope.
eeg, ts = load_eeg_trial(s, k); em = load_eeg_time(s, k)
g2 = load_gaze_trial_2d(s, k); at = load_audio_timestamps(s, k)
ali = align_modalities_to_trial(eeg=eeg, eeg_ts=ts, eeg_time_meta=em, gaze2d=g2, audio_timestamps=at)
raw = eeg_raw_to_mne(ali['eeg'])
raw = preprocess_eeg(raw, l_freq=1, h_freq=40, reference=('M1','M2'))
raw.filter(8, 13, verbose='ERROR')
alpha = raw.get_data().mean(0) ** 2  # squared voltage proxy
from scipy.signal import resample_poly
# Sample alpha to video fps
target_n = len(me)
alpha_rs = resample_poly(alpha, target_n, len(alpha))
env = None
from aad_utils.io import load_audio_file
tr = load_trials_csv().iloc[5]
a, sr_a = load_audio_file(tr['Device-1'])
env = audio_envelope(a, sr_a, sr_out=fps)
L = min(len(me), len(alpha_rs), len(env))
M = np.corrcoef(np.stack([me[:L], alpha_rs[:L], env[:L]]))
print('Corr matrix (motion, alpha, env):\\n', np.round(M, 3))
"""),
]
build('/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/11_scene_video.ipynb', CELLS)
print('Wrote 11_scene_video.ipynb')
