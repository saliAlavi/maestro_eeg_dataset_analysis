"""extract_vjepa: frozen V-JEPA 2 scene + gaze-foveated video embeddings per trial.

Why V-JEPA 2 (not a per-frame ImageNet CNN). The scene video is an egocentric view of a
room with LOUDSPEAKERS, not talking faces -- there is no lip motion to read, so the classic
audio-visual-speech levers do not apply here. What video *can* add is (i) absolute room-frame
orientation / foveation (which loudspeaker the subject is looking at -- experiment-side gaze is
uncalibrated, but the Tobii gaze2d is in scene-camera pixels, so a foveal crop needs no
calibration) and (ii) a self-supervised scene/motion representation to regularize the EEG
encoder at train time. V-JEPA 2 is a strong frozen video encoder for exactly that (scene +
motion structure), and we NEVER train on pixels.

For every MAIN trial k (1..100) we emit two whole-trial clip embeddings (64 uniformly-sampled
frames -> one V-JEPA 2 clip -> mean-pooled token embedding, 1024-d for ViT-L):
  * scene  : full downsampled frames                        -> scene context / head motion
  * fovea  : frames cropped around the Tobii gaze2d point   -> which loudspeaker is foveated

Output (one npz per subject, keyed by trial so content_best can join on trial_k):
  /fs/scratch/PAS2301/alialavi/cache/multimodal_aad__vjepa/s{S}.npz
    scene   (100, D) float32      fovea (100, D) float32
    trial_k (100,)   int64        present (100,) bool   attended (100,) int64

Video-folder mapping (the critical +5 offset) is handled by aad_utils.video_trial_dir; the
Tobii gaze timeline starts at ~0 like the scene video, so frame_time = idx / fps indexes gaze.

CLI:  python extract_vjepa.py --subject 3
      python extract_vjepa.py --subject 3 --model facebook/vjepa2-vitl-fpc64-256 --frames 64
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

from aad_utils import load_audio_timestamps, load_raw_gaze, video_trial_dir
from aad_utils.align import trial_window_from_audio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vjepa")

CACHE = Path("/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__vjepa")
N_MAIN = 100
DEFAULT_MODEL = "facebook/vjepa2-vitl-fpc64-256"
CROP_FRAC = 0.4                      # foveal crop = 40% of the shorter frame side
CROP_OUT = 256                       # foveal crops resized to a fixed square so frames stack


# --------------------------------------------------------------------------- frozen encoder
def load_encoder(model_id: str):
    """Return (processor, model, device, hidden_dim) for a frozen V-JEPA 2 encoder."""
    from transformers import AutoVideoProcessor, VJEPA2Model

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoVideoProcessor.from_pretrained(model_id)
    model = VJEPA2Model.from_pretrained(model_id).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    hidden = int(model.config.hidden_size)
    log.info("loaded %s on %s (hidden=%d)", model_id, dev, hidden)
    return proc, model, dev, hidden


@torch.no_grad()
def embed_clip(frames_rgb: np.ndarray, proc, model, dev) -> np.ndarray:
    """frames_rgb: (T,H,W,3) uint8 RGB -> mean-pooled token embedding (D,) float32."""
    inputs = proc(list(frames_rgb), return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    out = model(**inputs)
    hs = out.last_hidden_state            # (1, n_tokens, D)
    return hs.mean(1)[0].float().cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------- frame sampling
def _crop_about(frame: np.ndarray, gx: float, gy: float, frac: float) -> np.ndarray:
    """Crop a square foveal patch (frac * shorter-side) about normalized gaze (gx,gy)."""
    h, w = frame.shape[:2]
    half = int(frac * min(h, w) / 2)
    cx = int(np.clip(gx, 0.0, 1.0) * w)
    cy = int(np.clip(gy, 0.0, 1.0) * h)
    x0, x1 = max(0, cx - half), min(w, cx + half)
    y0, y1 = max(0, cy - half), min(h, cy + half)
    patch = frame[y0:y1, x0:x1]
    if patch.size == 0:
        patch = frame
    return cv2.resize(patch, (CROP_OUT, CROP_OUT))   # fixed size so all fovea frames stack


def sample_trial_frames(video_path: Path, gaze_df, n_frames: int, t_max: float | None = None):
    """Read n_frames uniformly across the AUDIO-ALIGNED trial window. Returns (scene, fovea)
    each (n_frames,H,W,3) RGB uint8, or (None, None) if the video is unreadable.

    t_max (seconds) trims sampling to the audio-playback window [0, t_max] instead of the whole
    mp4 -- the scene video runs ~3-5 s past audio offset (subject answering the question), which
    is off-task footage that would otherwise contaminate the whole-trial embedding. Per the
    project alignment convention (aad_utils.align) the Tobii recording start anchors to the audio
    start, so the trial window in video time is [0, audio_duration]."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if total <= 0:
        cap.release()
        return None, None
    last = total - 1
    if t_max is not None and fps > 0:                    # trim to the audio window [0, t_max]
        last = min(last, int(round(t_max * fps)))
    targets = np.linspace(0, last, n_frames).round().astype(int)
    target_set = set(int(t) for t in targets)

    # gaze2d (normalized scene-cam coords) interpolated onto frame times
    gt = gx = gy = None
    if gaze_df is not None and len(gaze_df) and "gaze2d_x" in gaze_df.columns:
        gt = gaze_df["t"].to_numpy(np.float64)
        gx = gaze_df["gaze2d_x"].to_numpy(np.float64)
        gy = gaze_df["gaze2d_y"].to_numpy(np.float64)
        ok = np.isfinite(gt) & np.isfinite(gx) & np.isfinite(gy)
        gt, gx, gy = gt[ok], gx[ok], gy[ok]
        if len(gt) == 0:
            gt = None

    scene_by_idx, fovea_by_idx = {}, {}
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i in target_set:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            scene_by_idx[i] = rgb
            if gt is not None:
                t = i / fps
                gxx = float(np.interp(t, gt, gx))
                gyy = float(np.interp(t, gt, gy))
            else:
                gxx, gyy = 0.5, 0.5          # center crop fallback (no gaze)
            fovea_by_idx[i] = _crop_about(rgb, gxx, gyy, CROP_FRAC)
        i += 1
        if i > targets.max():
            break
    cap.release()
    if not scene_by_idx:
        return None, None
    # assemble in target order, repeating last available frame for any missed index
    scene, fovea, last_s, last_f = [], [], None, None
    for t in targets:
        t = int(t)
        last_s = scene_by_idx.get(t, last_s)
        last_f = fovea_by_idx.get(t, last_f)
        if last_s is None:
            continue
        scene.append(last_s)
        fovea.append(last_f if last_f is not None else last_s)
    if len(scene) < n_frames:                # pad to exactly n_frames
        scene += [scene[-1]] * (n_frames - len(scene))
        fovea += [fovea[-1]] * (n_frames - len(fovea))
    return np.stack(scene[:n_frames]), np.stack(fovea[:n_frames])


# --------------------------------------------------------------------------- driver
def run_subject(subject: int, model_id: str, n_frames: int):
    proc, model, dev, hidden = load_encoder(model_id)
    scene = np.zeros((N_MAIN, hidden), np.float32)
    fovea = np.zeros((N_MAIN, hidden), np.float32)
    present = np.zeros(N_MAIN, bool)
    trial_k = np.arange(1, N_MAIN + 1, dtype=np.int64)

    for k in range(1, N_MAIN + 1):
        vdir = video_trial_dir(subject, k, kind="main")
        if vdir is None:
            log.warning("S%d trial %d: no video folder", subject, k)
            continue
        vid = vdir / "scenevideo.mp4"
        if not vid.exists():
            log.warning("S%d trial %d: %s missing", subject, k, vid)
            continue
        try:
            gz = load_raw_gaze(subject, k, kind="main")
        except Exception:
            gz = None
        try:                                              # audio-playback window -> trim off-task tail
            t_max = trial_window_from_audio(load_audio_timestamps(subject, k, kind="main")).duration
        except Exception:
            t_max = None
        s_frames, f_frames = sample_trial_frames(vid, gz, n_frames, t_max=t_max)
        if s_frames is None:
            log.warning("S%d trial %d: unreadable video", subject, k)
            continue
        scene[k - 1] = embed_clip(s_frames, proc, model, dev)
        fovea[k - 1] = embed_clip(f_frames, proc, model, dev)
        present[k - 1] = True
        if k % 10 == 0:
            log.info("S%d: %d/%d trials done", subject, k, N_MAIN)

    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"s{subject}.npz"
    np.savez_compressed(out, scene=scene, fovea=fovea, present=present, trial_k=trial_k,
                        model=np.array(model_id), n_frames=np.int64(n_frames))
    log.info("S%d: wrote %s (%d/%d trials with video)", subject, out, int(present.sum()), N_MAIN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--frames", type=int, default=64)
    a = ap.parse_args()
    run_subject(a.subject, a.model, a.frames)


if __name__ == "__main__":
    main()
