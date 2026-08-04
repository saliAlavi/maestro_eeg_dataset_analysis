"""extract_vjepa_windowed: frozen V-JEPA2 scene+fovea embeddings on 5 s WINDOWS of each trial's
scene video (the whole-trial cache s{S}.npz has one embedding per trial; this gives 5 s resolution
so the video branch can be scored at the same window as EEG/gaze).

Reuses the encoder, foveal crop, and audio-window trimming from extract_vjepa.py; reads each trial
video ONCE, then emits one scene+fovea clip embedding per 5 s window (hop 2.5 s).

Output: /fs/scratch/PAS2301/alialavi/cache/multimodal_aad__vjepa/s{S}_win5.npz
  scene (Nw,D) fovea (Nw,D) trial_k (Nw,) win_start (Nw,)   [Nw = total windows over all trials]

CLI:  python extract_vjepa_windowed.py --subject 3
"""
from __future__ import annotations
import argparse, importlib.util, logging, os, sys
from pathlib import Path
import numpy as np, cv2

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
_spec = importlib.util.spec_from_file_location("ev", os.path.join(_here, "extract_vjepa.py"))
EV = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(EV)
from aad_utils import load_audio_timestamps, load_raw_gaze, video_trial_dir      # noqa: E402
from aad_utils.align import trial_window_from_audio                              # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vjepaW")
CACHE = Path("/fs/scratch/PAS2301/alialavi/cache/multimodal_aad__vjepa")
N_MAIN = 100; WIN_S = 5.0; HOP_S = 2.5; NF = 64; MAXDIM = 320


def read_trial(video_path, gaze_df, t_max):
    """Read all frames up to t_max (downsampled); return frames(list RGB), times(np), gx,gy(per-frame)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    last = int(round((t_max if t_max else 1e9) * fps))
    gt = gx = gy = None
    if gaze_df is not None and len(gaze_df) and "gaze2d_x" in gaze_df.columns:
        gt = gaze_df["t"].to_numpy(np.float64); gx = gaze_df["gaze2d_x"].to_numpy(np.float64)
        gy = gaze_df["gaze2d_y"].to_numpy(np.float64)
        ok = np.isfinite(gt) & np.isfinite(gx) & np.isfinite(gy); gt, gx, gy = gt[ok], gx[ok], gy[ok]
        if len(gt) == 0:
            gt = None
    frames, times, gxs, gys = [], [], [], []
    i = 0
    while i <= last:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]; sc = MAXDIM / max(h, w)
        if sc < 1:
            frame = cv2.resize(frame, (int(w * sc), int(h * sc)))
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); t = i / fps; times.append(t)
        if gt is not None:
            gxs.append(float(np.interp(t, gt, gx))); gys.append(float(np.interp(t, gt, gy)))
        else:
            gxs.append(0.5); gys.append(0.5)
        i += 1
    cap.release()
    if len(frames) < 2:
        return None
    return frames, np.array(times), np.array(gxs), np.array(gys)


def window_clip(frames, times, gxs, gys, t_lo, t_hi):
    """64-frame scene+fovea clip uniformly sampled from frames in [t_lo, t_hi]."""
    idx = np.flatnonzero((times >= t_lo) & (times < t_hi))
    if len(idx) < 2:
        return None
    sel = idx[np.linspace(0, len(idx) - 1, NF).round().astype(int)]
    scene = np.stack([frames[i] for i in sel])
    fovea = np.stack([EV._crop_about(frames[i], gxs[i], gys[i], EV.CROP_FRAC) for i in sel])
    return scene, fovea


def run_subject(subject, model_id):
    proc, model, dev, hidden = EV.load_encoder(model_id)
    SC, FO, TK, WS = [], [], [], []
    for k in range(1, N_MAIN + 1):
        vdir = video_trial_dir(subject, k, kind="main")
        if vdir is None or not (vdir / "scenevideo.mp4").exists():
            continue
        try:
            gz = load_raw_gaze(subject, k, kind="main")
        except Exception:
            gz = None
        try:
            t_max = trial_window_from_audio(load_audio_timestamps(subject, k, kind="main")).duration
        except Exception:
            t_max = None
        rt = read_trial(vdir / "scenevideo.mp4", gz, t_max)
        if rt is None:
            log.warning("S%d trial %d unreadable", subject, k); continue
        frames, times, gxs, gys = rt; tend = times[-1]
        starts = np.arange(0, max(WIN_S, tend - WIN_S) + 1e-6, HOP_S)
        starts = [s for s in starts if s + WIN_S <= tend + 1e-6] or [0.0]
        for s0 in starts:
            clip = window_clip(frames, times, gxs, gys, s0, s0 + WIN_S)
            if clip is None:
                continue
            SC.append(EV.embed_clip(clip[0], proc, model, dev))
            FO.append(EV.embed_clip(clip[1], proc, model, dev))
            TK.append(k); WS.append(float(s0))
        if k % 10 == 0:
            log.info("S%d: %d/%d trials, %d windows", subject, k, N_MAIN, len(TK))
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"s{subject}_win5.npz"
    np.savez_compressed(out, scene=np.stack(SC).astype(np.float32), fovea=np.stack(FO).astype(np.float32),
                        trial_k=np.array(TK, np.int64), win_start=np.array(WS, np.float32),
                        model=np.array(model_id))
    log.info("S%d: wrote %s (%d windows)", subject, out, len(TK))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--model", default=EV.DEFAULT_MODEL)
    a = ap.parse_args()
    run_subject(a.subject, a.model)
