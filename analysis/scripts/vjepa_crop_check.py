"""vjepa_crop_check: visual sanity check that the foveal crop lands where the subject looks.

The fovea branch in content_best assumes `_crop_about(frame, gaze2d_x, gaze2d_y)` centers a patch
on the foveated loudspeaker. Before trusting vis=0.65, verify the gaze2d coordinate convention
(origin, axis order, y-flip) and the gaze<->video time sync are right: a mis-centered crop would
make "fovea" a glorified center-crop. This dumps, per subject, a montage of several trials at
multiple time points with the gaze dot + crop box drawn on the (audio-window-trimmed) scene frame.

CLI:  python vjepa_crop_check.py --subject 5
      python vjepa_crop_check.py --subject 5 --trials 1,30,70 --per-trial 3
Output: analysis/figures/vjepa_crops/s{S}.png
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from aad_utils import (load_audio_timestamps, load_raw_gaze, load_trials_csv,
                       video_trial_dir)
from aad_utils.align import trial_window_from_audio

CROP_FRAC = 0.4                      # must match extract_vjepa.CROP_FRAC
OUT = Path(__file__).resolve().parent.parent / "figures" / "vjepa_crops"


def _grab(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None


def _annotate(rgb, gx, gy):
    """Draw the gaze dot + the foveal crop box that extract_vjepa would take."""
    h, w = rgb.shape[:2]
    cx, cy = int(np.clip(gx, 0, 1) * w), int(np.clip(gy, 0, 1) * h)
    half = int(CROP_FRAC * min(h, w) / 2)
    img = rgb.copy()
    cv2.rectangle(img, (max(0, cx - half), max(0, cy - half)),
                  (min(w, cx + half), min(h, cy + half)), (0, 255, 0), 4)
    cv2.circle(img, (cx, cy), 10, (255, 0, 0), -1)
    return img


def run(subject: int, trials: list[int], per_trial: int):
    try:
        tcsv = load_trials_csv()
    except Exception:
        tcsv = None
    rows = []
    for k in trials:
        vdir = video_trial_dir(subject, k, kind="main")
        if vdir is None or not (vdir / "scenevideo.mp4").exists():
            print(f"S{subject} trial {k}: no video", flush=True); continue
        cap = cv2.VideoCapture(str(vdir / "scenevideo.mp4"))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        try:
            t_max = trial_window_from_audio(load_audio_timestamps(subject, k, kind="main")).duration
            last = min(total - 1, int(round(t_max * fps)))
        except Exception:
            last = total - 1
        gz = None
        try:
            g = load_raw_gaze(subject, k, kind="main")
            if len(g) and "gaze2d_x" in g.columns:
                gt = g["t"].to_numpy(float); gx = g["gaze2d_x"].to_numpy(float)
                gy = g["gaze2d_y"].to_numpy(float)
                ok = np.isfinite(gt) & np.isfinite(gx) & np.isfinite(gy)
                gz = (gt[ok], gx[ok], gy[ok]) if ok.any() else None
        except Exception:
            gz = None
        panel = []
        for idx in np.linspace(0, last, per_trial + 2)[1:-1].round().astype(int):
            rgb = _grab(cap, int(idx))
            if rgb is None:
                continue
            if gz is not None:
                t = idx / fps
                gxx = float(np.interp(t, gz[0], gz[1])); gyy = float(np.interp(t, gz[0], gz[2]))
            else:
                gxx = gyy = 0.5
            panel.append(cv2.resize(_annotate(rgb, gxx, gyy), (320, 240)))
        cap.release()
        if panel:
            att = ""
            if tcsv is not None:
                try:
                    att = f" att={int(tcsv.iloc[k - 1].get('Attended Speaker', -1))}"
                except Exception:
                    att = ""
            strip = np.concatenate(panel, axis=1)
            cv2.putText(strip, f"S{subject} trial {k}{att}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            rows.append(strip)
    if not rows:
        print(f"S{subject}: nothing to render", flush=True); return
    wmax = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0))) for r in rows]
    montage = np.concatenate(rows, axis=0)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"s{subject}.png"
    cv2.imwrite(str(out), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    print(f"S{subject}: wrote {out} ({len(rows)} trials x {per_trial} frames; "
          f"green box=foveal crop, red dot=gaze2d)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--trials", default="1,25,50,75,100")
    ap.add_argument("--per-trial", type=int, default=3)
    a = ap.parse_args()
    run(a.subject, [int(x) for x in a.trials.split(",") if x.strip()], a.per_trial)


if __name__ == "__main__":
    main()
