"""Windowed scene-video features for MAESTRO-Net.

Why this is shaped the way it is. The Tobii scene video is an egocentric view
of a quiet room with *loudspeakers, not talking faces* -- so there is no lip
motion to decode and most pixel content is redundant with the IMU (head
motion) and gaze. Training a 3-D CNN on 16 subjects would overfit instantly.
We therefore never train on pixels; the video stream is two cheap, honest
parts, both computed on the model's decision-window grid:

  1. Engineered motion/flow stats (dense optical flow + frame differencing),
     extending scripts/video_features.py from per-trial to per-window. Captures
     head dynamics.
  2. A FROZEN pretrained image encoder (torchvision ResNet-18 by default; CLIP
     if installed) run on ~1 frame/s, optionally cropped around the projected
     gaze point, mean-pooled per window then PCA-reduced. This is the one thing
     video adds over IMU: absolute orientation / foveation in the room frame.

The two parts are concatenated and L2-normed into a fixed ``video_dim`` vector
per window, cached to ``results/video_embeddings/s{subj}.npz`` so training reads
precomputed arrays. The encoder is frozen, so this is a one-off offline pass
(CPU-feasible but slow; run on a GPU node for speed).

CLI:
    python video_embeddings.py --subject 3                 # ResNet-18, no crop
    python video_embeddings.py --subject 3 --backbone clip --gaze-crop
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from aad_utils import RESULTS_DIR, load_trials_csv, video_trial_dir, trial_name

DS = (160, 90)            # downsample for flow stats (matches video_features.py)
FRAME_HZ = 1.0            # frames/s fed to the frozen encoder
ENGINEERED_DIM = 6        # motion_energy mean/std/peak + flow mag mean/std + dir consistency
EMBED_PCA_DIM = 10        # frozen-encoder embedding compressed to this many dims


# ----------------------------------------------------------------------------
# Frozen encoder
# ----------------------------------------------------------------------------
def _load_backbone(name: str):
    """Return (transform_fn, encode_fn, raw_dim) for a frozen image encoder, or
    None if unavailable (caller falls back to engineered-only features)."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if name == "clip":
            import open_clip
            model, _, prep = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k")
            model = model.visual.eval().to(dev)
            for p in model.parameters():
                p.requires_grad = False

            def encode(imgs):  # imgs: list of HxWx3 uint8 RGB
                import torch as _t
                from PIL import Image
                x = _t.stack([prep(Image.fromarray(im)) for im in imgs]).to(dev)
                with _t.no_grad():
                    return model(x).cpu().numpy()
            return encode, 512
        else:  # torchvision resnet18
            import torchvision
            from torchvision import transforms
            net = torchvision.models.resnet18(weights="IMAGENET1K_V1")
            net.fc = torch.nn.Identity()
            net = net.eval().to(dev)
            for p in net.parameters():
                p.requires_grad = False
            tf = transforms.Compose([
                transforms.ToPILImage(), transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            def encode(imgs):
                import torch as _t
                x = _t.stack([tf(im) for im in imgs]).to(dev)
                with _t.no_grad():
                    return net(x).cpu().numpy()
            return encode, 512
    except Exception as e:  # pragma: no cover - depends on optional deps
        print(f"[video] frozen backbone '{name}' unavailable ({e}); "
              f"using engineered features only.", flush=True)
        return None, 0


# ----------------------------------------------------------------------------
# Per-window features
# ----------------------------------------------------------------------------
def windowed_video_features(subject: int, k: int, *, win_s: float = 5.0,
                            hop_s: float = 1.0, backbone: str | None = "resnet18",
                            gaze_crop: bool = False, kind: str = "main") -> np.ndarray | None:
    """Return (n_windows, ENGINEERED_DIM + raw_embed_dim) for one trial, or None.

    Embedding columns are raw encoder output (PCA/L2 happens later, across all
    windows of a subject, in ``build_subject``)."""
    import cv2
    vd = video_trial_dir(subject, k, kind=kind)
    if vd is None:
        return None
    vp = vd / "scenevideo.mp4"
    if not vp.exists():
        return None

    enc, raw_dim = (_load_backbone(backbone) if backbone else (None, 0))
    cap = cv2.VideoCapture(str(vp))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    win_f, hop_f = int(win_s * fps), int(hop_s * fps)
    enc_stride = max(1, int(fps / FRAME_HZ))

    frames_gray, frames_rgb_idx, rgb_frames = [], [], []
    i, ok = 0, True
    prev = None
    energies, flow_mag, flow_dirc = [], [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(frame, DS), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            energies.append(float(cv2.absdiff(g, prev).mean()))
            if i % enc_stride == 0:
                flow = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                flow_mag.append(float(mag.mean()))
                flow_dirc.append((float(np.cos(ang).mean()), float(np.sin(ang).mean())))
        if enc is not None and i % enc_stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if gaze_crop:
                rgb = _gaze_crop(rgb, subject, k, i / fps, kind)
            rgb_frames.append(rgb)
            frames_rgb_idx.append(i)
        prev = g
        i += 1
    cap.release()
    if not energies:
        return None

    n_frames = i
    energies = np.asarray(energies)
    embeds = enc(rgb_frames) if (enc is not None and rgb_frames) else None

    rows = []
    for s in range(0, n_frames - win_f + 1, hop_f):
        e, end = s, s + win_f
        win_en = energies[max(0, e - 1):end - 1]
        # flow samples that fall in-window
        fi0, fi1 = e // enc_stride, end // enc_stride
        fmag = flow_mag[fi0:fi1] or [0.0]
        fdir = flow_dirc[fi0:fi1] or [(0.0, 0.0)]
        cs = np.array([c for c, _ in fdir]); ss = np.array([s_ for _, s_ in fdir])
        feat = [
            float(win_en.mean()) if len(win_en) else 0.0,
            float(win_en.std()) if len(win_en) else 0.0,
            float(np.percentile(win_en, 95)) if len(win_en) else 0.0,
            float(np.mean(fmag)), float(np.std(fmag)),
            float(np.sqrt(cs.mean() ** 2 + ss.mean() ** 2)),
        ]
        if embeds is not None:
            m = (np.asarray(frames_rgb_idx) >= e) & (np.asarray(frames_rgb_idx) < end)
            emb = embeds[m].mean(0) if m.any() else np.zeros(raw_dim, "f4")
            feat = np.concatenate([feat, emb]).astype("f4")
        rows.append(np.asarray(feat, "f4"))
    return np.stack(rows) if rows else None


def _gaze_crop(rgb: np.ndarray, subject: int, k: int, t_s: float, kind: str,
               frac: float = 0.4) -> np.ndarray:
    """Crop a window around the projected gaze point at time t_s; centre crop on
    failure. Uses the scene-projected gaze2d from the Tobii video stream."""
    try:
        from aad_utils import load_raw_gaze
        rg = load_raw_gaze(subject, k)
        row = rg.iloc[(rg["t"].astype(float) - t_s).abs().argmin()]
        gx, gy = float(row.get("gaze2d_x", 0.5)), float(row.get("gaze2d_y", 0.5))
    except Exception:
        gx, gy = 0.5, 0.5
    H, W = rgb.shape[:2]
    hw, hh = int(W * frac / 2), int(H * frac / 2)
    cx, cy = int(np.clip(gx, 0, 1) * W), int(np.clip(gy, 0, 1) * H)
    x0, y0 = max(0, cx - hw), max(0, cy - hh)
    return rgb[y0:y0 + 2 * hh, x0:x0 + 2 * hw]


def build_subject(subject: int, *, out_dir: Path, win_s: float, hop_s: float,
                  backbone: str | None, gaze_crop: bool, video_dim: int) -> None:
    """Extract per-window features for every trial, PCA-reduce the frozen-encoder
    block across the subject, L2-norm, pad/trim to ``video_dim``, and cache."""
    tr_csv = load_trials_csv()
    per_trial, trial_ids = {}, []
    for k in range(1, 101):
        feats = windowed_video_features(subject, k, win_s=win_s, hop_s=hop_s,
                                        backbone=backbone, gaze_crop=gaze_crop)
        if feats is None:
            continue
        per_trial[k] = feats
        trial_ids.append(k)
        print(f"[S{subject}] trial {k}: {feats.shape}", flush=True)
    if not per_trial:
        print(f"[S{subject}] no video found"); return

    has_embed = next(iter(per_trial.values())).shape[1] > ENGINEERED_DIM
    if has_embed:
        from sklearn.decomposition import PCA
        all_emb = np.vstack([v[:, ENGINEERED_DIM:] for v in per_trial.values()])
        n_comp = min(EMBED_PCA_DIM, all_emb.shape[0], all_emb.shape[1])
        pca = PCA(n_components=n_comp).fit(all_emb)
        for k in per_trial:
            eng, emb = per_trial[k][:, :ENGINEERED_DIM], per_trial[k][:, ENGINEERED_DIM:]
            per_trial[k] = np.concatenate([eng, pca.transform(emb)], axis=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    packed = {}
    for k, v in per_trial.items():
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-6)
        if v.shape[1] < video_dim:
            v = np.pad(v, ((0, 0), (0, video_dim - v.shape[1])))
        packed[f"trial_{k}"] = v[:, :video_dim].astype("f4")
    np.savez_compressed(out_dir / f"s{subject}.npz", **packed)
    print(f"[S{subject}] wrote {out_dir / f's{subject}.npz'} "
          f"({len(packed)} trials, dim={video_dim})", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--backbone", choices=["resnet18", "clip", "none"], default="resnet18")
    ap.add_argument("--gaze-crop", action="store_true")
    ap.add_argument("--win-s", type=float, default=5.0)
    ap.add_argument("--hop-s", type=float, default=1.0)
    ap.add_argument("--video-dim", type=int, default=16)
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "video_embeddings")
    a = ap.parse_args()
    build_subject(a.subject, out_dir=a.out, win_s=a.win_s, hop_s=a.hop_s,
                  backbone=None if a.backbone == "none" else a.backbone,
                  gaze_crop=a.gaze_crop, video_dim=a.video_dim)


if __name__ == "__main__":
    main()
