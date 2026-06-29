"""Lazy egocentric scene-video access, aligned to the same unix clock.

Video frames live on the Tobii recording clock: ``frame_unix(k) =
recording_start_unix + k / fps``. Given a segment expressed in seconds-from-
anchor, :func:`frame_index_range` returns the matching frame indices, and
:func:`read_frames` decodes them on demand (PyAV preferred, OpenCV fallback).

Decoding video is expensive, so by default a dataset item carries only the
video path + frame range; pass ``video_frames=True`` to ``load_aad`` to decode.
"""
from __future__ import annotations

import numpy as np


def frame_index_range(recording_start_unix: float, fps: float,
                      seg_start_unix: float, seg_end_unix: float,
                      n_frames_total: int | None = None) -> tuple[int, int]:
    if not fps:
        return (0, 0)
    f0 = int(np.floor((seg_start_unix - recording_start_unix) * fps))
    f1 = int(np.ceil((seg_end_unix - recording_start_unix) * fps))
    f0 = max(0, f0)
    if n_frames_total:
        f1 = min(f1, n_frames_total)
    return (f0, max(f0, f1))


def read_frames(path: str, start: int, stop: int, step: int = 1,
                max_frames: int | None = None) -> np.ndarray:
    """Decode frames [start, stop) as a ``(N, H, W, 3)`` uint8 RGB array."""
    if stop <= start:
        return np.empty((0,), dtype=np.uint8)
    try:
        return _read_pyav(path, start, stop, step, max_frames)
    except Exception:
        return _read_cv2(path, start, stop, step, max_frames)


def _read_pyav(path, start, stop, step, max_frames):
    import av
    out = []
    with av.open(path) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for i, frame in enumerate(c.decode(s)):
            if i < start:
                continue
            if i >= stop:
                break
            if (i - start) % step == 0:
                out.append(frame.to_ndarray(format="rgb24"))
                if max_frames and len(out) >= max_frames:
                    break
    return np.stack(out) if out else np.empty((0,), dtype=np.uint8)


def _read_cv2(path, start, stop, step, max_frames):
    import cv2
    cap = cv2.VideoCapture(path)
    out = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    i = start
    while i < stop:
        ok, fr = cap.read()
        if not ok:
            break
        if (i - start) % step == 0:
            out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            if max_frames and len(out) >= max_frames:
                break
        i += 1
    cap.release()
    return np.stack(out) if out else np.empty((0,), dtype=np.uint8)
