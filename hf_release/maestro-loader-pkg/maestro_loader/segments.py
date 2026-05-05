"""On-the-fly segmentation of multi-rate trial signals."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Segment:
    idx: int
    t_start_sec: float
    t_end_sec: float


def make_segments(
    duration_sec: float,
    segment_length: float | None,
    overlap: float = 0.0,
    drop_last: bool = True,
) -> list[Segment]:
    """Compute (start, end) offsets in seconds within a trial.

    ``segment_length=None`` returns one segment spanning the whole trial.
    """
    if segment_length is None or segment_length <= 0:
        return [Segment(idx=0, t_start_sec=0.0, t_end_sec=duration_sec)]
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1); got {overlap}")
    step = segment_length * (1.0 - overlap)
    segs: list[Segment] = []
    t = 0.0
    i = 0
    eps = 1e-9
    while t + segment_length <= duration_sec + eps:
        segs.append(Segment(idx=i, t_start_sec=t, t_end_sec=t + segment_length))
        i += 1
        t += step
    if not drop_last and (not segs or segs[-1].t_end_sec < duration_sec - eps):
        segs.append(Segment(idx=i, t_start_sec=max(0.0, duration_sec - segment_length),
                            t_end_sec=duration_sec))
    return segs


def slice_signal(arr, sfreq: float, t_start_sec: float, t_end_sec: float):
    """Slice a (T, ...) array by seconds. Pure NumPy/pandas-compatible."""
    i0 = int(round(t_start_sec * sfreq))
    i1 = int(round(t_end_sec * sfreq))
    return arr[i0:i1]
