"""Cross-modality time alignment.

The EEG stream uses an internal monotonic clock ``eeg_ts`` (seconds). The
``eeg_time_data.p`` record gives, for the first sample, ``first_sample_time``
(unix seconds) and the corresponding ``eeg_ts``. So:

    unix(t_eeg) = first_sample_time + (t_eeg - eeg_ts_at_first_sample)

Experiment-data 2D gaze uses unix seconds directly (``gaze_ts``).

Tobii raw gaze/IMU (Video Recordings) use a recording-relative clock starting
at ~0. Their wall-clock anchor comes from ``audio_timestamps.json``: each entry
gives ``start_time`` (unix) for a device. However the Tobii bundle itself does
not expose its recording-start wall-clock directly here — we treat the IMU/gaze
``timestamp == 0`` as aligned to the Tobii recording start. Because
``audio_timestamps.json`` contains the wall-clock ``start_time`` / ``end_time``
for each playback device, we can locate the trial's audio window in unix time
and subsequently align the recording-relative stream to it if the user provides
the Tobii recording-start time. When unavailable we align by matching the
(recording-relative) trial duration to the audio-playback window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def eeg_to_unix_time(eeg_ts: np.ndarray, eeg_time_meta: dict) -> np.ndarray:
    """Convert internal eeg_ts to wall-clock unix seconds."""
    t0_internal = float(eeg_time_meta["eeg_ts"])
    t0_unix = float(eeg_time_meta["first_sample_time"])
    return t0_unix + (np.asarray(eeg_ts, dtype=float) - t0_internal)


def gaze_to_unix_time(gaze_ts: np.ndarray, gaze_time_meta: dict | None = None) -> np.ndarray:
    """Experiment-data gaze_ts is already unix seconds — return as-is."""
    return np.asarray(gaze_ts, dtype=float)


@dataclass
class TrialWindow:
    """Wall-clock window for a trial in unix seconds."""
    t0: float  # earliest audio playback_start_time
    t1: float  # latest audio end_time
    playback: list[dict[str, Any]]

    @property
    def duration(self) -> float:
        return self.t1 - self.t0


def trial_window_from_audio(audio_timestamps: list[dict]) -> TrialWindow:
    t0 = min(a["playback_start_time"] for a in audio_timestamps)
    t1 = max(a["end_time"] for a in audio_timestamps)
    return TrialWindow(t0=t0, t1=t1, playback=audio_timestamps)


def _slice_by_unix(ts_unix: np.ndarray, window: TrialWindow) -> np.ndarray:
    return np.where((ts_unix >= window.t0) & (ts_unix <= window.t1))[0]


def align_modalities_to_trial(
    *,
    eeg: np.ndarray,
    eeg_ts: np.ndarray,
    eeg_time_meta: dict,
    gaze2d: pd.DataFrame,
    audio_timestamps: list[dict],
    raw_gaze: pd.DataFrame | None = None,
    raw_imu: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Clip every stream to the shared audio-playback window.

    Returns
    -------
    dict with keys ``window``, ``eeg``, ``eeg_unix``, ``gaze2d``, and optional
    ``raw_gaze``, ``raw_imu`` (each trimmed/annotated with unix time).
    """
    window = trial_window_from_audio(audio_timestamps)
    eeg_unix = eeg_to_unix_time(eeg_ts, eeg_time_meta)
    eeg_mask = (eeg_unix >= window.t0) & (eeg_unix <= window.t1)
    eeg_clip = eeg[eeg_mask]
    eeg_unix_clip = eeg_unix[eeg_mask]

    gaze2d_unix = gaze_to_unix_time(gaze2d["gaze_ts"].values)
    gmask = (gaze2d_unix >= window.t0) & (gaze2d_unix <= window.t1)
    gaze2d_clip = gaze2d.iloc[gmask].copy()
    gaze2d_clip["t_unix"] = gaze2d_unix[gmask]

    out: dict[str, Any] = dict(
        window=window,
        eeg=eeg_clip,
        eeg_unix=eeg_unix_clip,
        gaze2d=gaze2d_clip,
    )

    if raw_gaze is not None and len(raw_gaze) > 0:
        # Tobii clock is recording-relative; shift so its span matches window.
        t_rel = raw_gaze["t"].values
        rel_dur = float(np.nanmax(t_rel) - np.nanmin(t_rel)) if len(t_rel) else 0.0
        if rel_dur > 0:
            # Naive anchor: assume Tobii recording start ≈ first audio start.
            t_unix = window.t0 + (t_rel - np.nanmin(t_rel))
            rg = raw_gaze.copy()
            rg["t_unix"] = t_unix
            rg_mask = (rg["t_unix"] >= window.t0) & (rg["t_unix"] <= window.t1)
            out["raw_gaze"] = rg[rg_mask].reset_index(drop=True)
        else:
            out["raw_gaze"] = raw_gaze

    if raw_imu is not None and len(raw_imu) > 0:
        t_rel = raw_imu["t"].values
        if np.any(np.isfinite(t_rel)):
            t_unix = window.t0 + (t_rel - np.nanmin(t_rel))
            ri = raw_imu.copy()
            ri["t_unix"] = t_unix
            ri_mask = (ri["t_unix"] >= window.t0) & (ri["t_unix"] <= window.t1)
            out["raw_imu"] = ri[ri_mask].reset_index(drop=True)
        else:
            out["raw_imu"] = raw_imu

    return out
