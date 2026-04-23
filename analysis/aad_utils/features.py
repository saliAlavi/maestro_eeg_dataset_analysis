"""Feature extractors: audio envelopes, saccade detection, pupil baselining."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import hilbert, butter, filtfilt, resample_poly


def audio_envelope(
    audio: np.ndarray,
    sr_in: int,
    *,
    sr_out: float = 64.0,
    smooth_hz: float = 8.0,
    power: float = 0.3,
) -> np.ndarray:
    """Hilbert amplitude envelope, low-pass smoothed, compressed, and down-sampled."""
    env = np.abs(hilbert(audio.astype(np.float64)))
    if smooth_hz is not None:
        b, a = butter(4, smooth_hz / (sr_in / 2), btype="low")
        env = filtfilt(b, a, env)
    env = np.maximum(env, 0)
    if power is not None and power != 1.0:
        env = np.power(env + 1e-12, power)
    # Resample.
    from math import gcd
    up = int(round(sr_out * 1000))
    dn = int(round(sr_in * 1000))
    g = gcd(up, dn)
    env = resample_poly(env, up // g, dn // g)
    return env.astype(np.float32)


def gammatone_envelope(
    audio: np.ndarray,
    sr_in: int,
    *,
    n_bands: int = 28,
    low_hz: float = 80.0,
    high_hz: float = 8000.0,
    sr_out: float = 64.0,
) -> np.ndarray:
    """Approximate gammatone-filterbank envelopes via log-mel as a robust surrogate.

    (A proper gammatone filterbank is not in scipy/librosa by default; mel bands
    are a well-known proxy widely used in AAD literature.)
    """
    import librosa
    hop = max(1, int(round(sr_in / sr_out)))
    S = librosa.feature.melspectrogram(
        y=audio.astype(np.float32),
        sr=sr_in,
        n_mels=n_bands,
        fmin=low_hz,
        fmax=min(high_hz, sr_in / 2 - 1),
        hop_length=hop,
        power=1.0,
    )
    return np.log1p(S).T.astype(np.float32)  # (time, bands)


def mel_spectrogram(audio: np.ndarray, sr_in: int, *, n_mels: int = 80, hop_ms: float = 10.0) -> np.ndarray:
    import librosa
    hop = max(1, int(round(sr_in * hop_ms / 1000)))
    S = librosa.feature.melspectrogram(y=audio.astype(np.float32), sr=sr_in, n_mels=n_mels, hop_length=hop)
    return librosa.power_to_db(S).T.astype(np.float32)


# --------------------------------------------------------------------------- #
# Gaze events
# --------------------------------------------------------------------------- #
@dataclass
class SaccadeEvents:
    onsets: np.ndarray
    offsets: np.ndarray
    amplitudes: np.ndarray
    velocities: np.ndarray


def detect_saccades_ivt(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    velocity_threshold_deg_s: float = 30.0,
    screen_deg: tuple[float, float] = (82.0, 52.0),
) -> SaccadeEvents:
    """I-VT saccade detection on normalized 2D gaze.

    Assumes x,y ∈ [0,1] (Tobii scene-relative) and converts to degrees using
    a nominal FOV (82×52° for Tobii Glasses 3 scene camera).
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float) * screen_deg[0]
    y = np.asarray(y, dtype=float) * screen_deg[1]
    dt = np.diff(t)
    dt[dt <= 0] = np.nan
    vx = np.diff(x) / dt
    vy = np.diff(y) / dt
    v = np.sqrt(vx ** 2 + vy ** 2)
    is_sacc = v > velocity_threshold_deg_s
    # Segment runs of True.
    idx = np.where(np.diff(np.concatenate(([0], is_sacc.astype(int), [0]))) != 0)[0]
    onsets = idx[0::2]
    offsets = idx[1::2]
    amps = np.array([
        np.hypot(x[off] - x[on], y[off] - y[on]) if off > on else 0.0
        for on, off in zip(onsets, offsets)
    ])
    peakv = np.array([
        np.nanmax(v[on:off]) if off > on else np.nan
        for on, off in zip(onsets, offsets)
    ])
    return SaccadeEvents(
        onsets=t[onsets] if len(onsets) else np.array([]),
        offsets=t[offsets] if len(offsets) else np.array([]),
        amplitudes=amps,
        velocities=peakv,
    )


def pupil_baseline_correct(
    pupil: np.ndarray,
    t: np.ndarray,
    baseline_window: tuple[float, float] = (0.0, 0.5),
) -> np.ndarray:
    """Subtract mean pupil in a baseline window (seconds, relative to t[0])."""
    pupil = np.asarray(pupil, dtype=float)
    t = np.asarray(t, dtype=float)
    t0 = t[0] if len(t) else 0.0
    mask = (t - t0 >= baseline_window[0]) & (t - t0 < baseline_window[1])
    if mask.sum() < 3:
        base = np.nanmean(pupil)
    else:
        base = np.nanmean(pupil[mask])
    return pupil - base
