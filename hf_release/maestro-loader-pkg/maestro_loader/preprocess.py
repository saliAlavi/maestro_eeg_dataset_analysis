"""Signal preprocessing for the loader.

EEG preprocessing mirrors the project's ``aad_utils.preprocess`` defaults so a
segment loaded with ``preprocess=True`` matches what the decoder trains on:
notch 60 Hz, band-pass 1-40 Hz, robust re-reference (linked mastoids, else
common-average), and bad-channel interpolation. Requires MNE (``pip install
maestro-loader[mne]``); without it a SciPy-only band-pass + average-reference
fallback is used.

Audio features: ``waveform`` (raw, aligned) or ``mel`` — a 28-band log-mel
"gammatone proxy" (no librosa dependency) matching the decoder's audio front end.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

ADC_CLIP = 0.08388
DEFAULT_PREPROCESS = {
    "l_freq": 1.0, "h_freq": 40.0, "notch": 60.0, "reference": "auto",
    "interpolate_bads": True,
}
MASTOIDS = ("M1", "M2")


# --------------------------------------------------------------------------- #
# EEG
# --------------------------------------------------------------------------- #
def _detect_bads(data: np.ndarray, ch_names: list[str]) -> list[str]:
    """Flat / saturated / variance-outlier channels (lightweight)."""
    stds = data.std(axis=1)
    sat = (np.abs(data) >= ADC_CLIP).mean(axis=1)
    bad = set()
    for i, ch in enumerate(ch_names):
        if stds[i] < 1e-9 or sat[i] >= 0.1:
            bad.add(ch)
    rem = [i for i, ch in enumerate(ch_names) if ch not in bad]
    if len(rem) >= 3:
        ac = np.diff(data[rem], axis=1)
        v = np.var(ac, axis=1)
        med = np.median(v)
        mad = np.median(np.abs(v - med)) or 1e-30
        for j, i in enumerate(rem):
            if abs((v[j] - med) / (1.4826 * mad)) > 6.0:
                bad.add(ch_names[i])
    return sorted(bad)


def preprocess_eeg(data: np.ndarray, ch_names: list[str], sfreq: float,
                   cfg: dict | None = None) -> np.ndarray:
    """Preprocess EEG. ``data`` is ``(C, T)``; returns ``(C, T)`` (float32)."""
    cfg = {**DEFAULT_PREPROCESS, **(cfg or {})}
    data = np.asarray(data, dtype=np.float64)
    try:
        return _preprocess_mne(data, ch_names, sfreq, cfg).astype(np.float32)
    except Exception:
        return _preprocess_scipy(data, sfreq, cfg).astype(np.float32)


def _preprocess_mne(data, ch_names, sfreq, cfg):
    import mne
    info = mne.create_info(list(ch_names), sfreq, ch_types="eeg")
    try:
        info.set_montage("standard_1020", match_case=False, on_missing="ignore")
    except Exception:
        pass
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    bads = _detect_bads(data, ch_names)
    raw.info["bads"] = list(bads)
    if cfg.get("notch"):
        raw.notch_filter(freqs=cfg["notch"], verbose="ERROR")
    raw.filter(l_freq=cfg.get("l_freq"), h_freq=cfg.get("h_freq"), verbose="ERROR")
    ref = cfg.get("reference", "auto")
    if ref == "auto":
        good_m = [m for m in MASTOIDS if m in ch_names and m not in bads]
        if good_m:
            raw.set_eeg_reference(ref_channels=good_m, verbose="ERROR")
        else:
            raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    elif ref == "average":
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    elif isinstance(ref, (list, tuple)):
        raw.set_eeg_reference(ref_channels=list(ref), verbose="ERROR")
    if cfg.get("interpolate_bads") and raw.info["bads"]:
        try:
            raw.interpolate_bads(reset_bads=True, verbose="ERROR")
        except Exception:
            pass
    return raw.get_data()


def _preprocess_scipy(data, sfreq, cfg):
    x = data
    nyq = sfreq / 2.0
    if cfg.get("notch"):
        b, a = signal.iirnotch(cfg["notch"] / nyq, Q=30)
        x = signal.filtfilt(b, a, x, axis=1)
    lo = (cfg.get("l_freq") or 0.0) / nyq
    hi = min(cfg.get("h_freq") or nyq, nyq - 1e-3) / nyq
    if lo > 0 and hi < 1:
        b, a = signal.butter(4, [lo, hi], btype="band")
    elif hi < 1:
        b, a = signal.butter(4, hi, btype="low")
    else:
        b, a = (np.array([1.0]), np.array([1.0]))
    x = signal.filtfilt(b, a, x, axis=1)
    x = x - x.mean(axis=0, keepdims=True)  # average reference
    return x


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #
def resample_to(x: np.ndarray, sr_in: float, sr_out: float, axis: int = -1) -> np.ndarray:
    """Polyphase resample along ``axis`` (LCM-reduced ratio)."""
    if abs(sr_in - sr_out) < 1e-6:
        return x.astype(np.float32, copy=False)
    from math import gcd
    up, down = int(round(sr_out)), int(round(sr_in))
    g = gcd(up, down) or 1
    return signal.resample_poly(x, up // g, down // g, axis=axis).astype(np.float32)


def interp_to_grid(t_src: np.ndarray, x: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Linear interpolation of ``x`` (T_src, ...) sampled at ``t_src`` onto ``t_grid``."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return np.interp(t_grid, t_src, x).astype(np.float32)
    out = np.empty((len(t_grid),) + x.shape[1:], dtype=np.float32)
    for j in range(x.shape[1]):
        out[:, j] = np.interp(t_grid, t_src, x[:, j])
    return out


# --------------------------------------------------------------------------- #
# Audio features
# --------------------------------------------------------------------------- #
def _mel_filterbank(n_mels: int, n_fft: int, sr: float,
                    fmin: float = 80.0, fmax: float = 8000.0) -> np.ndarray:
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    fmax = min(fmax, sr / 2.0)
    m = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    freqs = mel2hz(m)
    bins = np.floor((n_fft + 1) * freqs / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        if c == l:
            c = l + 1
        if r == c:
            r = c + 1
        for k in range(l, c):
            if 0 <= k < fb.shape[1]:
                fb[i - 1, k] = (k - l) / max(c - l, 1)
        for k in range(c, r):
            if 0 <= k < fb.shape[1]:
                fb[i - 1, k] = (r - k) / max(r - c, 1)
    return fb


def mel_envelope(wav: np.ndarray, sr: float, sr_out: float, n_mels: int = 28) -> np.ndarray:
    """28-band log-mel envelope resampled to ``sr_out``. Returns ``(n_mels, T_out)``."""
    wav = np.asarray(wav, dtype=np.float32).ravel()
    hop = max(1, int(round(sr / sr_out)))
    n_fft = 1 << int(np.ceil(np.log2(max(hop * 4, 256))))
    win = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + max(0, (len(wav) - n_fft) // hop)
    if n_frames <= 0:
        return np.zeros((n_mels, 1), dtype=np.float32)
    fb = _mel_filterbank(n_mels, n_fft, sr)
    out = np.empty((n_mels, n_frames), dtype=np.float32)
    for i in range(n_frames):
        seg = wav[i * hop: i * hop + n_fft]
        if len(seg) < n_fft:
            seg = np.pad(seg, (0, n_fft - len(seg)))
        spec = np.abs(np.fft.rfft(seg * win)) ** 2
        out[:, i] = fb @ spec
    return np.log1p(out).astype(np.float32)
