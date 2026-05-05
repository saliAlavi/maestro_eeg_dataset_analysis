"""Bad EEG channel handling — drop / zero / interpolate."""
from __future__ import annotations

import numpy as np


def apply_bad_channels(
    eeg: np.ndarray,
    channels: list[str],
    bad: list[str],
    mode: str = "raw",
) -> tuple[np.ndarray, list[str]]:
    """Apply a bad-channel policy.

    Parameters
    ----------
    eeg
        ``(T, C)`` array.
    channels
        Channel names in the same order as columns of ``eeg``.
    bad
        Channel names to flag bad. Names not in ``channels`` are ignored.
    mode
        - ``"raw"`` — pass through unchanged (output ``channels`` unchanged).
        - ``"drop"`` — remove bad columns (output channels shrinks).
        - ``"zero"`` — keep shape, zero out bad columns.
        - ``"interpolate"`` — MNE spherical-spline interpolation on the
          ANT Neuro 10-20 layout (requires ``mne`` installed).

    Returns
    -------
    (eeg_out, channels_out)
    """
    bad_set = [b for b in bad if b in channels]
    if mode == "raw" or not bad_set:
        return eeg, list(channels)
    bad_idx = [channels.index(b) for b in bad_set]
    if mode == "drop":
        keep = [i for i in range(len(channels)) if i not in bad_idx]
        return eeg[:, keep], [channels[i] for i in keep]
    if mode == "zero":
        out = eeg.copy()
        out[:, bad_idx] = 0.0
        return out, list(channels)
    if mode == "interpolate":
        try:
            import mne
        except ImportError as e:
            raise ImportError(
                "bad_channels='interpolate' requires MNE. "
                "Install with: pip install maestro-loader[mne]"
            ) from e
        info = mne.create_info(ch_names=list(channels), sfreq=500.0, ch_types="eeg")
        info.set_montage("standard_1020", on_missing="ignore")
        raw = mne.io.RawArray(eeg.T.astype(np.float64), info, verbose="ERROR")
        raw.info["bads"] = list(bad_set)
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")
        return raw.get_data().T.astype(eeg.dtype), list(channels)
    raise ValueError(f"unknown bad_channels mode: {mode!r}")
