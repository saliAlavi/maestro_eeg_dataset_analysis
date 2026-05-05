"""Normalization helpers used by the loader."""
from __future__ import annotations

import numpy as np


def _stats(x: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (centre, scale) for ``x`` along the time axis (axis 0)."""
    if mode == "zscore":
        c = x.mean(axis=0, keepdims=True)
        s = x.std(axis=0, keepdims=True)
    elif mode == "robust":
        c = np.median(x, axis=0, keepdims=True)
        q75 = np.quantile(x, 0.75, axis=0, keepdims=True)
        q25 = np.quantile(x, 0.25, axis=0, keepdims=True)
        s = (q75 - q25)
    elif mode == "minmax":
        lo = x.min(axis=0, keepdims=True)
        hi = x.max(axis=0, keepdims=True)
        c = lo
        s = (hi - lo)
    else:
        raise ValueError(f"unknown normalize mode: {mode!r}")
    s = np.where(np.abs(s) < 1e-8, 1.0, s)
    return c, s


def apply_norm(x: np.ndarray, c: np.ndarray, s: np.ndarray, mode: str) -> np.ndarray:
    if mode == "minmax":
        # rescale to [0, 1]
        return ((x - c) / s).astype(np.float32, copy=False)
    return ((x - c) / s).astype(np.float32, copy=False)


def normalize_array(x: np.ndarray, mode: str | None) -> np.ndarray:
    if mode is None:
        return x
    c, s = _stats(x.astype(np.float64, copy=False), mode)
    return apply_norm(x, c, s, mode)
