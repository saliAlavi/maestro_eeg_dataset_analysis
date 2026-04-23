"""Publication-style matplotlib defaults and helpers."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# Colorblind-safe palette (Tol Muted-ish).
COLORS = dict(
    eeg="#332288",
    gaze="#117733",
    audio="#CC6677",
    video="#DDCC77",
    imu="#88CCEE",
    pupil="#AA4499",
    attended="#882255",
    unattended="#6699CC",
    chance="#999999",
)


def set_pub_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 1.25,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def bootstrap_ci(
    x: np.ndarray,
    *,
    n_boot: int = 5000,
    alpha: float = 0.05,
    statistic=np.mean,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = statistic(x[idx], axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(statistic(x)), float(lo), float(hi)


def save_fig(fig, name: str, dir_: str | Path, *, formats=("pdf", "png")) -> list[Path]:
    """Save a figure to both vector and raster formats with consistent style."""
    dir_ = Path(dir_)
    dir_.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in formats:
        p = dir_ / f"{name}.{ext}"
        fig.savefig(p)
        out.append(p)
    return out
