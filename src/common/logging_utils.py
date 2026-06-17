"""Project-wide logging set-up (stdlib ``logging`` + optional file sink)."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=_FMT,
        datefmt=_DATEFMT,
        handlers=handlers,
        force=True,
    )
    # Quieten chatty third-party loggers.
    for noisy in ("mne", "matplotlib", "h5py", "numba", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
