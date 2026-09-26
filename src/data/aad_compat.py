"""Compatibility shim around the repo's ``aad_utils`` package.

Two jobs:
  1. Put ``analysis/`` on ``sys.path`` so ``import aad_utils`` works from ``src/``.
  2. Repair the stale path constants. The installed config points
     ``TRIALS_CSV`` / ``PAIRS_DIR`` at ``DATA_ROOT/audio_stimuli_data`` but the
     data now lives under ``DATA_ROOT/experiment_data``. We patch both the
     ``config`` and ``io`` module namespaces (``io`` imports the names at load
     time) so every loader resolves correctly -- without editing their source.

Everything downstream imports loaders from here, never from ``aad_utils``
directly, so the patch is guaranteed to be applied exactly once.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ANALYSIS = Path(__file__).resolve().parents[2] / "analysis"
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

# The package lives in the repo's analysis/ directory, which a code-only
# release does not ship. It is needed only to rebuild caches from the raw
# recordings, so its absence must not break importing the rest of src/.
try:
    import aad_utils as A  # noqa: E402
    import aad_utils.config as _C  # noqa: E402
    import aad_utils.io as _IO  # noqa: E402
    _IMPORT_ERR = None
except ImportError as _e:
    A = _C = _IO = None
    _IMPORT_ERR = _e


def _patch_paths() -> None:
    correct_csv = _C.EXPERIMENT_DIR / "trials.csv"
    correct_pairs = _C.EXPERIMENT_DIR / "pairs"
    for mod in (_C, _IO):
        if hasattr(mod, "TRIALS_CSV"):
            mod.TRIALS_CSV = correct_csv
        if hasattr(mod, "PAIRS_DIR"):
            mod.PAIRS_DIR = correct_pairs
    # Keep the package-level export coherent too.
    A.TRIALS_CSV = correct_csv
    A.PAIRS_DIR = correct_pairs


if A is not None:
    _patch_paths()

    # Re-export the loaders we use, post-patch.
    DATA_ROOT = _C.DATA_ROOT
    EXPERIMENT_DIR = _C.EXPERIMENT_DIR
    EEG_CHANNELS = _C.EEG_CHANNELS
    EEG_SFREQ = _C.EEG_SFREQ
    ATTENDED_HEMISPHERE = _C.ATTENDED_HEMISPHERE
    PAIRS_DIR = _C.EXPERIMENT_DIR / "pairs"

    load_trials_csv = A.load_trials_csv
    trial_name = A.trial_name
    load_eeg_trial = A.load_eeg_trial
    load_eeg_time = A.load_eeg_time
    load_gaze_trial_2d = A.load_gaze_trial_2d
    load_audio_timestamps = A.load_audio_timestamps
    load_raw_gaze = A.load_raw_gaze
    load_raw_imu = A.load_raw_imu
    align_modalities_to_trial = A.align_modalities_to_trial
    eeg_raw_to_mne = A.eeg_raw_to_mne
    preprocess_eeg = A.preprocess_eeg
    gammatone_envelope = A.gammatone_envelope
    bootstrap_ci = A.bootstrap_ci


def __getattr__(name):
    # Only consulted for names the block above did not define, i.e. when
    # aad_utils was unavailable (PEP 562).
    raise ModuleNotFoundError(
        f"src.data.aad_compat.{name} needs the 'aad_utils' package from the "
        "repo's analysis/ directory, which is only required to rebuild caches "
        "from the raw recordings") from _IMPORT_ERR


def sanity_check() -> dict:
    """Cheap check that the patched paths actually resolve (used by --selftest)."""
    csv = load_trials_csv()
    return {
        "trials_csv_rows": int(csv.shape[0]),
        "pairs_dir_ok": PAIRS_DIR.exists(),
        "data_root_ok": DATA_ROOT.exists(),
    }
