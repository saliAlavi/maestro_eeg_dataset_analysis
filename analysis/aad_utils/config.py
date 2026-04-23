"""Global configuration: paths, electrode montage, experiment constants."""
from __future__ import annotations
from pathlib import Path

# ---- Paths ----
DATA_ROOT = Path("/fs/ess/PAS2301/Data/AAD Data Collection")
EXPERIMENT_DIR = DATA_ROOT / "experiment_data"
VIDEO_DIR = DATA_ROOT / "Video Recordings"
AUDIO_DIR = DATA_ROOT / "audio_stimuli_data"
TRIALS_CSV = AUDIO_DIR / "trials.csv"
PAIRS_DIR = AUDIO_DIR / "pairs"

PROJECT_ROOT = Path("/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu")
ANALYSIS_ROOT = PROJECT_ROOT / "analysis"
FIGURES_DIR = ANALYSIS_ROOT / "figures"
CACHE_DIR = ANALYSIS_ROOT / "cache"
RESULTS_DIR = ANALYSIS_ROOT / "results"
for _d in (FIGURES_DIR, CACHE_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- Experiment ----
N_SUBJECTS = 16
N_TRAIN_TRIALS = 5
N_MAIN_TRIALS = 100
EEG_SFREQ = 500.0  # Hz

# ANT Neuro 32-channel montage in pickle channel order (indices 0..31).
# Indices 32 and 33 in the raw pickle are a zero channel and a sample counter.
EEG_CHANNELS = [
    "Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6", "M1", "T7", "C3", "Cz",
    "C4", "T8", "M2", "CP5", "CP1", "CP2", "CP6", "P7",
    "P3", "Pz", "P4", "P8", "POz", "O1", "Oz", "O2",
]
assert len(EEG_CHANNELS) == 32
EEG_MASTOIDS = ("M1", "M2")

# Speaker spatial layout (azimuth in degrees, listener-centered; positive = right).
# Device-1 is the entire LEFT hemisphere pair (both at negative azimuth);
# Device-2 is the entire RIGHT hemisphere pair (both at positive azimuth);
# Device-3 is the back pair carrying noise.
#
# Each device plays a STEREO speech file: the 'Left' channel drives the
# farther-lateral speaker of that device (±67.5° for the speech devices,
# ±135° for noise) and the 'Right' channel drives the more-central
# speaker (±22.5° for the speech devices).
SPEAKER_AZIMUTHS = {
    ("Device-1", "Left"):  -67.5,  # far-left
    ("Device-1", "Right"): -22.5,  # near-left
    ("Device-2", "Left"):   22.5,  # near-right
    ("Device-2", "Right"):  67.5,  # far-right
    ("Device-3", "Left"): -135.0,  # back-left (noise)
    ("Device-3", "Right"): 135.0,  # back-right (noise)
}
# The "Attended Speaker" column in trials.csv is 1..4 mapping to:
#   1 -> Device-1 Left (-67.5°), 2 -> Device-1 Right (-22.5°),
#   3 -> Device-2 Left (+22.5°), 4 -> Device-2 Right (+67.5°).
ATTENDED_SPEAKER_MAP = {
    1: ("Device-1", "Left",  -67.5),  # LEFT hemisphere, far
    2: ("Device-1", "Right", -22.5),  # LEFT hemisphere, near
    3: ("Device-2", "Left",   22.5),  # RIGHT hemisphere, near
    4: ("Device-2", "Right",  67.5),  # RIGHT hemisphere, far
}
# Hemisphere membership (useful for left-vs-right binary AAD).
ATTENDED_HEMISPHERE = {1: "L", 2: "L", 3: "R", 4: "R"}
SPEAKER_DISTANCE_M = 4 * 0.3048  # 4 feet -> meters
