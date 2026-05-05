"""Notebook 00 — Setup and quickstart."""

CELLS = [
    ("md", """\
# Notebook 00 — Setup and Quickstart

**Series:** maestro-eeg-dataset analysis suite
**Authors:** Ali Alavi, N. Hasan, D. Williamson
**License:** CC-BY-4.0 (data) · Apache-2.0 (loader)

This notebook installs the [`maestro-loader`](https://pypi.org/project/maestro-loader/) Python package, downloads a small subset of the [`aspire-osu/maestro-eeg-dataset`](https://huggingface.co/datasets/aspire-osu/maestro-eeg-dataset) from the Hugging Face Hub, and runs a one-segment smoke test that exercises every modality. After you finish this notebook, the rest of the suite (notebooks 01–08) will run without further setup.

> **Estimated time:** ~3 minutes (mostly the first download).
"""),

    ("md", """\
## 1. Install dependencies

We pin `maestro-loader >= 0.1.2`. The optional extras pull in `torch` (for batched training in the AAD notebooks) and `mne` (for spherical-spline bad-channel interpolation).
"""),

    ("code", """\
%%capture
%pip install --quiet --upgrade "maestro-loader>=0.1.2" "matplotlib>=3.7" "seaborn>=0.13" "scipy>=1.10" "scikit-learn>=1.3" "tqdm>=4.66"
# Optional but recommended:
%pip install --quiet "mne>=1.6"
"""),

    ("code", """\
import maestro_loader, sys
print(f"maestro-loader {maestro_loader.__version__}")
print(f"Python {sys.version.split()[0]}")
"""),

    ("md", """\
## 2. Project-wide style and reproducibility

Every notebook in the suite imports this same setup block to keep figures consistent and seeds fixed.
"""),

    ("code", """\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Reproducibility ---
RNG_SEED = 1337
np.random.seed(RNG_SEED)

# --- Publication-quality plotting style ---
sns.set_context("paper", font_scale=1.05)
sns.set_style("ticks")
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "semibold",
    "axes.labelweight": "regular",
    "font.family": "DejaVu Sans",
})

# --- Dataset config (edit once; used everywhere) ---
REPO_ID_FULL   = "aspire-osu/maestro-eeg-dataset"
REPO_ID_SAMPLE = "aspire-osu/maestro-eeg-dataset-sample"
USE_SAMPLE     = True   # quick iteration; flip to False to use the full 41.7 GB release
REPO_ID = REPO_ID_SAMPLE if USE_SAMPLE else REPO_ID_FULL

print(f"Reading from: {REPO_ID}")
"""),

    ("md", """\
## 3. One-segment smoke test

We pull a single 2-second segment from Subject 1, Trial 1, with all five modalities. If this cell completes, every notebook downstream will work.
"""),

    ("code", """\
from maestro_loader import load_aad

ds = load_aad(
    subjects=[1],
    trials=[1],
    modalities=["eeg", "gaze", "imu", "audio"],
    segment_length=2.0,
    overlap=0.0,
    normalize="zscore",
    bad_channels="zero",
    repo_id=REPO_ID,
)
print(f"  segments: {len(ds)}")

sample = ds[0]
print(f"  subject={sample['subject_id']}  trial={sample['trial_id']}  "
      f"attended_speaker={sample['attended_speaker']}  azimuth={sample['azimuth_attended_deg']:.1f}°")
print(f"  EEG   : {sample['eeg'].shape}      @ {sample['eeg_sfreq']:.0f} Hz   ({len(sample['eeg_channels'])} channels)")
print(f"  gaze  : {sample['gaze'].shape}     @ {sample['gaze_sfreq']:.1f} Hz")
print(f"  IMU   : {sample['imu'].shape}      @ {sample['imu_sfreq']:.1f} Hz")
print(f"  audio : {len(sample['audio'])} mono speakers, each {sample['audio'][1].shape} @ {sample['audio_sfreq']:.0f} Hz")
"""),

    ("md", """\
## 4. Visual sanity check

A 2-second window of one EEG channel + the attended speaker's audio waveform — both should look continuous and non-clipping.
"""),

    ("code", """\
attended_spk = sample["attended_speaker"]
fig, axes = plt.subplots(2, 1, figsize=(8, 3.6), sharex=False)

t_eeg = np.arange(sample["eeg"].shape[0]) / sample["eeg_sfreq"]
axes[0].plot(t_eeg, sample["eeg"][:, sample["eeg_channels"].index("Cz")], lw=0.7, color="#2c7fb8")
axes[0].set(ylabel="Cz (z-scored)", title=f"EEG · {sample['subject_id']} · {sample['trial_id']} · 2-s window")

t_aud = np.arange(sample["audio"][attended_spk].shape[0]) / sample["audio_sfreq"]
axes[1].plot(t_aud, sample["audio"][attended_spk], lw=0.4, color="#252525")
axes[1].set(xlabel="Time (s)", ylabel="amplitude",
            title=f"Audio · attended speaker {attended_spk} (azimuth {sample['azimuth_attended_deg']:.1f}°)")

plt.tight_layout()
plt.show()
"""),

    ("md", """\
## 5. Where to next

| Notebook | Topic |
|---|---|
| `01_dataset_overview` | Cohort demographics, modality coverage, trial structure |
| `02_eeg_signal_quality` | Bad-channel detection, PSDs, drift, line noise |
| `03_gaze_dynamics` | Trajectories, fixations, pupil, blink rate |
| `04_audio_stimulus_features` | Per-speaker envelopes, mel spectrograms |
| `05_eeg_stimulus_decoding` | Backward (mTRF) speech-envelope reconstruction |
| `06_attention_decoding` | Binary AAD classification (4 attendable speakers) |
| `07_multimodal_fusion` | EEG + gaze + IMU late fusion for AAD |
| `08_benchmark_protocol` | Formal LOSO + within-subject leaderboard |

To run a notebook against the **full 41.7 GB dataset** instead of the 422 MB sample, set `USE_SAMPLE = False` in cell 2 of any notebook.
"""),
]
