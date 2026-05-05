"""Notebook 08 — Benchmark protocol and reference numbers."""

CELLS = [
    ("md", """\
# Notebook 08 — Benchmark protocol and reference numbers

This notebook defines the **official evaluation protocol** for the maestro-eeg-dataset and reports reference baselines that future submissions can compare against. We standardise:

| Aspect | Choice |
|---|---|
| **Task** | Binary AAD: which of the *two* attendable speakers in the same device is being attended? |
| **Window length** | 2.0 s (consistent with most AAD literature) |
| **Window overlap** | 50% |
| **EEG bandpass** | 1–32 Hz, then resample to 64 Hz |
| **Bad channels** | `interpolate` (with MNE) for cross-subject; `zero` for within-subject |
| **Within-subject CV** | leave-one-trial-out across the 100 main trials |
| **Cross-subject CV** | leave-one-subject-out (LOSO) — 16 folds |
| **Primary metric** | window-level accuracy |
| **Secondary metrics** | trial-level accuracy (majority vote within trial), AUROC of the score margin |

Reference baselines reported below: backward TRF (notebook 05/06) and EEG+gaze+IMU fusion (notebook 07).
"""),

    ("code", """\
%%capture
%pip install --quiet "maestro-loader>=0.1.2" "scikit-learn>=1.3" "scipy>=1.10" "matplotlib>=3.7" "seaborn>=0.13"
"""),

    ("code", """\
import json
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from maestro_loader import load_aad
from huggingface_hub import hf_hub_download

sns.set_context("paper", font_scale=1.0); sns.set_style("ticks")
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})

REPO_ID_FULL   = "aspire-osu/maestro-eeg-dataset"
REPO_ID_SAMPLE = "aspire-osu/maestro-eeg-dataset-sample"
USE_SAMPLE     = True
REPO_ID = REPO_ID_SAMPLE if USE_SAMPLE else REPO_ID_FULL
"""),

    ("md", """\
## 1. Splits — pre-defined LOSO folds via the loader

`load_aad(splits='loso', fold=K)` returns the test data for held-out subject `K + 1` (1-indexed → fold 0 = S01, fold 15 = S16). Use it on the full dataset as:

```python
for fold in range(16):
    train_ds = load_aad(subjects=[s for s in range(1, 17) if s != fold + 1], ...)
    test_ds  = load_aad(splits='loso', fold=fold, ...)
```

The sample dataset only has S01, S07, S14, so we just demonstrate the API here.
"""),

    ("code", """\
print("Available LOSO folds for this repo:")
subj_csv = pd.read_csv(hf_hub_download(REPO_ID, "metadata/subjects.csv", repo_type="dataset"))
for sid in subj_csv["subject_id"]:
    fold = int(sid.lstrip("S")) - 1
    print(f"  fold = {fold:2d}  → held-out = {sid}")
"""),

    ("md", """\
## 2. Load all metadata and the per-trial true labels
"""),

    ("code", """\
trials_meta   = pd.read_csv(hf_hub_download(REPO_ID, "metadata/trials.csv", repo_type="dataset"))
per_subj_meta = pd.read_csv(hf_hub_download(REPO_ID, "metadata/trials_per_subject.csv", repo_type="dataset"))
audio_layout  = json.loads(open(hf_hub_download(REPO_ID, "metadata/audio_layout.json", repo_type="dataset")).read())

print(f"trials      : {len(trials_meta)}")
print(f"per_subject : {len(per_subj_meta)}")
print(f"speaker layout: ")
for spk in audio_layout["speakers"]:
    flag = "✓ attendable" if spk["attendable"] else "✗ distractor"
    print(f"  speaker {spk['speaker']}  ({spk['channel']} of dev {spk['device']})  azimuth = {spk['azimuth_deg']:+.1f}°  {flag}")
"""),

    ("md", """\
## 3. Reference baselines

Numbers below are **representative** values from the within-subject and LOSO experiments in notebooks 06 and 07 (you should re-run them to populate this table on the full dataset). The protocol is designed so that any future model can be slotted in by replacing the `predict()` step.

| Model | Window Acc | Trial Acc | LOSO Window | LOSO Trial |
|---|---|---|---|---|
| Random chance | 50.0% | 50.0% | 50.0% | 50.0% |
| Backward TRF (notebook 06) | ~70–75% | ~80–85% | ~58–62% | ~63–68% |
| EEG + gaze + IMU fusion (notebook 07) | ~72–78% | ~82–88% | ~62–66% | ~67–72% |

(Final numbers depend on bad-channel policy, ridge regularization, and lag window — set the protocol cells above and rerun.)
"""),

    ("md", """\
## 4. Submission template

To submit a new model to the maestro-eeg-dataset leaderboard, ship a Python file that exposes:

```python
def predict(eeg, gaze=None, imu=None, audio_features=None, **kwargs) -> int:
    \"\"\"Return the attended speaker (1 or 2 for within-device binary AAD; or 1..4
    for the harder 4-way task) for ONE 2-second window.

    Inputs:
      eeg  : (T_eeg, 32)  float32 ndarray, 1-32 Hz bandpassed @ 64 Hz
      gaze : (T_gaze, 21) float32 ndarray @ ~50 Hz, may contain NaN
      imu  : (T_imu,  6)  float32 ndarray @ ~95 Hz
      audio_features: dict[int -> (T_aud,) float32] — Hilbert envelopes per
                       candidate speaker @ 64 Hz; only the candidates allowed
                       in the binary protocol are supplied (typically the L/R
                       of the same device).
    \"\"\"
    ...
```

Then run the protocol harness below to get the four numbers in the table.
"""),

    ("code", """\
# --- Sketch of the harness; fill in `your_model_predict` to use ---
def evaluate(predict_fn, repo_id=REPO_ID, segment_length=2.0, overlap=0.5):
    \"\"\"Returns dict of {window_acc, trial_acc, loso_window, loso_trial}.\"\"\"
    # within-subject
    within_w, within_t = [], []
    for subj in subj_csv["subject_id"]:
        s_int = int(subj.lstrip("S"))
        ds = load_aad(subjects=[s_int], trials="main",
                      modalities=["eeg","gaze","imu","audio"],
                      segment_length=segment_length, overlap=overlap,
                      bad_channels="zero", normalize="zscore",
                      repo_id=repo_id)
        # ... iterate ds, call predict_fn, accumulate accuracy
    return {"window_acc": np.nan, "trial_acc": np.nan,
            "loso_window": np.nan, "loso_trial": np.nan}

print("Plug your `predict_fn` into evaluate(...) to score your model.")
"""),

    ("md", """\
## 5. Reporting and citation

If you use this dataset in a publication, please report:

* The exact `maestro-loader` version (`pip show maestro-loader`) used.
* The hyperparameters of any pre-processing (filter band, resample target, normalisation, bad-channel policy).
* All four primary metrics on the **full** dataset (not the sample).
* CIs computed across LOSO folds for the LOSO numbers; across subjects for the within-subject numbers.

Citation:

```bibtex
@misc{maestro_eeg_2026,
  title  = {{maestro-eeg-dataset}: A Multimodal Auditory-Attention Dataset with EEG, Gaze, IMU, and Egocentric Video},
  author = {Alavi, Ali and Hasan, N. and Williamson, D.},
  year   = {2026},
  publisher = {Hugging Face},
  howpublished = {\\url{https://huggingface.co/datasets/aspire-osu/maestro-eeg-dataset}},
}
```
"""),
]
