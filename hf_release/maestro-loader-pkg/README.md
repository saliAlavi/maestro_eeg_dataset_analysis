# maestro-loader

Loader for the [`aspire-osu/maestro-eeg-dataset`](https://huggingface.co/datasets/aspire-osu/maestro-eeg-dataset)
— a multimodal auditory-attention-decoding (AAD) dataset with **EEG, eye-gaze,
head-IMU, spatialized multi-speaker audio, and Tobii Glasses 3 scene video**
(16 subjects × 105 trials).

The loader's job is to hand you **perfectly cross-modal-aligned segments** and
**leakage-free train/test splits**, in either a near-raw or a model-ready form.

## Install

```bash
pip install maestro-loader            # core (numpy/pandas/scipy/soundfile/hf_hub)
pip install maestro-loader[torch]     # + PyTorch tensors & DataLoaders
pip install maestro-loader[mne]       # + MNE EEG preprocessing / bad-chan interp
pip install maestro-loader[video]     # + PyAV scene-video decoding
pip install maestro-loader[all]
```

## What "aligned" means

Each modality runs on its own clock (EEG internal clock, audio playback clock,
Tobii recording clock). Every trial ships a `timing` record with the wall-clock
(unix) anchors, and the loader uses them to clip all streams to the single
window where **EEG, audio playback, and the Tobii recording are simultaneously
live** — the *perfect-alignment window*. With `preprocess=True` (or any
`target_sfreq`) all continuous streams are additionally resampled onto **one
shared time grid**, so EEG / gaze / IMU / audio come out the same length and are
sample-for-sample aligned. IMU and video share the gaze (Tobii) clock; EEG and
audio are tied in through the per-device playback timestamps.

## Quick start

```python
from maestro_loader import load_aad

# (A) Near-raw, aligned, native sample rates — bring your own preprocessing
ds = load_aad(
    subjects=[1, 2, 3], trials="main",
    modalities="all",                       # eeg, gaze, imu, audio, video
    segment_length=5.0, overlap=0.5,        # seconds
    preprocess=False,                       # aligned raw
)
seg = ds[0]
seg["eeg"]    # (32, 2500)  @ 500 Hz        seg["gaze"]  # (19, ~250) @ ~50 Hz
seg["imu"]    # (6, ~500)   @ ~100 Hz       seg["audio"] # (6, 80000) @ 16 kHz  (6 speakers)
seg["video_path"], seg["video_frame_range"]
seg["attended_speaker"]   # 1..4    seg["hemisphere"]  # 0=L 1=R    seg["inout"]  # 0=inner 1=outer

# (B) Model-ready: decoder EEG pipeline + everything on a common 64 Hz grid
ds = load_aad(
    subjects=[1], modalities=["eeg", "gaze", "imu", "audio"],
    segment_length=5.0, preprocess=True,    # notch+bandpass+robust ref, target_sfreq=64
    audio_feature="mel",                    # 28-band log-mel "gammatone proxy"
)
seg = ds[0]
seg["eeg"]    # (32, 320)        seg["audio"]  # (6, 28, 320)   — all length 320, aligned
```

`preprocess=True` mirrors the decoder's front end (notch 60 Hz, band-pass
1–40 Hz, linked-mastoid/avg robust reference, bad-channel interpolation). Pass a
dict to override, e.g. `preprocess={"l_freq": 0.5, "h_freq": 30, "reference": "average"}`.

## Train / test splits — trial-disjoint, LOSO **and** intra-subject

The atomic unit of every split is a **`(subject, trial)`** — a trial is *never*
divided across train and test, so windows can't leak between them. This holds in
both settings.

```python
from maestro_loader import load_aad, get_dataloaders

# Leave-one-subject-out (fold = held-out subject index, 0..15)
train_dl, test_dl = get_dataloaders(
    setting="loso", fold=3,
    modalities=["eeg", "audio"], segment_length=5.0, preprocess=True,
    batch_size=64,
)

# Intra-subject k-fold (trial-disjoint within each subject)
train_dl, test_dl = get_dataloaders(
    setting="intra", fold=0, n_folds=5, scheme="chrono",   # or "random"
    subjects=[1], modalities=["eeg"], segment_length=5.0,
)
```

Need the raw datasets (or NumPy)? Use `load_aad(..., split={"setting": "loso",
"fold": 3})` → `(train_ds, test_ds)`. Verify the guarantee yourself with
`from maestro_loader import assert_trial_disjoint; assert_trial_disjoint(train.units, test.units)`.

**Note on LOSO trial-disjointness.** The 105 *stimuli* are shared across
subjects by design, so LOSO holds out the *subject* (all of their recordings);
no recording — i.e. no `(subject, trial)` — is ever in both splits. If you need a
held-out *stimulus* set as well, restrict `trials=` to disjoint id ranges per
call.

## Streaming from HuggingFace vs local

Omit `local_path` to stream (files are lazily downloaded & cached by
`huggingface_hub`); pass `local_path="/path/to/maestro-eeg-dataset"` to read a
local copy. `return_format="torch"|"numpy"`.

See the dataset card on Hugging Face for the on-disk schema and channel order.
