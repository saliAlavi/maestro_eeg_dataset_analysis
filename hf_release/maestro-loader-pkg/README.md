# maestro-loader

Loader for the [`aspire-osu/maestro-eeg-dataset`](https://huggingface.co/datasets/aspire-osu/maestro-eeg-dataset) — a multimodal auditory-attention-decoding (AAD) dataset with EEG, gaze, IMU, audio, and Tobii Glasses 3 scene video.

## Install

```bash
pip install maestro-loader            # core
pip install maestro-loader[torch]     # + PyTorch return format
pip install maestro-loader[mne]       # + MNE-based bad-channel interpolation
pip install maestro-loader[all]
```

## Quick start

```python
from maestro_loader import load_aad

ds = load_aad(
    subjects=[1, 2, 3],
    trials="main",
    modalities=["eeg", "gaze", "audio"],
    segment_length=2.0,        # seconds
    overlap=0.5,
    normalize="zscore",
    bad_channels="interpolate",
    return_format="torch",
)

for sample in ds:
    eeg = sample["eeg"]                 # (T, 32) torch.Tensor
    gaze = sample["gaze"]               # (T, 21)
    audio = sample["audio"]             # dict[speaker_id → (T_audio,) waveform]
    label = sample["attended_speaker"]  # int in {1..4}
```

See the dataset README on Hugging Face for the full data schema.
