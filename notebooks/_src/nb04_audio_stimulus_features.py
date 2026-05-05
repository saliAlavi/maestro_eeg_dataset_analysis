"""Notebook 04 — Audio stimulus features."""

CELLS = [
    ("md", """\
# Notebook 04 — Audio stimulus features

The maestro stimuli ship as 6 mono FLAC files per trial — one per speaker — at 16 kHz native sample rate. Standard auditory-attention-decoding pipelines reduce these to two low-dimensional time-series:

1. **Broadband Hilbert envelope** at ~64 Hz — the canonical AAD feature for backward stimulus reconstruction.
2. **Mel spectrogram** (40 mel bands, ~64 Hz) — richer features for forward decoding.

We compute both, validate that the attended speaker is at least a viable candidate (energy, vocal range), and prepare them for notebook 05.
"""),

    ("code", """\
%%capture
%pip install --quiet "maestro-loader>=0.1.2" "soundfile>=0.12" "scipy>=1.10" "librosa>=0.10" "matplotlib>=3.7" "seaborn>=0.13"
"""),

    ("code", """\
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import soundfile as sf
import librosa
from scipy.signal import hilbert, resample_poly
from huggingface_hub import hf_hub_download
from maestro_loader import load_aad

sns.set_context("paper", font_scale=1.0); sns.set_style("ticks")
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})

REPO_ID_FULL   = "aspire-osu/maestro-eeg-dataset"
REPO_ID_SAMPLE = "aspire-osu/maestro-eeg-dataset-sample"
USE_SAMPLE     = True
REPO_ID = REPO_ID_SAMPLE if USE_SAMPLE else REPO_ID_FULL

AUDIO_FS_OUT = 64.0          # downsample target for envelopes / mel
N_MELS       = 40
"""),

    ("md", """\
## 1. Pull a trial's worth of audio (all 6 speakers)

We use the loader without segmentation so we get the whole 30-s trial per speaker.
"""),

    ("code", """\
TRIAL_ID = 1
ds = load_aad(
    subjects=[1], trials=[TRIAL_ID],
    modalities=["audio"],
    segment_length=None, normalize=None, repo_id=REPO_ID,
)
sample = ds[0]
audio_fs = sample["audio_sfreq"]
print(f"Trial: {sample['trial_id']}  attended speaker: {sample['attended_speaker']}  "
      f"audio Fs = {audio_fs:.0f} Hz")

waves = sample["audio"]
for spk in sorted(waves):
    print(f"  speaker {spk}: {waves[spk].shape}  rms={np.sqrt(np.mean(waves[spk]**2)):.4f}")
"""),

    ("md", """\
## 2. Broadband envelope

Hilbert magnitude → 8-Hz low-pass → resample to 64 Hz. This is exactly the input every linear stimulus-reconstruction model in the AAD literature uses.
"""),

    ("code", """\
def hilbert_envelope(wav: np.ndarray, fs_in: float, fs_out: float = AUDIO_FS_OUT) -> np.ndarray:
    env = np.abs(hilbert(wav))
    g = int(round(fs_in / fs_out))
    return resample_poly(env, up=1, down=g)

envs = {spk: hilbert_envelope(w, audio_fs) for spk, w in waves.items()}

fig, ax = plt.subplots(figsize=(9, 3.4))
attended = sample["attended_speaker"]
t = np.arange(envs[1].shape[0]) / AUDIO_FS_OUT
for spk, env in envs.items():
    style = dict(lw=1.4, color="crimson") if spk == attended else dict(lw=0.7, alpha=0.6)
    ax.plot(t, env / env.max(), label=f"spk {spk}{' (attended)' if spk == attended else ''}", **style)
ax.set(xlabel="Time (s)", ylabel="normalised envelope",
       title=f"Hilbert envelopes · trial {sample['trial_id']} (attended in red)")
ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=8)
plt.tight_layout(); plt.show()
"""),

    ("md", """\
## 3. Mel spectrogram per speaker

40 mel bands (50–7500 Hz), 25 ms window, 15 ms hop → ~64 Hz frame rate. Useful for forward decoding (EEG → audio features) and as a richer alternative to the broadband envelope.
"""),

    ("code", """\
def mel_spectrogram(wav: np.ndarray, fs: float, n_mels: int = N_MELS) -> np.ndarray:
    return librosa.feature.melspectrogram(
        y=wav.astype(np.float32), sr=int(fs),
        n_fft=512, hop_length=int(fs * 0.015), win_length=int(fs * 0.025),
        n_mels=n_mels, fmin=50, fmax=7500,
    )

mels = {spk: librosa.power_to_db(mel_spectrogram(w, audio_fs)) for spk, w in waves.items()}

fig, axes = plt.subplots(2, 3, figsize=(12, 5), sharex=True, sharey=True)
for ax, spk in zip(axes.flat, sorted(mels)):
    M = mels[spk]
    im = ax.imshow(M, aspect="auto", origin="lower", cmap="magma",
                   extent=(0, M.shape[1] * 0.015, 0, N_MELS))
    title = f"speaker {spk}{' · ATTENDED' if spk == attended else ''}"
    ax.set(title=title, xlabel="Time (s)", ylabel="mel band")
plt.tight_layout(); plt.show()
"""),

    ("md", """\
## 4. Speaker-level spectral fingerprints

Average the mel spectrogram over time → a 40-d vector per speaker. This is a quick check that the 6 speakers really are 6 distinct voices.
"""),

    ("code", """\
fingerprint = pd.DataFrame({f"spk{s}": mels[s].mean(axis=1) for s in sorted(mels)})

corr = fingerprint.corr()
fig, ax = plt.subplots(figsize=(4.6, 4.0))
sns.heatmap(corr, vmin=0.6, vmax=1.0, cmap="rocket_r",
            annot=True, fmt=".2f", ax=ax, cbar_kws={"label": "Pearson r"})
ax.set(title="Mel-fingerprint correlation across speakers (one trial)")
plt.tight_layout(); plt.show()
"""),

    ("md", """\
## 5. Attended-speaker priors and SNR labels

`metadata/trials.csv` ships per-channel power values. They're the ground-truth used to set per-trial SNR. Plotting the attended-speaker channel power against trial SNR is a sanity check of the audio engineering.
"""),

    ("code", """\
trials_meta = pd.read_csv(hf_hub_download(REPO_ID, "metadata/trials.csv", repo_type="dataset"))

# Speaker → power column
COL = {1: "device1_left_power", 2: "device1_right_power",
       3: "device2_left_power", 4: "device2_right_power",
       5: "device3_left_power", 6: "device3_right_power"}

m = trials_meta[trials_meta["kind"] == "main"].copy()
m["attended_power"] = m.apply(lambda r: r[COL[int(r["attended_speaker"])]], axis=1)
m["attended_power_db"] = 10 * np.log10(m["attended_power"] + 1e-12)

fig, ax = plt.subplots(figsize=(5.5, 3.4))
ax.scatter(m["snr_db"], m["attended_power_db"], s=18, color="#2c7fb8", alpha=0.6, edgecolor="white")
ax.set(xlabel="Trial SNR (dB, from CSV)",
       ylabel="Attended-speaker channel power (dB)",
       title="Audio-mix calibration check")
plt.tight_layout(); plt.show()
"""),

    ("md", """\
## 6. Outputs ready for notebook 05

We now have:
* `envs[spk]` — broadband envelope at 64 Hz, one array per speaker.
* `mels[spk]` — log-mel spectrogram at ~64 Hz, one (40, T) array per speaker.

Notebook 05 (`05_eeg_stimulus_decoding`) consumes the envelope and trains a backward TRF that reconstructs it from EEG, then asks "which speaker's envelope does my reconstruction look most like?" — that is the AAD task.
"""),
]
