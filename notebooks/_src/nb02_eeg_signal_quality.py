"""Notebook 02 — EEG signal quality."""

CELLS = [
    ("md", """\
# Notebook 02 — EEG signal quality

We audit the 32-channel ANT Neuro EEG at 500 Hz across multiple subjects:

1. **Power spectral density (PSD)** per channel — identifies broadband noise, line interference, and dead channels.
2. **Per-channel range / std** — exposes saturation and drift outliers.
3. **Drift / DC level** — slow trends that high-pass filtering will remove.
4. **Spatial summary** — a topographic-style overview of channel-level health.
5. **Bad-channel verification** — does the auto-flag in `metadata/bad_channels.csv` agree with what we see here?

Findings here motivate the bad-channel and reference choices used in the AAD baselines (notebooks 05–08).
"""),

    ("code", """\
%%capture
%pip install --quiet "maestro-loader>=0.1.2" "scipy>=1.10" "matplotlib>=3.7" "seaborn>=0.13"
"""),

    ("code", """\
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy import signal
from maestro_loader import load_aad
from huggingface_hub import hf_hub_download

sns.set_context("paper", font_scale=1.0); sns.set_style("ticks")
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})

REPO_ID_FULL   = "aspire-osu/maestro-eeg-dataset"
REPO_ID_SAMPLE = "aspire-osu/maestro-eeg-dataset-sample"
USE_SAMPLE     = True
REPO_ID = REPO_ID_SAMPLE if USE_SAMPLE else REPO_ID_FULL

# Subjects available in the chosen repo (sample = 3, full = 16)
PROBE_SUBJECTS = [1, 7, 14] if USE_SAMPLE else [1, 5, 8, 12]
PROBE_TRIAL    = 1
SFREQ          = 500.0
"""),

    ("md", """\
## 1. Pull one main trial per subject

Whole-trial EEG (≈30 s × 500 Hz × 32 channels), no segmentation, no normalisation, **no bad-channel handling** — we want to see the raw quality first.
"""),

    ("code", """\
def fetch_trial_eeg(subject: int, trial: int) -> tuple[np.ndarray, list[str]]:
    ds = load_aad(
        subjects=[subject], trials=[trial], modalities=["eeg"],
        segment_length=None, normalize=None, bad_channels="raw",
        repo_id=REPO_ID,
    )
    s = ds[0]
    return s["eeg"], s["eeg_channels"]

trial_eeg = {s: fetch_trial_eeg(s, PROBE_TRIAL) for s in PROBE_SUBJECTS}
for s, (e, ch) in trial_eeg.items():
    print(f"  S{s:02d} eval_{PROBE_TRIAL:03d}: {e.shape}  range=[{e.min():.4f}, {e.max():.4f}]  std={e.std():.4f}")
"""),

    ("md", """\
## 2. Power spectral density per channel

Welch's method, 4-second segments, Hann window, 50% overlap. We expect:
- A **1/f-like** roll-off from 1–40 Hz on healthy channels.
- A spike near **60 Hz** = US line noise.
- Flat / clipped curves on saturated channels (e.g. M2).
"""),

    ("code", """\
def channel_psds(eeg: np.ndarray, fs: float = SFREQ) -> tuple[np.ndarray, np.ndarray]:
    nperseg = int(4 * fs)
    f, pxx = signal.welch(eeg, fs=fs, nperseg=min(nperseg, eeg.shape[0]), axis=0)
    return f, pxx                                  # pxx: (n_freq, n_channels)

fig, axes = plt.subplots(1, len(PROBE_SUBJECTS), figsize=(4.2 * len(PROBE_SUBJECTS), 3.4),
                          sharey=True)
if len(PROBE_SUBJECTS) == 1: axes = [axes]
for ax, s in zip(axes, PROBE_SUBJECTS):
    eeg, channels = trial_eeg[s]
    f, pxx = channel_psds(eeg)
    pxx_db = 10 * np.log10(pxx + 1e-20)
    bad_idx = [i for i, ch in enumerate(channels) if ch in ("M1", "M2")]
    for i in range(eeg.shape[1]):
        c = "crimson" if i in bad_idx else "#3b3b3b"
        a = 0.95 if i in bad_idx else 0.25
        ax.semilogx(f[1:], pxx_db[1:, i], lw=0.6, color=c, alpha=a)
    ax.axvline(60, color="#aaaaaa", ls=":", lw=0.8)
    ax.set(xlim=(1, 100), xlabel="Frequency (Hz)", title=f"S{s:02d} · all 32 ch")
axes[0].set_ylabel("PSD (10·log₁₀)")
plt.suptitle("Per-channel PSD (red = mastoid M1/M2)", y=1.04, fontsize=11)
plt.tight_layout(); plt.show()
"""),

    ("md", """\
## 3. Per-channel range and standard deviation

Outlier detection at a glance. Anything an order of magnitude above the cohort median std is suspect.
"""),

    ("code", """\
diag = []
for s, (eeg, channels) in trial_eeg.items():
    for i, ch in enumerate(channels):
        x = eeg[:, i]
        diag.append({"subject": f"S{s:02d}", "channel": ch,
                     "std": x.std(),
                     "range": x.max() - x.min(),
                     "frac_at_clip": float((np.abs(x) >= np.abs(x).max() - 1e-7).mean())})
diag_df = pd.DataFrame(diag)

mastoid = diag_df["channel"].isin(["M1", "M2"])
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
sns.boxplot(data=diag_df.loc[~mastoid], x="subject", y="std", ax=axes[0], color="#74c476")
sns.stripplot(data=diag_df.loc[mastoid], x="subject", y="std",
              hue="channel", ax=axes[0], dodge=True, palette={"M1":"crimson","M2":"darkorange"},
              size=7, edgecolor="white", linewidth=0.5)
axes[0].set(title="Channel std (boxes = non-mastoid; markers = mastoid)", ylabel="std")
axes[0].legend(title="", loc="upper right", frameon=False)

sns.boxplot(data=diag_df.loc[~mastoid], x="subject", y="frac_at_clip", ax=axes[1], color="#74c476")
sns.stripplot(data=diag_df.loc[mastoid], x="subject", y="frac_at_clip",
              hue="channel", ax=axes[1], dodge=True, palette={"M1":"crimson","M2":"darkorange"},
              size=7, edgecolor="white", linewidth=0.5)
axes[1].set(title="Fraction of samples at channel max-|abs|", ylabel="fraction")
axes[1].legend_.remove()
plt.tight_layout(); plt.show()
"""),

    ("md", """\
## 4. Drift / low-frequency content

Below ~0.5 Hz the signal carries DC level and slow electrode/sweat drift. Healthy preprocessing high-passes at 1 Hz; we visualise what's removed.
"""),

    ("code", """\
def lowfreq_envelope(eeg: np.ndarray, fs: float = SFREQ, cutoff: float = 0.5) -> np.ndarray:
    sos = signal.butter(4, cutoff, btype="lowpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, eeg, axis=0)

s_demo = PROBE_SUBJECTS[0]
eeg, channels = trial_eeg[s_demo]
drift = lowfreq_envelope(eeg)

fig, ax = plt.subplots(figsize=(8, 3.0))
t = np.arange(eeg.shape[0]) / SFREQ
for ch_name in ["Fp1", "Cz", "O1", "M2"]:
    if ch_name in channels:
        i = channels.index(ch_name)
        ax.plot(t, drift[:, i], lw=1.0, label=ch_name)
ax.set(xlabel="Time (s)", ylabel="< 0.5 Hz amplitude",
       title=f"Drift envelope · S{s_demo:02d} · eval_{PROBE_TRIAL:03d}")
ax.legend(frameon=False, loc="upper right"); plt.tight_layout(); plt.show()
"""),

    ("md", """\
## 5. Spatial channel-health summary

A simple bar-grid arranged loosely by montage region. Mastoids and any saturated channels jump out.
"""),

    ("code", """\
fig, axes = plt.subplots(1, len(PROBE_SUBJECTS), figsize=(5 * len(PROBE_SUBJECTS), 3.0))
if len(PROBE_SUBJECTS) == 1: axes = [axes]
for ax, s in zip(axes, PROBE_SUBJECTS):
    sub = diag_df[diag_df["subject"] == f"S{s:02d}"].copy()
    sub = sub.sort_values("std", ascending=False)
    colors = ["crimson" if c in ("M1","M2") else "#5a8db8" for c in sub["channel"]]
    ax.barh(sub["channel"], sub["std"], color=colors, edgecolor="white")
    ax.set(title=f"S{s:02d} · channel std (sorted)", xlabel="std")
    ax.tick_params(axis="y", labelsize=7)
plt.tight_layout(); plt.show()
"""),

    ("md", """\
## 6. Verification: do flagged channels agree with PSD inspection?

Cross-check the auto-flagged bad channels in `bad_channels.csv` against what we just observed.
"""),

    ("code", """\
bad_csv = pd.read_csv(hf_hub_download(REPO_ID, "metadata/bad_channels.csv", repo_type="dataset"))
print(bad_csv[bad_csv["subject_id"].isin([f"S{s:02d}" for s in PROBE_SUBJECTS])][
    ["subject_id", "bad_channels", "method"]
].to_string(index=False))
"""),

    ("md", """\
## 7. Recommendation for downstream notebooks

* Use `bad_channels="interpolate"` (requires `mne`) when training cross-subject models — replaces the bad-mastoid signals with spherical-spline-interpolated estimates.
* For purely within-subject decoding, `bad_channels="zero"` is sufficient and faster.
* Apply a 1–32 Hz band-pass before feeding to AAD models (notebook 05) to remove drift and line noise.
"""),
]
