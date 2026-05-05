"""Notebook 03 — Gaze dynamics."""

CELLS = [
    ("md", """\
# Notebook 03 — Gaze dynamics

The Tobii Glasses 3 stream provides 21 columns at ~50 Hz: 2-D and 3-D gaze, per-eye gaze origin/direction, pupil diameter. We characterise:

1. **Validity** — fraction of usable samples (non-NaN) per trial.
2. **Trajectories** — 2-D fixation-density heatmaps in the recording reference frame.
3. **Pupil dynamics** — left-vs-right symmetry, drift, blink rate.
4. **Saccade & fixation rates** — derived from the velocity trace.

These features are inputs to the multimodal-fusion AAD model in notebook 07.
"""),

    ("code", """\
%%capture
%pip install --quiet "maestro-loader>=0.1.2" "scipy>=1.10" "matplotlib>=3.7" "seaborn>=0.13"
"""),

    ("code", """\
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy import signal, ndimage
from maestro_loader import load_aad

sns.set_context("paper", font_scale=1.0); sns.set_style("ticks")
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})

REPO_ID_FULL   = "aspire-osu/maestro-eeg-dataset"
REPO_ID_SAMPLE = "aspire-osu/maestro-eeg-dataset-sample"
USE_SAMPLE     = True
REPO_ID = REPO_ID_SAMPLE if USE_SAMPLE else REPO_ID_FULL
"""),

    ("md", """\
## 1. Pull whole-trial gaze for several subjects
"""),

    ("code", """\
PROBE_SUBJECTS = [1, 7, 14] if USE_SAMPLE else [1, 5, 8, 12]
PROBE_TRIAL    = 1

def fetch_gaze_trial(subject, trial):
    ds = load_aad(
        subjects=[subject], trials=[trial], modalities=["gaze"],
        segment_length=None, normalize=None, repo_id=REPO_ID,
    )
    s = ds[0]
    cols = ["t"] + s["gaze_columns"] if "t" not in s["gaze_columns"] else s["gaze_columns"]
    df = pd.DataFrame(s["gaze"], columns=s["gaze_columns"])
    df["sfreq"] = s["gaze_sfreq"]
    return df

trials_g = {s: fetch_gaze_trial(s, PROBE_TRIAL) for s in PROBE_SUBJECTS}
for s, df in trials_g.items():
    print(f"  S{s:02d}: {df.shape}, sfreq={df['sfreq'].iloc[0]:.1f} Hz")
"""),

    ("md", """\
## 2. Gaze validity

Tobii returns NaN when the eye-tracker can't get a confident estimate (blink, gaze off-screen, occluded). Validity = `1 − fraction_NaN` on the 2-D gaze coordinates.
"""),

    ("code", """\
val = []
for s, df in trials_g.items():
    valid_2d = (~df[["gaze2d_x", "gaze2d_y"]].isna().any(axis=1)).mean()
    val.append({"subject": f"S{s:02d}", "valid_frac": valid_2d})
print(pd.DataFrame(val).to_string(index=False))
"""),

    ("md", """\
## 3. Fixation-density heatmaps

A simple 2-D histogram of valid `(gaze2d_x, gaze2d_y)` samples, smoothed with a Gaussian kernel. The Tobii 2-D coordinate is normalised in [0, 1].
"""),

    ("code", """\
fig, axes = plt.subplots(1, len(PROBE_SUBJECTS), figsize=(3.5 * len(PROBE_SUBJECTS), 3.4),
                          sharex=True, sharey=True)
if len(PROBE_SUBJECTS) == 1: axes = [axes]
for ax, s in zip(axes, PROBE_SUBJECTS):
    df = trials_g[s].dropna(subset=["gaze2d_x", "gaze2d_y"])
    h, xe, ye = np.histogram2d(df["gaze2d_x"], df["gaze2d_y"],
                                bins=80, range=[[0, 1], [0, 1]])
    h = ndimage.gaussian_filter(h, sigma=2.0)
    ax.imshow(h.T, origin="lower", extent=(0, 1, 0, 1), cmap="magma", aspect="equal")
    ax.set(xlabel="gaze2d_x", title=f"S{s:02d}")
axes[0].set_ylabel("gaze2d_y")
plt.suptitle("Fixation density (Gaussian-smoothed)", y=1.04, fontsize=11)
plt.tight_layout(); plt.show()
"""),

    ("md", """\
## 4. Pupil dynamics

Left and right pupil diameters (mm) should track each other closely. Persistent left-right asymmetry indicates an instrument issue.
"""),

    ("code", """\
fig, axes = plt.subplots(1, len(PROBE_SUBJECTS), figsize=(4 * len(PROBE_SUBJECTS), 3.0), sharey=True)
if len(PROBE_SUBJECTS) == 1: axes = [axes]
for ax, s in zip(axes, PROBE_SUBJECTS):
    df = trials_g[s]
    t = np.arange(len(df)) / df["sfreq"].iloc[0]
    ax.plot(t, df["L_pupil"], lw=0.8, color="#3182bd", label="left", alpha=0.85)
    ax.plot(t, df["R_pupil"], lw=0.8, color="#e6550d", label="right", alpha=0.85)
    ax.set(xlabel="Time (s)", title=f"S{s:02d} · pupil")
    if ax is axes[0]:
        ax.set_ylabel("Pupil diameter (mm)")
        ax.legend(frameon=False, loc="upper right")
plt.tight_layout(); plt.show()
"""),

    ("md", """\
## 5. Saccade and fixation rates (derived from velocity)

We approximate saccade rate by counting peaks in the gaze-velocity magnitude exceeding ~30°/s (in 2-D normalised units, ~0.5 / s on the [0,1] frame).
"""),

    ("code", """\
def saccade_rate(df: pd.DataFrame, vel_thresh: float = 0.5) -> float:
    fs = df["sfreq"].iloc[0]
    xy = df[["gaze2d_x", "gaze2d_y"]].interpolate(limit=10).to_numpy()
    if len(xy) < 4: return np.nan
    vx, vy = np.gradient(xy, axis=0).T
    speed = np.hypot(vx, vy) * fs
    peaks, _ = signal.find_peaks(speed, height=vel_thresh, distance=int(0.05 * fs))
    return len(peaks) / (len(xy) / fs)             # peaks/sec

sac = pd.DataFrame([
    {"subject": f"S{s:02d}", "saccade_rate_hz": saccade_rate(df), "valid_frac": float((~df[["gaze2d_x","gaze2d_y"]].isna().any(axis=1)).mean())}
    for s, df in trials_g.items()
])
print(sac.to_string(index=False))
"""),

    ("md", """\
## 6. What gaze tells us about attention

Auditory attention does **not** require eye movement, but covert gaze can still bias toward the spatial location of the attended speaker. Notebook 07 quantifies this by training an attention decoder on gaze alone and fusing it with EEG.
"""),
]
