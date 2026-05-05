"""Notebook 05 — EEG stimulus decoding (backward TRF)."""

CELLS = [
    ("md", """\
# Notebook 05 — EEG stimulus decoding (backward TRF)

The classical AAD baseline: a **backward Time-Response Function** that reconstructs the speech-envelope of the **attended** speaker from the listener's EEG.

For each subject we:

1. Pull all main trials (EEG + audio).
2. Compute Hilbert envelopes for the attended and one unattended (competing) speaker.
3. Lag-augment the EEG (0–250 ms post-stimulus, the classical TRF window).
4. Fit a per-subject Ridge regression on a train fold; evaluate reconstruction Pearson r on a held-out fold.
5. Compare reconstruction quality of the **attended** vs. **unattended** envelope — a higher attended r is the signature of selective attention in EEG.

We use only `numpy`, `scipy`, `scikit-learn`, and `maestro-loader` — no heavy MNE pipeline required.
"""),

    ("code", """\
%%capture
%pip install --quiet "maestro-loader>=0.1.2" "scikit-learn>=1.3" "scipy>=1.10" "matplotlib>=3.7" "seaborn>=0.13"
"""),

    ("code", """\
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy.signal import hilbert, resample_poly, butter, sosfiltfilt
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from maestro_loader import load_aad

sns.set_context("paper", font_scale=1.0); sns.set_style("ticks")
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})

REPO_ID_FULL   = "aspire-osu/maestro-eeg-dataset"
REPO_ID_SAMPLE = "aspire-osu/maestro-eeg-dataset-sample"
USE_SAMPLE     = True
REPO_ID = REPO_ID_SAMPLE if USE_SAMPLE else REPO_ID_FULL

# --- Hyperparameters (standard AAD literature defaults) ---
EEG_FS_OUT = 64.0          # downsample EEG to envelope rate
TRF_LAGS_S = (0.0, 0.250)  # backward lags (s)
EEG_BAND   = (1.0, 32.0)   # band-pass before TRF
RIDGE_ALPHA = 1.0
N_FOLDS    = 5

PROBE_SUBJECT = 1
"""),

    ("md", """\
## 1. Helpers — envelope, EEG resample, lag matrix
"""),

    ("code", """\
def speech_envelope(wav: np.ndarray, fs_in: float, fs_out: float = EEG_FS_OUT) -> np.ndarray:
    env = np.abs(hilbert(wav))
    g = int(round(fs_in / fs_out))
    return resample_poly(env, up=1, down=g)

def filter_eeg(eeg: np.ndarray, fs: float, band=EEG_BAND) -> np.ndarray:
    sos = butter(4, band, btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, eeg, axis=0)

def downsample_eeg(eeg: np.ndarray, fs_in: float, fs_out: float = EEG_FS_OUT) -> np.ndarray:
    g = int(round(fs_in / fs_out))
    return resample_poly(eeg, up=1, down=g, axis=0)

def lagged_features(X: np.ndarray, fs: float, lag_window=TRF_LAGS_S) -> np.ndarray:
    \"\"\"X: (T, C); returns (T, C * n_lags) with lags [0, max_lag] in samples.\"\"\"
    n_lags = int(round((lag_window[1] - lag_window[0]) * fs)) + 1
    pad = np.zeros((n_lags - 1, X.shape[1]))
    Xp = np.vstack([pad, X])
    return np.hstack([Xp[i: i + X.shape[0]] for i in range(n_lags)])
"""),

    ("md", """\
## 2. Pull EEG + audio for one subject, all main trials

Each trial yields one (T, C) EEG tensor at 500 Hz and 6 audio waveforms at 16 kHz. We resample EEG and audio envelopes to 64 Hz and align lengths.
"""),

    ("code", """\
def fetch_trial(subject: int, trial: int):
    ds = load_aad(
        subjects=[subject], trials=[trial],
        modalities=["eeg", "audio"],
        segment_length=None, normalize=None,
        bad_channels="zero", reference="raw",
        repo_id=REPO_ID,
    )
    return ds[0]

PROBE_TRIALS = list(range(1, 6)) if USE_SAMPLE else list(range(1, 21))   # all 5 (sample) or 20 (full)
print(f"Probing S{PROBE_SUBJECT:02d} on {len(PROBE_TRIALS)} trials...")

# --- Pull and align ---
trials_data = []
for k in PROBE_TRIALS:
    try:
        s = fetch_trial(PROBE_SUBJECT, k)
    except Exception as e:
        print(f"  trial {k}: SKIP ({e})"); continue
    eeg = filter_eeg(s["eeg"], s["eeg_sfreq"])
    eeg = downsample_eeg(eeg, s["eeg_sfreq"])
    eeg = StandardScaler().fit_transform(eeg)
    envs = {spk: speech_envelope(s["audio"][spk], s["audio_sfreq"]) for spk in s["audio"]}
    T = min(eeg.shape[0], min(e.shape[0] for e in envs.values()))
    eeg = eeg[:T]
    envs = {k: e[:T] for k, e in envs.items()}
    trials_data.append({"trial": k, "eeg": eeg, "envs": envs, "attended": s["attended_speaker"]})
print(f"  retained {len(trials_data)} trials")
"""),

    ("md", """\
## 3. Train and evaluate the backward TRF (per-trial CV within subject)

We pool trials, build a single design matrix `X = lag_aug(EEG)` and a single target `y = envelope_attended`, then run 5-fold cross-validation on contiguous segments.
"""),

    ("code", """\
def build_design(trials_data, target="attended", competing_offset=1):
    \"\"\"Stack EEG lag-features and the chosen envelope across trials.\"\"\"
    X_parts, y_parts = [], []
    for td in trials_data:
        X = lagged_features(td["eeg"], EEG_FS_OUT)
        if target == "attended":
            y = td["envs"][td["attended"]]
        elif target == "competing":
            # Pick the lowest-numbered attendable speaker that's NOT the attended one.
            others = [s for s in td["envs"] if s != td["attended"] and s in (1,2,3,4)]
            y = td["envs"][others[0]]
        else:
            raise ValueError(target)
        X_parts.append(X); y_parts.append(y)
    return np.vstack(X_parts), np.concatenate(y_parts)

X_att, y_att = build_design(trials_data, target="attended")
X_unt, y_unt = build_design(trials_data, target="competing")
print(f"design X: {X_att.shape}, y: {y_att.shape}")
"""),

    ("code", """\
def cv_reconstruction_r(X, y, n_folds=N_FOLDS, alpha=RIDGE_ALPHA):
    kf = KFold(n_splits=n_folds, shuffle=False)
    rs = []
    for fold, (tr, te) in enumerate(kf.split(X)):
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        sx = StandardScaler().fit(Xtr); Xtr = sx.transform(Xtr); Xte = sx.transform(Xte)
        sy = StandardScaler().fit(ytr.reshape(-1, 1)); ytr = sy.transform(ytr.reshape(-1, 1)).ravel(); yte = sy.transform(yte.reshape(-1, 1)).ravel()
        model = Ridge(alpha=alpha).fit(Xtr, ytr)
        pred = model.predict(Xte)
        r = np.corrcoef(pred, yte)[0, 1]
        rs.append(r)
    return rs

r_attended  = cv_reconstruction_r(X_att, y_att)
r_competing = cv_reconstruction_r(X_unt, y_unt)
print(f"Attended  reconstruction r: mean = {np.mean(r_attended):+.3f}  (folds: {[f'{r:.3f}' for r in r_attended]})")
print(f"Competing reconstruction r: mean = {np.mean(r_competing):+.3f}  (folds: {[f'{r:.3f}' for r in r_competing]})")
"""),

    ("md", """\
## 4. Visualise: attended vs. competing reconstruction

A consistent gap (attended > competing) is the canonical AAD effect.
"""),

    ("code", """\
folds = np.arange(1, N_FOLDS + 1)
fig, ax = plt.subplots(figsize=(5.6, 3.4))
ax.bar(folds - 0.18, r_attended,  width=0.36, label="attended",  color="#2c7fb8", edgecolor="white")
ax.bar(folds + 0.18, r_competing, width=0.36, label="competing", color="#cb6f7d", edgecolor="white")
ax.axhline(0, color="#bbbbbb", lw=0.7)
ax.set(xlabel="CV fold", ylabel="Reconstruction Pearson r",
       title=f"S{PROBE_SUBJECT:02d} backward-TRF reconstruction (lags {int(TRF_LAGS_S[1]*1000)} ms)")
ax.legend(frameon=False); ax.set_xticks(folds)
plt.tight_layout(); plt.show()
"""),

    ("md", """\
## 5. TRF kernel inspection (model interpretability)

Reshape the ridge weights back to (lag, channel) and average across channels — a typical TRF peaks around 100–200 ms post-stimulus.
"""),

    ("code", """\
sx = StandardScaler().fit(X_att); Xs = sx.transform(X_att)
sy = StandardScaler().fit(y_att.reshape(-1, 1)); ys = sy.transform(y_att.reshape(-1, 1)).ravel()
model_full = Ridge(alpha=RIDGE_ALPHA).fit(Xs, ys)

n_lags = int(round((TRF_LAGS_S[1] - TRF_LAGS_S[0]) * EEG_FS_OUT)) + 1
W = model_full.coef_.reshape(n_lags, -1)               # (lag, channel)
trf_avg = W.mean(axis=1)
lags_ms = np.linspace(TRF_LAGS_S[0]*1000, TRF_LAGS_S[1]*1000, n_lags)

fig, ax = plt.subplots(figsize=(6, 3.0))
ax.plot(lags_ms, trf_avg, lw=1.6, color="#2c7fb8")
ax.axhline(0, color="#bbbbbb", lw=0.7)
ax.set(xlabel="Lag (ms)", ylabel="weight (a.u.)",
       title=f"Backward TRF — channel-averaged · S{PROBE_SUBJECT:02d}")
plt.tight_layout(); plt.show()
"""),

    ("md", """\
## 6. Going further

* Loop over all 16 subjects and aggregate — see notebook 06 (`06_attention_decoding`) where we do exactly that and convert these correlations into a binary AAD decision.
* Replace ridge with [boosting](https://github.com/LABSN/expyfun) or a small CNN — notebook 06 also shows how the loader plugs straight into a `torch.utils.data.DataLoader`.
* Add **forward** TRF (envelope → EEG): swap `X` and `y` and inspect the resulting auditory cortex topography (notebook 02 covers spatial layout).
"""),
]
