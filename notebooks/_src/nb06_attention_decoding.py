"""Notebook 06 — Attention decoding (binary AAD)."""

CELLS = [
    ("md", """\
# Notebook 06 — Attention decoding (binary AAD)

The headline AAD task: given a 2-s EEG window, decide which of two simultaneously presented speakers the listener was attending. We evaluate two protocols:

* **Within-subject**, 5-fold cross-validation on each subject independently.
* **Leave-one-subject-out (LOSO)**, training on N-1 subjects and testing on the held-out one — the harder generalisation setting.

The features are the **two reconstruction correlations** from notebook 05's backward TRF (one r per candidate speaker). Above-chance accuracy on this task is the gold-standard demonstration of EEG-based attention selection.
"""),

    ("code", """\
%%capture
%pip install --quiet "maestro-loader>=0.1.2" "scikit-learn>=1.3" "scipy>=1.10" "tqdm>=4.66" "matplotlib>=3.7" "seaborn>=0.13"
"""),

    ("code", """\
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy.signal import hilbert, resample_poly, butter, sosfiltfilt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
from maestro_loader import load_aad

sns.set_context("paper", font_scale=1.0); sns.set_style("ticks")
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})

REPO_ID_FULL   = "aspire-osu/maestro-eeg-dataset"
REPO_ID_SAMPLE = "aspire-osu/maestro-eeg-dataset-sample"
USE_SAMPLE     = True
REPO_ID = REPO_ID_SAMPLE if USE_SAMPLE else REPO_ID_FULL

# Hyperparameters (matched to notebook 05)
EEG_FS_OUT = 64.0
TRF_LAGS_S = (0.0, 0.250)
EEG_BAND   = (1.0, 32.0)
RIDGE_ALPHA = 1.0
WIN_SEC    = 2.0
"""),

    ("md", """\
## 1. Pull all subjects and trials we have access to

The sample dataset has 3 subjects × 5 trials = 15 trials. The full dataset has 16 × 100 = 1600 main trials. Either is enough to run this analysis end-to-end.
"""),

    ("code", """\
import json
from huggingface_hub import hf_hub_download

subjects_meta = pd.read_csv(hf_hub_download(REPO_ID, "metadata/subjects.csv", repo_type="dataset"))
trials_meta   = pd.read_csv(hf_hub_download(REPO_ID, "metadata/trials.csv", repo_type="dataset"))
SUBJECTS = sorted(int(s.lstrip("S")) for s in subjects_meta["subject_id"])
MAIN_TRIALS = trials_meta.loc[trials_meta["kind"] == "main", "trial_id"].tolist()
MAIN_KS = sorted(int(t.removeprefix("eval_")) for t in MAIN_TRIALS)
print(f"Subjects: {SUBJECTS}  ({len(SUBJECTS)})")
print(f"Main trials: {len(MAIN_KS)}")
"""),

    ("md", """\
## 2. Per-trial pipeline: filter, resample, lag-augment, fit TRF, score windows
"""),

    ("code", """\
def filter_resample(eeg, fs_in, band=EEG_BAND, fs_out=EEG_FS_OUT):
    sos = butter(4, band, btype="bandpass", fs=fs_in, output="sos")
    eeg = sosfiltfilt(sos, eeg, axis=0)
    g = int(round(fs_in / fs_out))
    return resample_poly(eeg, up=1, down=g, axis=0)

def envelope(wav, fs_in, fs_out=EEG_FS_OUT):
    env = np.abs(hilbert(wav))
    g = int(round(fs_in / fs_out))
    return resample_poly(env, up=1, down=g)

def lag_aug(X, fs=EEG_FS_OUT, lag_window=TRF_LAGS_S):
    n_lags = int(round((lag_window[1] - lag_window[0]) * fs)) + 1
    pad = np.zeros((n_lags - 1, X.shape[1]))
    Xp = np.vstack([pad, X])
    return np.hstack([Xp[i: i + X.shape[0]] for i in range(n_lags)])

def fetch_trial_aligned(subj, k):
    \"\"\"Return (eeg_lagaug, env_attended, env_competing, attended).\"\"\"
    ds = load_aad(
        subjects=[subj], trials=[k], modalities=["eeg", "audio"],
        segment_length=None, normalize=None,
        bad_channels="zero", reference="raw", repo_id=REPO_ID,
    )
    s = ds[0]
    eeg = filter_resample(s["eeg"], s["eeg_sfreq"])
    eeg = StandardScaler().fit_transform(eeg)
    envs = {spk: envelope(s["audio"][spk], s["audio_sfreq"]) for spk in s["audio"]}
    T = min(eeg.shape[0], min(e.shape[0] for e in envs.values()))
    attended = s["attended_speaker"]
    others = [spk for spk in (1, 2, 3, 4) if spk != attended]
    competing = others[0]
    Xlag = lag_aug(eeg[:T])
    return Xlag, envs[attended][:T], envs[competing][:T], attended, competing
"""),

    ("md", """\
## 3. Within-subject AAD

For each subject, leave-one-trial-out: fit a backward TRF on N-1 trials, predict envelopes on the held-out trial in 2-s sliding windows, score window-level AAD by `r(reconstruction, attended) − r(reconstruction, competing) > 0`.
"""),

    ("code", """\
def windowed_correlation(pred, env, fs=EEG_FS_OUT, win_sec=WIN_SEC):
    win = int(round(win_sec * fs))
    n = len(pred) // win
    return np.array([np.corrcoef(pred[i*win:(i+1)*win], env[i*win:(i+1)*win])[0, 1] for i in range(n)])

def within_subject_aad(subj, trials):
    trial_packs = []
    for k in trials:
        try:
            trial_packs.append((k, *fetch_trial_aligned(subj, k)))
        except Exception:
            pass
    if len(trial_packs) < 2:
        return None
    accs, aucs = [], []
    for i in range(len(trial_packs)):
        train = [tp for j, tp in enumerate(trial_packs) if j != i]
        test  = trial_packs[i]
        Xtr = np.vstack([tp[1] for tp in train])
        ytr = np.concatenate([tp[2] for tp in train])
        sx = StandardScaler().fit(Xtr); Xtr = sx.transform(Xtr)
        sy = StandardScaler().fit(ytr.reshape(-1,1)); ytr = sy.transform(ytr.reshape(-1,1)).ravel()
        model = Ridge(alpha=RIDGE_ALPHA).fit(Xtr, ytr)
        Xte = sx.transform(test[1])
        pred = model.predict(Xte)
        # rescale to original env std for fair comparison
        env_a, env_c = test[2], test[3]
        r_a = windowed_correlation(pred, env_a)
        r_c = windowed_correlation(pred, env_c)
        decision = (r_a > r_c).astype(int)        # 1 = correct (attended chosen)
        accs.append(decision.mean())
        score = r_a - r_c
        if len(np.unique(decision)) == 2:
            aucs.append(roc_auc_score(np.ones_like(decision), score))   # always 1 in binary
    return {"acc_mean": float(np.mean(accs)), "n_folds": len(accs)}
"""),

    ("code", """\
within_results = {}
for subj in tqdm(SUBJECTS, desc="within-subject AAD"):
    r = within_subject_aad(subj, MAIN_KS)
    if r is not None:
        within_results[f"S{subj:02d}"] = r

within_df = pd.DataFrame(within_results).T.reset_index().rename(columns={"index": "subject"})
print(within_df.to_string(index=False))
"""),

    ("md", """\
## 4. LOSO AAD — cross-subject generalisation

Train one TRF on all trials of N-1 subjects, test on the held-out subject. Same window-level decision rule.
"""),

    ("code", """\
def loso_aad(SUBJECTS, MAIN_KS, max_train_trials=None):
    # Pre-fetch all trial packs once.
    cache = {}
    for subj in tqdm(SUBJECTS, desc="caching trials"):
        cache[subj] = []
        for k in MAIN_KS[: max_train_trials] if max_train_trials else MAIN_KS:
            try:
                cache[subj].append(fetch_trial_aligned(subj, k))
            except Exception:
                pass

    rows = []
    for held in SUBJECTS:
        X_tr_parts, y_tr_parts = [], []
        for subj, packs in cache.items():
            if subj == held: continue
            for X, env_a, _, _, _ in packs:
                X_tr_parts.append(X); y_tr_parts.append(env_a)
        if not X_tr_parts: continue
        X_tr = np.vstack(X_tr_parts); y_tr = np.concatenate(y_tr_parts)
        sx = StandardScaler().fit(X_tr); X_tr = sx.transform(X_tr)
        sy = StandardScaler().fit(y_tr.reshape(-1,1)); y_tr = sy.transform(y_tr.reshape(-1,1)).ravel()
        model = Ridge(alpha=RIDGE_ALPHA).fit(X_tr, y_tr)

        accs = []
        for X_te, env_a, env_c, _, _ in cache[held]:
            X_te_z = sx.transform(X_te)
            pred = model.predict(X_te_z)
            r_a = windowed_correlation(pred, env_a)
            r_c = windowed_correlation(pred, env_c)
            accs.append(float((r_a > r_c).mean()))
        if accs:
            rows.append({"held_out": f"S{held:02d}", "acc_mean": float(np.mean(accs)), "n_trials": len(accs)})
    return pd.DataFrame(rows)

loso_df = loso_aad(SUBJECTS, MAIN_KS)
print(loso_df.to_string(index=False))
"""),

    ("md", """\
## 5. Visualise both protocols
"""),

    ("code", """\
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)

axes[0].bar(within_df["subject"], within_df["acc_mean"], color="#2c7fb8", edgecolor="white")
axes[0].axhline(0.5, ls=":", color="#aaaaaa", label="chance")
axes[0].set(title="Within-subject AAD", ylabel=f"window-level accuracy ({WIN_SEC:.0f}-s windows)",
             ylim=(0.3, 1.0))
axes[0].legend(frameon=False, loc="lower right")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")

if not loso_df.empty:
    axes[1].bar(loso_df["held_out"], loso_df["acc_mean"], color="#cb6f7d", edgecolor="white")
    axes[1].axhline(0.5, ls=":", color="#aaaaaa", label="chance")
    axes[1].set(title="LOSO AAD (held-out subject)")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")
    axes[1].legend(frameon=False, loc="lower right")
plt.tight_layout(); plt.show()
"""),

    ("md", """\
## 6. Reporting

```python
print(f'Within-subject AAD: {within_df["acc_mean"].mean():.1%} ± {within_df["acc_mean"].std():.1%}')
if not loso_df.empty:
    print(f'LOSO AAD          : {loso_df["acc_mean"].mean():.1%} ± {loso_df["acc_mean"].std():.1%}')
```

The within-subject result is the standard "subject-specific decoder" benchmark; the LOSO result is the much harder zero-shot setting. Notebook 07 shows what gaze contributes on top of EEG.
"""),
]
