"""Notebook 01 — Dataset overview."""

CELLS = [
    ("md", """\
# Notebook 01 — Dataset overview

A cohort-level summary of the 16-participant, 105-trial maestro-eeg-dataset. We characterise:

1. **Demographics** — age, gender, handedness, ear preference, race/ethnicity.
2. **Trial structure** — five training + 100 main trials, four attendable speakers per trial.
3. **Modality coverage** — which (subject, trial) pairs have which modalities, plus any gaps.
4. **Behavioural performance** — per-subject comprehension accuracy.
5. **Bad-channel inventory** — automatically flagged EEG channels.

All figures and tables in this notebook are reproducible from the dataset alone — no extra files needed.
"""),

    ("code", """\
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from huggingface_hub import hf_hub_download

sns.set_context("paper", font_scale=1.05); sns.set_style("ticks")
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})

REPO_ID_FULL   = "aspire-osu/maestro-eeg-dataset"
REPO_ID_SAMPLE = "aspire-osu/maestro-eeg-dataset-sample"
USE_SAMPLE     = True
REPO_ID = REPO_ID_SAMPLE if USE_SAMPLE else REPO_ID_FULL

def load_csv(name): return pd.read_csv(hf_hub_download(REPO_ID, f"metadata/{name}", repo_type="dataset"))

trials      = load_csv("trials.csv")
per_subject = load_csv("trials_per_subject.csv")
subjects    = load_csv("subjects.csv")
bad_chans   = load_csv("bad_channels.csv")
print(f"trials.csv:            {len(trials):4d} rows")
print(f"trials_per_subject.csv:{len(per_subject):4d} rows")
print(f"subjects.csv:          {len(subjects):4d} rows")
print(f"bad_channels.csv:      {len(bad_chans):4d} rows")
"""),

    ("md", """\
## 1. Demographics

The cohort is recruited from the OSU undergraduate and graduate population. Ages 18–30, balanced for gender, predominantly right-handed.
"""),

    ("code", """\
def safe_int(x):
    try: return int(x)
    except (TypeError, ValueError): return np.nan

subj = subjects.copy()
subj["age"] = subj["age"].map(safe_int)

print(subj[["subject_id", "age", "gender", "handedness", "ear_preference", "race_ethnicity"]].to_string(index=False))
"""),

    ("code", """\
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))

axes[0].hist(subj["age"].dropna(), bins=np.arange(17, 32), edgecolor="white", color="#2c7fb8")
axes[0].set(xlabel="Age (years)", ylabel="# subjects", title="Age distribution")

subj["gender"].value_counts().plot.barh(ax=axes[1], color=["#fa9fb5", "#74c476", "#cccccc"])
axes[1].set(xlabel="# subjects", title="Gender")

subj["handedness"].value_counts().plot.barh(ax=axes[2], color=["#9ecae1", "#fdae6b", "#cccccc"])
axes[2].set(xlabel="# subjects", title="Handedness")

plt.tight_layout(); plt.show()
"""),

    ("md", """\
## 2. Trial structure

Each subject completes 5 training (familiarisation) trials and 100 main (evaluation) trials. Each trial presents 6 simultaneous speech streams from 3 stereo devices. Speakers 1–4 are attendable targets; 5 and 6 are fixed distractors.
"""),

    ("code", """\
print("Trial breakdown:")
print(trials["kind"].value_counts().to_string())
print()
print("Attended speaker frequency across all trials:")
print(trials["attended_speaker"].value_counts().sort_index().to_string())
print()
print("Per-trial SNR distribution (dB):")
print(trials.groupby("kind")["snr_db"].describe()[["count","mean","std","min","max"]].round(1).to_string())
"""),

    ("code", """\
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
trials["attended_speaker"].value_counts().sort_index().plot.bar(ax=axes[0], color="#54a888")
axes[0].set(xlabel="Speaker (1–4 attendable)", ylabel="# trials", title="Attended-speaker balance")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

axes[1].hist(trials.loc[trials["kind"] == "main", "snr_db"], bins=15, color="#cb6f7d", edgecolor="white")
axes[1].set(xlabel="SNR (dB)", ylabel="# main trials", title="Per-trial SNR distribution")
plt.tight_layout(); plt.show()
"""),

    ("md", """\
## 3. Modality coverage

Every (subject, trial) cell should have EEG, gaze, IMU, audio, video, and timing. Any blanks are real source-data gaps and surfaced explicitly.
"""),

    ("code", """\
from huggingface_hub import HfApi
api = HfApi()
files = api.list_repo_files(REPO_ID, repo_type="dataset")

import re
cov = {}
for f in files:
    if not f.endswith((".parquet", ".mp4", ".flac", ".json")):
        continue
    parts = f.split("/")
    if len(parts) < 2: continue
    if parts[0] == "data" and len(parts) >= 4:           # data/{mod}/subject=SXX/trial=...parquet
        mod = parts[1]
    elif parts[0] == "media" and parts[1] in ("video", "timing", "audio"):
        mod = parts[1]
    else:
        continue
    cov[mod] = cov.get(mod, 0) + 1

print("Files per modality (current repo):")
for k in ("eeg","gaze","imu","audio","video","timing"):
    print(f"  {k:8s}  {cov.get(k, 0)}")
"""),

    ("md", """\
## 4. Behavioural performance

After each trial, participants answer a 4-AFC comprehension question about the **attended** speech. Above-chance accuracy (chance = 25%) confirms participants attended as instructed.
"""),

    ("code", """\
ps = per_subject.copy()
ps["correct"] = ps["comprehension_correct"].fillna(False).astype(int)

per_subj_acc = (ps.loc[ps["kind"] == "main"]
                  .groupby("subject_id")["correct"].mean()
                  .sort_values(ascending=False) * 100)

cohort_mean = per_subj_acc.mean()
cohort_se   = per_subj_acc.std(ddof=1) / np.sqrt(len(per_subj_acc))

fig, ax = plt.subplots(figsize=(8, 3.6))
ax.bar(per_subj_acc.index, per_subj_acc.values, color="#4c87b3", edgecolor="white")
ax.axhline(25, ls=":", color="#aaaaaa", label="chance (25%)")
ax.axhline(cohort_mean, ls="--", color="crimson", label=f"cohort mean ({cohort_mean:.1f}% ± {cohort_se:.1f})")
ax.set(xlabel="Subject", ylabel="Comprehension accuracy (%)", title="Per-subject behavioural performance (main trials)")
ax.set_xticklabels(per_subj_acc.index, rotation=45, ha="right")
ax.legend(frameon=False, loc="lower right"); plt.tight_layout(); plt.show()
"""),

    ("md", """\
## 5. Bad-channel inventory

Auto-detected from each subject's first available trial via ADC saturation (>5% samples at the channel's own |max|) and zero-variance checks. Mastoid channels (M1, M2) are the dominant failure mode.
"""),

    ("code", """\
bc = bad_chans.copy()
bc["bad_channels"] = bc["bad_channels"].fillna("")
bc["n_bad"]        = bc["bad_channels"].apply(lambda s: 0 if not s else len(s.split(";")))
print(bc[["subject_id", "bad_channels", "n_bad"]].to_string(index=False))

if bc["n_bad"].sum() > 0:
    flat = (bc["bad_channels"].str.split(";").explode().replace("", np.nan).dropna().value_counts())
    fig, ax = plt.subplots(figsize=(6, 2.8))
    flat.plot.bar(ax=ax, color="#e8743b", edgecolor="white")
    ax.set(xlabel="Channel", ylabel="# subjects flagged",
           title="Bad-channel frequency across cohort")
    plt.tight_layout(); plt.show()
"""),

    ("md", """\
## 6. Summary

* **n = 16** subjects (full dataset; sample has 3), ages 18–30, predominantly right-handed.
* **100 main + 5 training trials** per subject; balanced attended-speaker assignment over speakers 1–4.
* All five modalities present for nearly every (subject, trial) cell; the loader exposes `missing=True` for any gaps.
* Cohort comprehension accuracy is well above chance (25%), validating the attention paradigm.
* The dominant data-quality issue is mastoid clipping (M1/M2) — handled by the loader's `bad_channels=` policy.

The next notebook digs into EEG signal quality directly from the parquet shards.
"""),
]
