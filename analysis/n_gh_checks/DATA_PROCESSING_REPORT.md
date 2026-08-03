# What was wrong with the MAESTRO repo's data processing — and how ours differs

Short comparison of the public `ASPIRE-OSU/MAESTRO` benchmark's train/val data
pipeline against the leakage-safe pipeline in `n_gh_checks` (`gh_data.py` →
`src/data`). Verified against the repo source and reproduced empirically.

## The problems (repo)

**1. No held-out test set — the reported number is selected on the data it reports.**
Every train script (`train_pooled/hemisphere/eccentricity/reconstruction`,
`train_loso_hot`) splits into train + one held-out fold, early-stops **and**
picks the checkpoint on that fold, then reports `max_over_epochs(val_acc)` on the
**same** fold. `val == test`. Taking the best epoch on the evaluation set inflates
every number.

**2. Subjects are not held out (pooled family) — the biggest single inflation.**
Hemisphere/eccentricity/4-class use `StratifiedKFold` over per-`(subject,trial)`
instances with **no subject grouping**, so the same subjects sit in train and
test. Because gaze/IMU are **uncalibrated** (subject-relative), the model gets to
see each test subject's calibration during training — the entire reason gaze
"decodes" hemisphere at 78%. Only LOSO is subject-disjoint.

**3. Stimulus + label leakage (pooled family).** The same 100 stimuli replay to
all 16 subjects; K-folding over the ~1600 `(subject,trial)` rows puts **identical
audio** — and the identical attended label (attended is deterministic,
`((trial-1)%4)+1`) — in train and test. The shared audio encoder can memorize
stimulus identity.

**4. 30 s non-overlapping windows → ~1 decision window per trial.** Very little
data per fold, which amplifies the selection bias in (1).

**5. Attended-loudness confound only partly controlled.** The attended speaker is
+3–18 dB louder in 100% of trials. The repo z-scores each envelope (which helps)
but does not equalize loudness at source, leaving a residual energy cue for the
4-class task.

**6. Per-device audio lag not compensated.** The released build writes
raw-demuxed FLACs with a single `audio.t0` for all 6 speakers, but the 3 devices
start ~163 ms apart (max 245 ms, >100 ms in 100% of trials) — comparable to the
cortical tracking lag itself, so EEG↔envelope alignment is corrupted for speakers
on the two later devices.

**7. Misc.** `held_out_trial_frac` defaults disagree across scripts (0.1 vs 0.2),
so LOSO late-fusion can load mismatched checkpoints; and there is no
overt-orienting / motion control, though gaze/IMU (~60–78%) is largely orienting.

## How ours differs (the fixes)

| # | repo | ours (`n_gh_checks` / `src.data`) |
|---|---|---|
| 1 | val == test, best-epoch-on-eval | **inner-val** carved from train for early stop/selection → **single held-out test** eval |
| 2 | subjects pooled into train+test | **subject-disjoint LOSO**; within-subject uses **chrono-forward** folds (train precedes test) |
| 3 | same stimuli/labels in train+test | **trial-level splits** (no window straddles the boundary); LOSO holds out subject *and* content |
| 4 | 30 s, no overlap (~1 win/trial) | **5 s @ 0.5 overlap** (~11 win/trial) |
| 5 | z-score only | **loudness-matched at source** (`table_power`: divide each channel by √documented presented power) **+ per-window-permuted** 4-class candidates → no slot→direction shortcut |
| 6 | single `audio.t0` for all speakers | **per-device alignment**: each FLAC sliced from its own `playback_start`; anchor = `max(eeg_start, max(playback_start))` |
| — | — | full-band EEG kept for spatial tasks (alpha/beta lateralisation preserved) |

## Bottom line — the inflation decomposes into two roughly equal halves

Running the same code with `--data-method github` vs `proper` (holding the
controlled data fixed) isolates the *methodology* leak; the residual gap to the
repo's published numbers is the *data-pipeline* difference. EEG, chance in ():

| task (chance) | **proper LOSO** (honest) | **github method, our data** (+ leaky methodology) | **repo reported** (+ uncontrolled data) |
|---|---|---|---|
| hemisphere (.50) | **0.56** | 0.67 | 0.78 |
| eccentricity (.50) | **0.56** | 0.64 | 0.69 |
| 4-class speaker (.25) | **0.36** | 0.45–0.50 | 0.58 |
| reconstruction r | ~0.00 | ~0.00 | ~0.00 |

- **Leaky methodology** (subject-pooling + best-epoch-on-eval + 30 s windows) adds
  ~0.08–0.12 on our controlled data — reproduced empirically, not just argued.
- **Uncontrolled data** (their EEG preprocessing, non-loudness-matched envelopes,
  single-`t0` audio) adds the remaining ~0.09–0.11 to reach the repo's numbers.
- **Honest signal** above chance is small (~0.06 hemisphere, ~0.11 4-class).

The single biggest lever is **subject-disjointness**: with uncalibrated gaze/IMU,
pooling the same subjects into train and test is what manufactured most of the
reported multimodal accuracy.
