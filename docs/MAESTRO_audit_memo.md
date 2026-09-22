# MAESTRO Benchmark — Data-Leakage & Correctness Audit

**Target:** `github.com/ASPIRE-OSU/MAESTRO` — official benchmark code for the MAESTRO paper (Hassan, Alavi, Williamson).
**Version audited:** commit `e1048104148f40f79beba845a48edefbdc03e27a` (2026-07-21).
**Scope:** `scripts/*.py` (loader, models, all training/fusion/analysis scripts) + the dataset builder in this project (`hf_release/build/build_dataset.py`, `hf_release/maestro-loader-pkg/`).
**All `file:line` references below are relative to the MAESTRO repo root unless prefixed with `hf_release/`.**

---

## Verdict

The repo does **not** implement strict leakage prevention, and the headline validation numbers are **not reported correctly**. There are three independent problem classes:

1. **Evaluation methodology (universal):** every task reports the *best-epoch score on the same split used for early stopping and checkpoint selection* — there is no held-out test set. All headline numbers are selection-inflated.
2. **Split leakage (pooled family):** T1-pooled / T2 / T3 / T4 and their fusion/analysis reuse identical stimuli **and** identical subjects across train and test. Only LOSO is clean.
3. **Synchronization defect:** per-device audio onset stagger (~160 ms, in 100% of trials) is not compensated, so 2 of the 3 audio devices' speakers are misaligned to EEG.

The **loudness confound is handled correctly** (per-envelope z-score removes the +13 dB attended-is-loudest cue).

### Issue index

| # | Severity | Issue | Primary location |
|---|---|---|---|
| A1 | **Critical** | No held-out test set; reported metric = best-epoch on selection split | all `train_*.py` (see below) |
| A2 | **Critical** | LOSO selects/checkpoints on the reported test subject | `train_loso_hot.py:143-179, 283, 321-329` |
| A3 | High | "Best fusion" = max over 11 combos on the reported metric | `README.md:38`; `late_fusion.py:49, 311-318, 528` |
| A4 | High | Late-fusion combiner selected on eval fold; "(no leakage)" printout misleading | `late_fusion.py:196-231, 426` |
| B1 | **Critical** | Stimulus+label leakage: split over per-(subject×trial) counter, same stimuli in train & test | `dataloader.py:718, 746-750, 786-797` |
| B2 | High | Pooled family not subject-disjoint (same subjects in train & test) | `dataloader.py:706, 786-797` |
| B3 | Medium | LOSO `held_out_trial_frac` default mismatch across scripts → split can silently differ | `train_loso_hot.py:492` vs `late_fusion.py:495` |
| C1 | High | Per-device audio onset stagger not compensated (~160 ms, 100% of trials) | `dataloader.py:411, 541, 556-558` |
| C2 | Low | Shipped `bad_channels.csv` unused; bad channels re-detected heuristically | `dataloader.py:151-173`; `README.md:167` |
| — | ✅ correct | Loudness confound removed by per-envelope z-score | `dataloader.py:257-263` |

---

## A. Evaluation methodology — validation is not reported correctly

### A1 (Critical) — No held-out test set anywhere; reported number is the best epoch on the early-stopping split

Every training script uses a **2-way split (train + one held-out fold)** and uses that single held-out fold to (a) early-stop, (b) select the best-epoch checkpoint, and (c) produce the reported number. The returned value is literally `max_over_epochs(val_metric)`:

| Script | Best-epoch selection | Returned | Reported as |
|---|---|---|---|
| `train_pooled.py` | `:112-113` | `best_acc` `:120` | `val_accuracy` `:173`, mean `:177` |
| `train_hemisphere.py` | `:258-259` | `best_val_acc` `:269` | `val_accuracy` `:352`, `:358` |
| `train_eccentricity.py` | `:242-243` | `best_val_acc` `:253` | `val_accuracy` `:336`, `:342` |
| `train_reconstruction.py` | `:156-157` | `best_val_r` `:169` | `val_pearson_r` `:249`, `:256` |
| `train_loso_hot.py` | `:168-169` | `best_val_acc` `:179` | `test_accuracy` `:329`, `:338` |

There is **no third (test) split** in the codebase (a search for `test_idx`/`test_loader`/nested CV returns nothing). Selecting the best of ~50 epochs on the same data that is reported is upward-biased model selection. **Affects every number in `README.md` Results (§40-94).**

**Fix:** carve an inner validation split (from training subjects/content) for early stopping and checkpoint selection; report the selected checkpoint on data never inspected during selection. Use nested CV for the pooled tasks.

### A2 (Critical) — LOSO early-stops and checkpoints on the held-out test subject

`train_loso_hot.py` sets the validation loader to the held-out subject on held-out content (`val_mask` `:283`) and feeds it to `train_model` (`:321-325`), which selects the best epoch on it (`:168-170`) and returns that as `test_accuracy` (`:329`). The docstring is explicit: *"Val set = held-out subject on held-out trial content (true test set)"* (`:145`). So the reported test subject is used for model selection. With `held_out_trial_frac` this eval set is ~10 windows/subject → the best-epoch selection is both strongly biased and high-variance, and the reported ±std understates it.

### A3 (High) — "Best fusion" is a maximum over 11 combinations on the reported metric

`README.md:38` defines *"Best fusion"* as the *best-performing* of all 11 multi-modality combinations. `late_fusion.py` sweeps all 11 (`MULTI_MODALITY_MODES` `:49`, loop `:528`), each reported number itself already a best-epoch selection (A4). Reporting the max over 11 hypotheses evaluated on the same fold is multiple-comparison selection on the eval set. **Affects `README.md:48, 60, 71, 82` ("Best fusion" rows).**

### A4 (High) — Late-fusion combiner selected on the eval fold; "(no leakage)" is misleading

`train_combiner` trains the softmax weights on the train fold (models frozen — correct, `:188-192`) but then picks the **best-epoch weights by `val_acc`** (`:226-227`) and returns that as the result (`:231`, surfaced at `:311-318`, `:390`). The summary prints `"(no leakage)"` (`:426`), but the combiner is model-selected and reported on the same eval fold; for pooled tasks that fold also carries the B1 content leak.

---

## B. Split leakage

### B1 (Critical) — Stimulus + label leakage: pooled split is over (subject×trial), not trial content

`build_dataset` assigns a **fresh integer id per (subject, trial)** — `tid_ctr` increments inside the subject×trial loop (`dataloader.py:718, 746-750`), producing `trial_meta_ids` (`:769`). The same 100 stimuli are replayed to all 16 subjects, so each stimulus (e.g. `eval_014`) yields 16 distinct ids. `get_trial_level_splits` then runs `StratifiedKFold` over those ids with **no content or subject grouping** (`:786-797`). Result: subject A's `eval_014` can be in train while subject B's `eval_014` is in test — **identical audio waveforms in both folds**, and because the attended-speaker label is read per stimulus (`:724`) it is identical across subjects, so the *label* is duplicated too.

The decoder is match-mismatch with a **shared audio encoder over the candidate envelopes** (`model_classification.py:129-134, 172-175`; `model_spatial.py:131-136, 173-177`), so it can memorize stimulus-specific envelope features and carry them from train to test. This inflates all modes (EEG included, since every mode uses the audio keys).

**Consumers of the leaky split:** `train_pooled.py:149`; `train_hemisphere.py:307-310` (StratifiedKFold on `trial_meta_ids`); `train_eccentricity.py:291-294`; `train_reconstruction.py:215`; `late_fusion.py` (pooled/hemi/ecc tasks) `:281`; `analyze_snr.py:255`; `analyze_error.py:129`.

**LOSO is the exception** — it computes a trial-content hold-out once and shares it across folds (`train_loso_hot.py`: `split_trial_tids` `:184-214`, `is_train_tid`/`is_heldout_tid` `:270-271`, masks `:281-283`), so eval content is never trained on. The authors clearly knew this leak exists but did not apply the fix to the pooled family. **Affects `README.md:40-94` except the LOSO table (§52-61).**

**Fix:** split by trial-content id (`GroupKFold` on the original `eval_00N`), exactly as LOSO does.

### B2 (High) — Pooled family is not subject-disjoint

`build_dataset` pools all 16 subjects (`subj_list` `:706`) and `get_trial_level_splits` does not group by subject (`:786-797`), so each subject's trials are scattered across all folds → the model trains and tests on the same subjects (different trials). This is a disclosed "pooled" protocol, but combined with B1 it means the model sees **both the same subject and the same stimulus** in train and test. Only LOSO holds subjects out (`train_loso_hot.py:281`).

### B3 (Medium) — LOSO `held_out_trial_frac` default mismatch can silently break the guarantee

`run_loso_late_fusion` reconstructs the content split and *assumes it matches* the checkpoints' split (`late_fusion.py:346-351`). But the CLI defaults disagree: `train_loso_hot.py --held_out_trial_frac` defaults to **0.1** (`:492`) while `late_fusion.py` defaults to **0.2** (`:495`) — and the function/docstring say 0.2 (`train_loso_hot.py:226`). Run both with defaults and the fusion's "held-out" content includes trials the single-modality checkpoints were **trained on** → reintroduced leakage. The coupling is undocumented and unchecked.

**Fix:** persist the split (or the frac + seed) with the checkpoints and `assert` equality at fusion time.

---

## C. Synchronization / preprocessing correctness

### C1 (High) — Per-device audio onset stagger is not compensated

The 3 stereo devices do **not** start playback simultaneously. The loader uses a **single** `audio.t0_unix` for **all 6 speakers**: `_load_sync` returns `audio_t0_unix` (`dataloader.py:411`), and `load_trial` slices every speaker's waveform from it (`audio_t0 = sync["audio_t0_unix"]` `:541`; `i0 = round((anchor_unix - audio_t0)*sr)` `:556-558`; comment "ONE shared reference time … for every speaker" `:538`). `README.md:282` states this explicitly ("all four speakers' audio sharing one reference time regardless of recording device"). The HF loader does the same (`hf_release/maestro-loader-pkg/maestro_loader/align.py:205, 216`; `timing.py:59-60`).

The dataset builder collapsed the devices to `audio_t0 = min(playback_start)` (`hf_release/build/build_dataset.py:223`) and wrote **raw-demuxed FLACs with no per-device trim** (`:187`), so the stagger is not baked into the audio either. **Because `audio_t0 = min(ps)`, the earliest device is aligned and the other two are left shifted by their onset offset.**

**Measured** across all 1600 eval `audio_timestamps.json`: inter-device onset spread **mean 163 ms, median 163 ms, max 245 ms, > 100 ms in 100% of trials** (all 3 devices present in every trial). That is on the order of the neural tracking lag itself, so it corrupts genuine EEG↔envelope alignment for **speakers 3-6** (Devices 2 & 3), which are the keys in every task.

Contrast this project's own `src/data/windows.py:build_trial`, which slices *each* device from its own onset (`o0 = round((anchor - ps[di])*sr)` `:317`) — exactly the correction the release drops.

**Recoverable without a rebuild:** the per-device onset **is preserved** in the timing JSON (`hf_release/build/build_dataset.py:260-265`, `audio.devices[].playback_start_unix`); the loaders simply never read it. Fix `load_trial` to slice speaker *n* from its device's own onset:

```python
# device for speaker n (1-based): (n-1)//2 ; use that device's playback_start_unix
dev_t0 = sync["audio_devices"][(spk_n - 1)//2]["playback_start_unix"]
i0 = max(0, int(round((anchor_unix - dev_t0) * sr)))
i1 = int(round((anchor_unix + trial_end_sec - dev_t0) * sr))
```
(and add `audio.devices` to `_load_sync`'s return, `dataloader.py:402-412`).

### C2 (Low) — Shipped `bad_channels.csv` is unused

`README.md:167` ships a curated per-subject `bad_channels.csv`, and `README.md:276` describes "bad-channel detection", but `dataloader.py` re-derives bad channels heuristically per trial in `_detect_bad_channels` (`:151-173`, called `:202`) and never reads the CSV. The heuristic may disagree with the curated list (e.g. the known dead M1 / saturated M2 on Subject 1). Low severity, but the released metadata does not match what the loader uses.

---

## What the repo gets right (do not change)

- **Loudness confound removed.** Each speaker's envelope is z-scored independently (`extract_envelope` `:257-263` → `_zscore` `:129-133`). Since playback loudness is a scalar gain and the Hilbert envelope of `g·x` is `g·env(x)`, z-score cancels `g` exactly. **Verified on 60 released trials:** attended is the loudest candidate in 60/60 (mean **+13.45 dB**, range +5.8…+17.6 dB) in the raw envelopes, and **+0.000 dB** after the release's own `extract_envelope`. Cosine similarity is additionally scale-invariant (`model_classification.py:136-137`), and the hemisphere/eccentricity sums of two unit-variance envelopes don't reintroduce level.
- **No global-scaler leak.** All z-scoring is per-trial, per-modality, from within-trial statistics (`:129-133`, applied at EEG `:236`, envelope `:263`, gaze `:337`, IMU `:372`, video `:305`).
- **Per-trial EEG referencing / bad-channel / interpolation** — no cross-trial information (`filter_reference_eeg:178-236`).
- **~1 window per trial** (35 s trial, 30 s non-overlapping window; `:655-658`) → no within-trial adjacent-window leakage; trial-level and window-level splits coincide.
- **LOSO holds out subject AND content** — the correct design (`train_loso_hot.py:281-283`); it is only spoiled by A2's selection bias.
- **Reproducibility fix** — per-mode/per-fold seeding prevents architecturally-identical modes from colliding (`_mode_seed`, e.g. `train_pooled.py:23-39`).
- **Frozen single-modality models** during fusion (`late_fusion.py:188-192`).

---

## Recommended fixes (priority order)

1. **Add a real test set** (A1, A2): nested CV for pooled; an inner validation split from the training subjects for LOSO early stopping. Report the selected checkpoint on held-out data.
2. **Content-disjoint splits** (B1, B2): replace `get_trial_level_splits` with `GroupKFold` on trial-content id for `train_pooled/hemisphere/eccentricity/reconstruction` and `analyze_snr/analyze_error`.
3. **Fix late-fusion reporting** (A3, A4): select the combiner and the "best fusion" combination on an inner split; report on the outer test.
4. **Fix per-device audio alignment** (C1): read `audio.devices[].playback_start_unix` and slice each speaker from its own device onset; re-run all tasks.
5. **Lock the LOSO split** (B3): persist frac+seed with checkpoints and assert at fusion time.
6. **Use the shipped `bad_channels.csv`** (C2), or drop it from the release and document the heuristic.

**Expected effect:** the pooled EEG number (58.6%) should fall toward — or below — the LOSO EEG number (56.9%) once B1 and A1 are fixed, and any EEG envelope-tracking claim (T4, EEG T1) should be re-measured after C1, consistent with the finding that EEG envelope/stimulus tracking is weak-but-present while the strong signal is spatial/overt-orienting.

---

## Appendix — reproduction

- **Inter-device stagger (C1):** over all 1600 `experiment_data/Subject*/Eval-*/audio_timestamps.json`, spread = `max(playback_start_time) − min(playback_start_time)` → mean/median ≈ 0.163 s, max 0.245 s, > 0.1 s in 100% of trials.
- **Loudness (correct):** for each released trial, per-speaker Hilbert-envelope RMS (raw) vs. after `dataloader.extract_envelope` → raw attended-vs-others +13.45 dB (attended loudest in 60/60), z-scored +0.000 dB.
