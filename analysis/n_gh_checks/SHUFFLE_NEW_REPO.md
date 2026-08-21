# Re-audit of the updated MAESTRO repo (2026-08-12) — is the leakage fixed?

The public `github.com/ASPIRE-OSU/MAESTRO` repo was rewritten on **2026-08-12**
(HEAD `4d4c505`), after our July–August audit. We pulled it fresh, diffed it, and
ran our EEG-shuffle null **on their own committed checkpoints** to answer one
question: *is the headline four-class number now neural?*

## What they fixed (real, and it matches our critiques)

The entire data + training layer was rewritten (`train_aad.py` is new; `dataloader.py`
+332 lines). These are genuine improvements:

1. **Real held-out test.** `train_model` early-stops on an inner-val split carved from
   train and returns `best_inner_val` ("NOT a final reportable number"); `evaluate_test`
   is a **one-time** eval of the selected checkpoint on the official test split. Kills the
   old *val == test, max-over-epochs* leak.
2. **Content-disjoint LOSO.** For LOSO they now layer a **global content holdout** on top
   of the subject split: the held-out subject's test is restricted to held-out trial
   content, and the 15 training subjects are restricted to non-held-out content
   (`train_aad.py:182–214`). Kills the "same stimuli in train and test" leak.
3. **RMS loudness-equalization** across the trial's 4 speakers before enveloping
   (`dataloader.py:241–246, 529–535`). Kills the raw-energy shortcut.
4. **Per-window candidate permutation** with the label following the permutation
   (`dataloader.py:946–950`). Kills the fixed-slot → direction shortcut.

## What they did NOT fix (the decisive flaw)

- The **model is architecturally identical** — the only diff in `model_classification.py`
  / `model_spatial.py` is a docstring rename (`git diff 18e0b16^..18e0b16`). `AADModel`
  still runs each candidate envelope through a learned per-candidate `audio_encoder`
  (`in_channels=1`) + learned `sim_proj` similarity head (`model_classification.py:128–176`)
  — the same **audio → label path** we adjudicated as non-neural.
- **No EEG-shuffle null anywhere in the repo** (grep is clean). The one control that
  exposes the audio path is still absent.

## The test — EEG-shuffle null on their committed LOSO checkpoints

`shuffle_new_repo.py` imports **their** `dataloader.py` + `model_classification.py`,
rebuilds each LOSO fold's test set **identically** to `train_aad.run_official_splits`
(official subject split + global content holdout, seed 42), loads **their** committed
checkpoint `results/results_aad_loso_*/fold_K_eeg_loso.pt`, reproduces the reported
accuracy, then re-runs with the **EEG permuted across the fold's test windows** (audio
candidates + labels held fixed, averaged over 20 permutations). Eval-only, no retraining.
Checkpoints load with `strict=True` — "all keys matched" — so these are exactly the nets
that produced their README numbers.

### The shuffle
Each window's brain recording (EEG / gaze / IMU / video) is swapped for **another
window's** recording — a permutation of whole examples within the fold's test set, *not*
a channel scramble. Every example stays an intact real recording; only the brain↔trial
correspondence is broken. The **audio candidates are always fed and never changed** — window
*i* always keeps its own 4 real candidate envelopes (with their deterministic per-window
speaker permutation). So the only difference between "real" and "null" is which brain sits
next to the audio. Null averaged over 20 permutations.

### Result — the T1 number is **not neural** for ANY modality or protocol (real == null)

Full sweep: **4 modalities × 2 official protocols × 5 decision windows = 40 configs**, all
on their committed checkpoints. We reproduce their reported number to within 0.003
everywhere (so this is their exact pipeline), and the shuffle null **equals** it every time.

**LOSO (chance 0.25):**

| modality | metric | 5 s | 10 s | 15 s | 20 s | 30 s |
|---|---|---|---|---|---|---|
| **eeg**   | real (= their reported) | 0.4970 | 0.4463 | 0.4593 | 0.5063 | 0.5188 |
|           | shuffle null            | 0.4971 | 0.4458 | 0.4586 | 0.5063 | 0.5188 |
|           | margin                  | −0.0002 | +0.0005 | +0.0006 | +0.0000 | +0.0000 |
| **gaze**  | real                    | 0.5004 | 0.4919 | 0.4755 | 0.4985 | 0.5390 |
|           | shuffle null            | 0.5004 | 0.4925 | 0.4764 | 0.4993 | 0.5396 |
|           | margin                  | −0.0001 | −0.0006 | −0.0009 | −0.0008 | −0.0006 |
| **imu**   | real                    | 0.4992 | 0.4618 | 0.4852 | 0.5237 | 0.5238 |
|           | shuffle null            | 0.4992 | 0.4618 | 0.4852 | 0.5245 | 0.5253 |
|           | margin                  | +0.0000 | +0.0000 | +0.0000 | −0.0008 | −0.0014 |
| **video** | real                    | 0.5045 | 0.4683 | 0.4865 | 0.5156 | 0.5156 |
|           | shuffle null            | 0.5045 | 0.4718 | 0.4868 | 0.5152 | 0.5164 |
|           | margin                  | +0.0000 | −0.0035 | −0.0003 | +0.0005 | −0.0008 |

**Within-subject (chance 0.25):**

| modality | metric | 5 s | 10 s | 15 s | 20 s | 30 s |
|---|---|---|---|---|---|---|
| **eeg**   | real (= their reported) | 0.4757 | 0.4754 | 0.4894 | 0.5287 | 0.5306 |
|           | shuffle null            | 0.4757 | 0.4755 | 0.4890 | 0.5288 | 0.5306 |
|           | margin                  | −0.0000 | −0.0001 | +0.0004 | −0.0000 | −0.0000 |
| **gaze**  | real                    | 0.4763 | 0.4854 | 0.5031 | 0.5174 | 0.5387 |
|           | shuffle null            | 0.4763 | 0.4855 | 0.5035 | 0.5166 | 0.5397 |
|           | margin                  | +0.0000 | −0.0001 | −0.0004 | +0.0008 | −0.0010 |
| **imu**   | real                    | 0.4728 | 0.4985 | 0.4901 | 0.4805 | 0.5036 |
|           | shuffle null            | 0.4728 | 0.4983 | 0.4901 | 0.4805 | 0.5032 |
|           | margin                  | +0.0001 | +0.0002 | −0.0000 | −0.0000 | +0.0005 |
| **video** | real                    | 0.4900 | 0.4899 | 0.4771 | 0.4313 | 0.4550 |
|           | shuffle null            | 0.4900 | 0.4898 | 0.4772 | 0.4306 | 0.4550 |
|           | margin                  | −0.0000 | +0.0000 | −0.0002 | +0.0006 | −0.0000 |

- **max |margin| across all 40 configs = 0.0035; mean |margin| = 0.0004.** Not a single
  configuration shows a significant real-vs-null gap; several have the null fractionally
  *above* real. Permuting the brain modality does not change the decision.
- This holds for **all four modalities**. Every single-modality checkpoint (eeg/gaze/imu/
  video) shares the same learned `audio_encoder`, and that shared audio path is doing 100 %
  of the work — the brain/behavior branch is inert. It is the empirical version of their own
  table's tell: EEG ≈ gaze ≈ IMU ≈ video ≈ 50 %, because none of them matter.
- RMS loudness-equalization removes *energy* but not the envelope-**shape** acoustic marking
  of the attended talker, which the learned audio encoder reads regardless of the brain
  input — exactly as `TRUTHFUL_BASELINE.md` predicted (content-disjoint + loudness-match does
  not drop the null).

*Data note:* the HF release is missing gaze/IMU parquet for S02/eval_077 & eval_078 (2 of
1600 trials); those windows are skipped in the gaze/imu builds and do not affect the result.

## Verdict

**Partially fixed.** The split/evaluation leakage is genuinely fixed (held-out test,
content-disjoint LOSO, loudness equalization, candidate permutation) — real methodological
progress that pulled the number from ~0.58 to ~0.50. But the **model architecture is
unchanged and they never added the null**, so the residual ~0.50 four-class result is the
**acoustic-marking / audio-only floor, not brain decoding** — empirically real == null on
their own checkpoints at both window sizes. This is the exact lesson of `ADJUDICATION.md`:
strict subject/content-disjoint splits are **necessary but not sufficient**; a model with a
learned stimulus encoder converts an admissible split back into an inadmissible result, and
only the EEG-shuffle null catches it.

Data: `results/shuffle_new/eeg_loso_{w5_h2.5,w10_h5}.json`. Jobs 6871504 / 6871509.
