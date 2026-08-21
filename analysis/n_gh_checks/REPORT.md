# Re-audit of the updated MAESTRO repo — experiments & results

**Repo:** `github.com/ASPIRE-OSU/MAESTRO`, HEAD `4d4c505` (rewritten 2026-08-12, after our
July–Aug audit) · **Inputs:** their own committed checkpoints + locally-mirrored HF dataset ·
**All experiments are eval-only, no retraining.**

**Question:** the repo was updated — is the leakage fixed? If the T1 "attended-source
decoding" number is still non-neural, *why*, and is that finding an artifact?

---

## E1 — What changed in the code

Diffed the new tree against the version we previously audited. The entire data/training
layer was rewritten, but the **model is byte-identical** (only a docstring was renamed).
Below: which of our original critiques were addressed.

| Concern (from our audit) | Status | Where |
|---|---|---|
| val == test, report max-over-epochs | **fixed** — inner-val early stop → one-time test eval | `train_aad.py:104,135` |
| LOSO shares stimulus content across subjects | **fixed** — global content holdout on the subject split | `train_aad.py:182–214` |
| loudness / raw-energy shortcut | **fixed (nominally)** — RMS-equalize the 4 speakers | `dataloader.py:241–246` |
| fixed slot → direction shortcut | **fixed** — per-window candidate permutation | `dataloader.py:946–950` |
| learned audio→label path (audio encoder + sim head) | **NOT fixed** — architecture unchanged | `model_classification.py:128–176` |
| no brain-shuffle null control | **NOT fixed** — absent from the repo | (grep clean) |

---

## E2 — Brain-shuffle null on their checkpoints (`shuffle_new_repo.py`)

Rebuilt each fold's test set identically to their `train_aad.run_official_splits`, loaded
their committed checkpoint (`strict=True`, all keys matched), reproduced their reported
accuracy, then re-ran with the **brain modality permuted across test windows** — each window
keeps its own audio candidates + label, only the brain that sits next to them is shuffled
(20 permutations). **4 modalities × 2 protocols × 5 decision windows = 40 configs.**

**LOSO (chance 0.25) — real (= their reported) vs brain-shuffle null:**

| modality | metric | 5 s | 10 s | 15 s | 20 s | 30 s |
|---|---|---|---|---|---|---|
| eeg | real | 0.4970 | 0.4463 | 0.4593 | 0.5063 | 0.5188 |
| | null | 0.4971 | 0.4458 | 0.4586 | 0.5063 | 0.5188 |
| gaze | real | 0.5004 | 0.4919 | 0.4755 | 0.4985 | 0.5390 |
| | null | 0.5004 | 0.4925 | 0.4764 | 0.4993 | 0.5396 |
| imu | real | 0.4992 | 0.4618 | 0.4852 | 0.5237 | 0.5238 |
| | null | 0.4992 | 0.4618 | 0.4852 | 0.5245 | 0.5253 |
| video | real | 0.5045 | 0.4683 | 0.4865 | 0.5156 | 0.5156 |
| | null | 0.5045 | 0.4718 | 0.4868 | 0.5152 | 0.5164 |

**Within-subject (chance 0.25):**

| modality | metric | 5 s | 10 s | 15 s | 20 s | 30 s |
|---|---|---|---|---|---|---|
| eeg | real | 0.4757 | 0.4754 | 0.4894 | 0.5287 | 0.5306 |
| | null | 0.4757 | 0.4755 | 0.4890 | 0.5288 | 0.5306 |
| gaze | real | 0.4763 | 0.4854 | 0.5031 | 0.5174 | 0.5387 |
| | null | 0.4763 | 0.4855 | 0.5035 | 0.5166 | 0.5397 |
| imu | real | 0.4728 | 0.4985 | 0.4901 | 0.4805 | 0.5036 |
| | null | 0.4728 | 0.4983 | 0.4901 | 0.4805 | 0.5032 |
| video | real | 0.4900 | 0.4899 | 0.4771 | 0.4313 | 0.4550 |
| | null | 0.4900 | 0.4898 | 0.4772 | 0.4306 | 0.4550 |

**Summary:** max |margin| across all 40 configs = **0.0035**, mean |margin| = **0.0004**, no
significant gap; real reproduces their reported number to within 0.003. → the decision is
100 % from the candidate audio; the brain/behaviour modality is unused.

---

## E3 — Root cause of the audio-only ~0.50 (`audio_rootcause.py` + metadata)

Two hypotheses: (a) train/test audio overlap, (b) acoustic marking of the attended talker.
Checked the manifest for overlap, then built the exact z-scored candidate envelopes the
model sees and tested whether a level-free shape probe can pick the attended talker on
content-disjoint folds.

**(a) Overlap — refuted:**

| check | value |
|---|---|
| distinct attendable voices across 100 trials (400 slots) | 400 (0 reuse) |
| mean occurrences per voice | 1.00 |
| voices shared between train and test (fold 0) | 0 |

**(b) Loudness confound + who can decode it:**

| decoder | 4-way acc (chance 0.25) |
|---|---|
| attended == loudest of 4 (frequency over 100 trials) | 100 / 100 (median +15.1 dB) |
| loudness oracle (argmax raw RMS) | **1.0000** |
| **shape-only linear probe** (z-scored envelope, content-disjoint 5-fold) | **0.6500** |
| deep `AADModel` audio-only floor (their number) | ~0.50 |

**Which shape feature carries it (level-free, on z-scored envelope):**

| feature | AUC (att vs not) | direction | corr with loudness |
|---|---|---|---|
| kurtosis | 0.190 | attended lower | −0.27 |
| skew | 0.219 | attended lower | −0.46 |
| Gini sparsity | 0.297 | attended lower | −0.43 |
| dynamic range p95−p5 | 0.619 | attended higher | +0.22 |
| HF 8–20 Hz power | 0.401 | attended lower | −0.19 |

**Summary:** no overlap; the attended talker is systematically **louder (+15 dB)**, and
although z-scoring removes level, loudness left a **content-transferable envelope-shape
fingerprint** (louder ⇒ cleaner ⇒ deeper, less-peaky modulation) that a trivial linear probe
reads at 0.65 — *above* the deep model. It is a property of the target *role*, so no
subject/content split removes it.

---

## E4 — Ruling out shuffle-artifacts (`shuffle_diagnostics.py` + `raw_probe.py`)

Tested whether real == null could be an artifact: collapsed embeddings, a too-gentle shuffle
(same participant / embeddings cluster by subject), or preprocessing having destroyed the
neural signal. Brain manipulated at the **input** (fed through their encoder).

**(a) Are the brain embeddings collapsed?** `brain_enc` mean pairwise cosine across windows:

| modality | mean pairwise cosine (1.0 = collapsed) | norm CV |
|---|---|---|
| eeg | 0.9995 | 0.04 |
| gaze | 0.9963 | 0.05 |
| imu | 1.0000 | 0.01 |
| video | 0.9995 | 0.04 |

**(b) Input-ablation battery — LOSO 4-way accuracy (chance 0.25):**

| modality | real | within-shuffle | constant (mean) | zeros | Gaussian noise |
|---|---|---|---|---|---|
| eeg | 0.4969 | 0.4962 | 0.4969 | 0.4969 | 0.4965 |
| gaze | 0.5149 | 0.5153 | 0.5156 | 0.5156 | 0.5149 |
| imu | 0.5112 | 0.5112 | 0.5112 | 0.5112 | 0.5112 |
| video | 0.5162 | 0.5166 | 0.5156 | 0.5156 | 0.5169 |

**(c) Cross-subject shuffle + decision-flip (within fold 0, EEG):**

| input | 4-way acc | argmax flip rate vs real |
|---|---|---|
| real | 0.4000 | — |
| within-shuffle | 0.4004 | 0.0004 |
| cross-subject shuffle | 0.4004 | 0.0004 |
| constant / zeros / noise | 0.4000 | 0.000 |

(flip rate ≤ 0.002 for every modality and every manipulation, both protocols)

**(d) Did preprocessing discard the signal? — positive control (`raw_probe.py`, linear probe, trial-disjoint):**

| decode target | from RAW preprocessed EEG | from model's `brain_enc` | chance |
|---|---|---|---|
| subject identity (16-way) | **0.900** | 0.101 | 0.0625 |
| attended speaker (4-way) | **0.337** | brain-path margin 0.000 | 0.25 |

**Summary:** embeddings are collapsed **and** zeros/noise/cross-subject give identical
accuracy (so it is not participant-clustering or a gentle shuffle) **and** the decision never
flips **and** the raw EEG still carries 90 %-decodable subject identity and above-chance
attended-speaker info — the *encoder* collapsed it, preprocessing did not. A trivial linear
probe reads more neural attention (0.337) than the whole deep model's brain path (0.000).

---

## Conclusion

The update fixed the **split/evaluation leakage** (pulling the number from ~0.58 to ~0.50) but
**not the core flaw**. The T1 ~0.50 four-way is a **shortcut / Clever-Hans** result: the
attended talker is systematically louder (+15 dB), imprinting a content-transferable
envelope-shape fingerprint that the learned audio encoder reads directly; the model trained
itself to ignore the brain (and gaze/IMU/video) entirely. It is not brain decoding, not
train/test overlap, and not a preprocessing/shuffle artifact — only the brain-shuffle null
exposes it, and their pipeline never runs one.

**Fix for a valid benchmark:** equalize candidate envelopes in *distribution/shape* per trial
(rank/histogram matching or matched modulation), not just level; report every model against
its brain-shuffle null.

**Detail & data:** `SHUFFLE_NEW_REPO.md` (E1–E2) · `ROOT_CAUSE.md` (E3) · `DIAGNOSTICS.md`
(E4). JSON in `results/{shuffle_new, rootcause.json, diagnostics}/`. Scripts:
`shuffle_new_repo.py`, `audio_rootcause.py`, `shuffle_diagnostics.py`, `raw_probe.py`.
Jobs 6871504/6871509/6871803-6, 6876966, 6886770-3, 6886896.

**Prior art for the brain-shuffle test:** circular-shift/mismatch null in mTRF stimulus
reconstruction (Crosse et al. 2016); the match–mismatch paradigm (de Cheveigné; Francart;
ICASSP-2024 challenge); permutation nulls (Combrisson & Jerbi 2015; Ojala & Garriga 2010);
permutation importance / model reliance (Breiman 2001; Fisher, Rudin & Dominici 2019);
shortcut/Clever-Hans (Geirhos et al. 2020; Lapuschkin et al. 2019).
