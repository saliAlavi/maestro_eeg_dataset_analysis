# Dataset-paper baseline — design & rationale

## 1. Goal and positioning

The dataset paper needs a baseline that **shows the corpus contains real, decodable EEG
signal** (not noise) — **honestly** (leakage-safe), **clearly better** than the
inadmissible public github model (which collapses to ~chance under a leakage-safe
protocol; see `analysis/n_gh_checks`), yet **deliberately below** the team's held-back
multimodal branch-and-fuse method (a separate method paper). This is an **EEG-only,
single-reconstruction** model — a floor with a clean monotone path up to the method
paper.

## 2. The task: four-way attended-talker match-mismatch (chance 0.25 at EVERY window)

We decode **which of the four real co-present attendable talkers** (loudspeakers 1–4) the
listener attends — the same four-way task as the method paper, so chance is **structurally
0.25 at any decision window** (5 s or the whole ~30 s trial): a window always contains
exactly those four real streams, so the number of choices never changes with length. We
report **both** framings from this one candidate set:
- **4-way** (argmax over the four talkers; chance 0.25) — the headline;
- **binary** (attended vs each other talker; chance 0.5) — the canonical AAD metric.

The two confounds this corpus contains are defused **architecturally** (the method-paper
recipe), not by substituting spoilers:
1. **Loudness.** The decision is a **scale-free correlation** of the EEG reconstruction
   with each candidate (`backward.mm_scores`). An uninformative reconstruction correlates
   ≈0 with *every* talker, so amplitude cannot help — the branch can only win by
   reconstructing genuine temporal structure.
2. **Clip identity / the deterministic attended schedule** (`attended==((k-1)%4)+1`, the
   same 100 stimuli replayed to every subject). The four candidate streams are **permuted
   per (subject,trial)** and the label becomes the attended talker's permuted **slot**; the
   same `trial_k` lands on different slots for different subjects, so `trial_k→attended`
   cannot be memorised. The model has **no audio→label path** at all (audio enters only as
   a correlation *target*), and the **EEG-shuffle null then verifies chance = 0.25**.

> **Why the four *real* talkers, not same-talker time-shifted spoilers.** An earlier
> version used the attended talker's envelope plus time-shifted "spoilers" of the same
> talker. That is confound-free at short windows, but the matched and shifted candidates
> are **not exchangeable** at long windows (a circular-shift spoiler can carry a wrap
> discontinuity; shift-room collapses as the window approaches the trial length), so the
> scrambled-EEG null **drifted** from 0.25 up to ~0.29–0.34. Using the four real
> co-present talkers restores a null that is **flat** at every window (a pure-noise guesser
> scores 0.250; the theoretical four-choice chance) — the property a four-choice baseline must
> have. The empirical EEG-shuffle null sits a touch higher, ~0.265: the attended talker is
> acoustically **marked**, so the trained decoder matches it slightly even with EEG scrambled
> (0.265 = the *audio-only* floor). That +0.015 is present equally in the real accuracy and is
> removed by measuring the neural margin over the empirical null — so it is never credited to
> the brain, and unlike the spoiler null it does **not drift** with window.

This is the canonical AAD paradigm (Wong/O'Sullivan/Fuglsang; Accou/Francart; ICASSP-2023
challenge), on the four real talkers rather than a two-way stimulus/mismatch pair.

## 3. The models: backward (stimulus-reconstruction) decoders

Envelope tracking on this corpus is **weak but present**, and — consistent with the AAD
literature and the team's prior work (a tuned backward decoder reaches binary ~0.60,
t=3.79) — it is detected by **backward decoders**, not by a contrastive match-mismatch
network (see §8). Both models reconstruct the attended **broadband envelope** from EEG,
and the match-mismatch decision correlates the reconstruction against each candidate.

- **VLAAI backward (headline; modern deep).** A VLAAI-style fully-convolutional network
  (Accou et al. 2023, *Nature Sci. Reports*): a learned 32→128 spatial projection, then 4
  **residual conv blocks**, each two `Conv1d(128,128,k9)`+LeakyReLU+Dropout(0.2) followed
  by an **output-context** `Conv1d(k33)`, then a 1×1 conv to the 1-D envelope. **No
  BatchNorm** (avoids train-subject running-stat leak in LOSO). ~0.5 M params.
- **Linear backward (reference floor; canonical).** A single causal `Conv1d(32,1,k=64)`
  integrating a ~1 s lag window across channels — the standard linear AAD decoder.

**Training:** reconstruct the **attended talker's** broadband envelope, **negative-Pearson**
loss; AdamW, cosine LR + warmup, grad-clip 1.0; early stop on inner-val **binary MM
accuracy** (patience 15); single held-out test eval of the best-val checkpoint. Seeds 0–2.
VLAAI: lr 1e-3, wd 1e-4, ≤80 ep, batch 128, dropout 0.2. Linear: lr 1e-3, wd 1e-2
(ridge), ≤60 ep.

**Decision / metrics:** correlate the reconstruction `r̂` with each of the four real
candidate envelopes → scores; **4-way** = argmax over the four talkers (chance 0.25);
**binary** = fraction of the other three talkers `r̂` correlates with less than the
attended. A fully-convolutional decoder trained at 5 s scores at any window, so we also
report a **decision-window curve** (train 5 s, evaluate 5 → 30 s, reconstruction-ensembled
across seeds); the four-talker candidate set keeps the null at 0.25 at every point.

### 3a. The minimum innovation on VLAAI
Two small, EEG-only, confound-free add-ons that raise accuracy without touching any
reserved method-paper lever:
- **Multi-band (spectrogram) reconstruction (`--bands 28`)** — reconstruct the **28 gammatone
  bands** and fuse **per-band (Fisher-z) correlations** in the decision. Cortical tracking is
  band-heterogeneous, so the richer spectro-temporal target carries more decodable structure.
  Significant on its own at LOSO (4-way **+0.013, p=3e-4**). The reserved "learned
  similarity/CCA head" is *not* used — the decision is still plain per-band correlation.
- **Match-mismatch margin (`--mm-margin`)** — an auxiliary CE over the reconstruction's
  correlations with the 4 candidates, aligning training with the decision metric. Adds a
  further significant LOSO lift (combined 4-way +0.020, p=1e-3; binary +0.014, p=7e-3), but
  **overfits within-subject** — so multiband-alone is the within headline (see §6, `report/`).

Both are toggled from the CLI; the full ablation (plain → +multiband → +margin) is in the
report table with a paired-significance test vs. plain VLAAI.

## 4. Modern element + what's deliberately left out
- **VLAAI deep backward net** is the recent-research headline decoder (deep residual
  fully-convolutional reconstruction, the current SOTA-lineage envelope decoder), paired
  with the linear decoder as the field-canonical reference.
- **Headroom for the method paper (untouched levers):** a learned similarity/CCA head;
  the gaze shrinkage-LDA direction branch; frozen V-JEPA2 video; head-IMU; reliability-
  weighted multimodal late fusion; subject FiLM / test-time adaptation / SSL pretraining.

## 5. Leakage controls (the whole point)
1. **Proper cross-modal alignment (reused, not re-derived).** EEG, per-talker audio, gaze,
   and IMU each have a *different* recording lag. We do **not** align them ourselves; we
   read the project data-processing module's frozen cache (`*_pa2_af64.npz`, `pa2` =
   proper-alignment v2, `af64` = audio features at 64 Hz) — the exact aligned dataset the
   method paper is built on — so we cannot reintroduce an alignment mistake.
2. **Structural chance of 0.25 at every window.** Candidates are the four real co-present
   talkers, so the number of choices is 4 at any window length; the EEG-shuffle null is
   0.25 at 5 s and at the whole trial alike (no window-dependent drift). See §2.
3. **No loudness cue:** the decision is a **scale-free correlation** — an uninformative
   reconstruction correlates ≈0 with every talker.
4. **Deterministic schedule cannot reach the model:** the four candidates are **permuted
   per (subject,trial)** and the label is the attended **slot**; the model's only inputs
   are `(eeg, candidates)` with no audio→label path, so `trial_k→attended` is unusable.
5. **Trial-disjoint train/val in BOTH protocols.** *within* = per-subject StratifiedKFold(5);
   the inner-val is a further attended-stratified, **trial-disjoint** hold-out of the
   training trials (test fold, val, train share no trial). *LOSO* = held-out subject for
   test; train/val from the other 15 subjects split by **trial_k content** (a val trial_k
   never appears in train), so the same stimulus is never in both train and val
   (`data.py splits`).
6. **No subject-stat leak:** BatchNorm-free (running-stat-free) models; per-trial EEG and
   per-band candidate z-scoring; no subject embedding.
7. **EEG-shuffle null** (reconstruct from scrambled EEG → must hit 0.25) reported every
   run; the **paired per-subject acc-vs-null test** is the rigorous neural evidence. A
   **causality/lag curve** is also logged: genuine cortical tracking is causal (audio
   leads EEG ~100–250 ms), which separates neural tracking from any instantaneous
   stimulus bleed.

## 6. Results (16 subjects; see `report/summary.md` + `report/curve.md`)
**Headline — 4-way decision-window curve (VLAAI + multiband + margin, LOSO):** 4-way rises
**0.348 (5 s) → 0.431 (15 s) → 0.497 (30 s)** over a **flat ~0.265 null** (theoretical chance
0.25); neural margin Δ = +0.083 → +0.230, all **p < 10⁻¹⁰** (paired). It crosses **0.40 at
~15 s** — honestly, over a null that does not drift.

**Fixed 5 s (LOSO):** VLAAI+mb+margin 4-way **0.335** (null 0.261, binary 0.577), t=7.0
p=2e-6 vs null; plain VLAAI 0.315; **linear 0.254 ≈ chance** (the reference floor proves no
trivial leak). **Innovation (LOSO):** +multiband +0.013 (p=3e-4), +margin a further +0.007
(combined +0.020, p=1e-3). **Within:** the margin overfits the data-starved folds, so
multiband-alone is the within headline. **Within decision-window curve (`report/curve_within.md`,
VLAAI + multiband, 5-fold, 5-seed ensemble):** 4-way **0.317 (5 s) → 0.354 (15 s) → 0.406 (30 s)**
over the same flat ~0.26 null (Δ +0.06 → +0.15, all p<1e-10) — per-subject decoding crosses 0.35
at 15 s and reaches ~0.41 at the whole trial. Modest and honest — the corpus's envelope tracking
is genuinely weak — but a clean, confound-free demonstration that the dataset carries decodable
neural signal in *both* protocols, with wide headroom to the multimodal method.

## 7. What's reported alongside (for the paper)
- The **linear** backward decoder (canonical reference floor).
- The **github** model's collapse under a leakage-safe protocol (`n_gh_checks`) as the
  inadmissible prior baseline.
- The confound audit itself (source-ID is inadmissible; content-disjoint splits required)
  — a methodological contribution telling future dataset users what is and isn't valid.

## 8. Design provenance (and a negative result worth reporting)
A 5-way judge panel selected a modern **contrastive** match-mismatch model
(*NeuroCLIP-AAD*, CLIP-style EEG↔envelope InfoNCE) as the headline. It is kept in the
repo (`models/model.py`, `train.py`) because it produced an informative **negative
result**: across 4-way and binary framings it sat **at the null** (acc ≈ null ≈ chance) —
the contrastive frame-cosine is the wrong inductive bias for this corpus's weak tracking.
Switching to **backward/reconstruction** decoders (this document) recovered the signal,
matching the AAD literature and the team's prior findings. The lesson — *reconstruction
beats contrastive for weak envelope tracking here* — is itself a useful baseline finding.
