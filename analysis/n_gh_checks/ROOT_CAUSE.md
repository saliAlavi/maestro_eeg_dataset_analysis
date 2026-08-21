# Root cause: why the MAESTRO audio encoder decodes the attended talker at ~0.50

We showed the repo's T1 "attended-source decoding" is non-neural (real == null for
all 4 modalities × 2 protocols × 5 windows; see `SHUFFLE_NEW_REPO.md`): the decision
comes entirely from the candidate audio, not the brain. This note finds *why the audio
alone is enough*, and rules out the two obvious explanations.

## It is NOT train/test audio overlap

From `audio_manifest.json` (100 main trials × 4 attendable talkers = 400 slots):

- **Every talker is a unique voice** — 400 distinct `spkid`s, **zero** reused across
  trials (mean occurrences/spkid = 1.00).
- Within a fold, **train and test share 0 voices** (fold-0: 80 test vs 320 train voices,
  intersection 0). The official within splits are content-disjoint; LOSO adds a content
  holdout on top.

So there is no audio/voice to memorize and no leakage of specific stimuli across the
split. Overlap is refuted — yet the audio-only accuracy is still ~0.50.

## It IS an acoustic "marking" of the attended talker, baked into the stimulus design

The attended talker is systematically **louder** than its competitors:

- **attended == loudest of the 4 in 100/100 trials**, median **+15.1 dB** (range 4.8–22.7 dB;
  from the `device*_power` columns of `trials.csv`).
- Argmax of raw RMS gives a **1.0000** four-way "loudness oracle" — the label is perfectly
  determined by loudness in the raw audio.

The repo's `extract_envelope` z-scores each candidate (and even applies an RMS
pre-scaling), which removes *scalar level* — so the model cannot read loudness directly.
**But loudness left a residual, level-free SHAPE fingerprint that transfers across
content**, and that is what the encoder reads.

### Proof — a level-free linear probe reproduces the effect

`audio_rootcause.py` builds each candidate's z-scored envelope exactly as the repo does
(target-RMS scale → |Hilbert| → 20 Hz LP → 64 Hz → z-score), then computes 8 **scale-
invariant** shape features (kurtosis, skew, p95−p5 dynamic range, silence fraction,
modulation-band power ratios, Gini sparsity) and trains a plain logistic "attended vs not"
probe, **content-disjoint 5-fold**, argmax over the 4 candidates per trial:

| decoder | 4-way acc (chance 0.25) |
|---|---|
| Loudness oracle (argmax raw RMS) | **1.0000** |
| **Shape-only linear probe** (z-scored envelope, content-disjoint) | **0.6500** |
| Deep `AADModel` audio-only floor (their number) | ~0.50 |

A trivial linear model on level-free shape features **beats** the deep encoder. So the deep
audio path is not learning anything subtle — it is picking up a shape artifact that a
logistic regression reads even better.

### What the shape fingerprint is (and its causal link to loudness)

Per-feature separability of attended vs unattended, and each feature's correlation with the
raw loudness that defines "attended":

| feature (on z-scored envelope) | AUC(att vs not) | corr with loudness |
|---|---|---|
| kurtosis            | 0.190 (attended **lower**) | −0.27 |
| skew                | 0.219 (attended **lower**) | −0.46 |
| Gini sparsity       | 0.297 (attended **lower**) | −0.43 |
| dynamic range p95−p5| 0.619 (attended **higher**)| +0.22 |
| HF 8–20 Hz power    | 0.401 (attended **lower**) | −0.19 |

The attended (louder) talker has a **less peaky / less sparse / less kurtotic** envelope
with a **fuller dynamic range** — and each of these shape features is **correlated with the
raw loudness**. Mechanism: the competing talkers are presented ~15 dB down, so relative to
their noise/masking floor they are lower-SNR; z-scoring a low-SNR envelope yields a spikier,
higher-kurtosis, sparser shape than the clean, continuous envelope of the loud target. The
loudness manipulation therefore imprints a modulation-shape signature on the *target role*
itself, independent of the specific voice — so it survives z-scoring and generalizes to
unseen talkers and unseen subjects.

## Conclusion

The ~0.50 four-way is **not** brain decoding, **not** train/test overlap, and **not** the
audio being "too similar." It is the opposite: the attended talker is made systematically
**different (louder)** by the AAD stimulus design, and that loudness leaves a **level-free
envelope-shape fingerprint of the target role** that a learned (or even linear) audio
encoder reads off the candidates directly. Because the fingerprint is a property of the
*role*, not the content, no subject- or content-disjoint split removes it — only the
EEG-shuffle null exposes that the model is reading it instead of the brain.

Implication for a *correct* benchmark: the candidate envelopes must be equalized in
**distribution/shape**, not just level (e.g. rank/histogram normalization or matched
modulation statistics per trial), and every model must be reported against its
brain-shuffle null. Data: `results/rootcause.json`; job 6876966.
