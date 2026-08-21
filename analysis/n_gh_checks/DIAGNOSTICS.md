# Diagnostics: ruling out alternative explanations for real == null

The brain-shuffle null showed real == null for all 4 modalities × 2 protocols ×
5 windows (`SHUFFLE_NEW_REPO.md`). This note rules out the obvious "the shuffle
didn't really change anything" objections, on their committed w10_h5 checkpoints.
Jobs 6886770–3 (battery) + 6886896 (raw probe).

## 1. Are the brain embeddings collapsed? — Yes, effectively constant

Mean pairwise cosine of the model's pooled `brain_enc` across test windows:

| modality | mean pairwise cosine | norm CV |
|---|---|---|
| eeg   | **0.9995** | 0.04 |
| gaze  | 0.9963 | 0.05 |
| imu   | **1.0000** | 0.01 |
| video | 0.9995 | 0.04 |

The encoder outputs a near-constant vector regardless of input, so the reshuffled
embeddings are ~identical to the originals and the cosine scores barely move.

## 2. Not just "gentle shuffle": constant / zeros / noise / cross-subject all equal real

**LOSO 4-way accuracy (chance 0.25) by what is fed as the brain input:**

| modality | real | within-shuffle | constant (mean) | zeros | Gaussian noise |
|---|---|---|---|---|---|
| eeg   | 0.4969 | 0.4962 | 0.4969 | **0.4969** | 0.4965 |
| gaze  | 0.5149 | 0.5153 | 0.5156 | **0.5156** | 0.5149 |
| imu   | 0.5112 | 0.5112 | 0.5112 | **0.5112** | 0.5112 |
| video | 0.5162 | 0.5166 | 0.5156 | **0.5156** | 0.5169 |

Feeding **literal zeros** as the brain gives the same accuracy as the real brain.
In the within-subject fold (pools all 16 subjects) a **cross-subject shuffle**
(each window's brain replaced by a *different participant's*) also equals real
(eeg 0.4004 vs 0.4000; gaze/imu/video identical). Since zeros/noise carry zero
participant information yet score identically, participant-clustering of the
embeddings cannot be what sustains the accuracy.

## 3. The decision never changes

Per-trial 4-way argmax **flip rate** vs the real-brain prediction is **0.000–0.002**
for every modality and every manipulation (shuffle, constant, zeros, noise,
cross-subject). The winning candidate is a function of the audio alone.

## 4. Did preprocessing discard the neural signal? — No; the ENCODER collapsed it

Positive control on the repo's own preprocessed EEG (their 64 Hz pipeline),
simple linear probe, trial-disjoint:

| decode target | from RAW preprocessed EEG | from model's `brain_enc` | chance |
|---|---|---|---|
| **subject identity** (16-way) | **0.900** | 0.101 | 0.0625 |
| **attended speaker** (4-way)  | **0.337** | (model brain margin 0.000) | 0.25 |

Raw preprocessed EEG carries **90 %-decodable** subject structure and **above-chance
(0.337) attended-speaker** information — a trivial linear probe extracts more genuine
neural attention signal than the entire deep model's brain path (which contributes a
0.000 margin). The model's encoder compresses that 0.90 subject-decodable signal down
to 0.10. So the information is present after preprocessing; the **encoder discarded it**
because the audio shortcut alone minimizes the training loss. (The 0.337 attended probe
is attention-*related* linear structure; part may reflect covert alpha lateralisation
and part the overt-orienting/gaze confound — either way it proves the signal survives
preprocessing.)

## Conclusion

Every alternative explanation for real == null is ruled out:
- embeddings are collapsed (cos ≈ 1), **and**
- zeros / noise / cross-subject inputs give identical accuracy (so it is not
  participant-clustering, not a too-gentle shuffle), **and**
- the decision never flips, **and**
- preprocessing preserved the neural signal (raw-EEG → subject 0.90, → attended
  0.337); the model's encoder collapsed it.

The ~0.50 four-way is 100 % audio-driven for all four modalities. The brain/behaviour
path is vestigial: the network trained itself to ignore the EEG (and gaze/IMU/video)
because the loudness-induced acoustic marking of the attended talker is the easy
shortcut. Data: `results/diagnostics/*.json`.
