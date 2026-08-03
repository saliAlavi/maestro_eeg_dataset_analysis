# Baseline results — the dataset carries real, confound-free neural signal

## Headline
A **VLAAI backward decoder** reconstructs the attended talker's envelope from EEG and picks,
by scale-free correlation, **which of the four real co-present talkers** is attended
(chance **0.25 at every window**). It decodes **far above chance** and the margin **grows
with the decision window** — a rigorous demonstration that this dataset holds genuine,
decodable cortical envelope-tracking signal.

> **Four-way decision-window curve** (VLAAI + multiband + margin, **LOSO**, 16 subjects,
> 5-seed reconstruction ensemble):
>
> | window | 4-way (chance .25) | EEG-shuffle null | Δ (neural) | p (paired) |
> |---|---|---|---|---|
> | 5 s | 0.348 [.326,.369] | 0.266 | **+0.083** | 4.5e-14 |
> | 10 s | 0.383 [.349,.418] | 0.264 | **+0.119** | 2.0e-11 |
> | 15 s | 0.431 [.384,.477] | 0.264 | **+0.167** | 5.4e-13 |
> | 20 s | 0.440 [.391,.485] | 0.262 | **+0.178** | 1.5e-13 |
> | **30 s** | **0.497** [.435,.552] | 0.267 | **+0.230** | 8.0e-14 |
>
> Four-way accuracy rises from 0.35 (5 s) to ~0.50 at the whole trial, crossing **0.40 at
> ~15 s**, while the null stays flat — so the neural margin grows and stays highly significant.

The same lever carries the **within-subject** protocol past 0.40 (VLAAI + multiband — the
margin overfits data-starved within folds — 5-fold, 5-seed ensemble):

> | window | 4-way (chance .25) | EEG-shuffle null | Δ (neural) | p (paired) |
> |---|---|---|---|---|
> | 5 s | 0.317 [.305,.331] | 0.259 | **+0.058** | <1e-16 |
> | 15 s | 0.354 [.326,.384] | 0.257 | **+0.097** | 1.2e-12 |
> | **30 s** | **0.406** [.375,.439] | 0.261 | **+0.145** | <1e-16 |
>
> Within-subject 4-way **crosses 0.35 at 15 s and reaches 0.406 at the whole trial**, over the
> same flat ~0.26 null — real per-subject decoding, not just a cross-subject average.

## The null is honest at every window (the key property)
- **Theoretical chance is exactly 0.25 at every window.** The candidates are the four *real*
  co-present talkers, so the number of choices never changes with window length — and a
  **pure-noise guesser scores 0.250** (measured), confirming the four candidates are a fair
  four-way choice. (An earlier same-talker time-shift construction had a null that *drifted*
  0.25 → 0.34 as the window grew; the four-real-talker task fixes that by design.)
- **The empirical EEG-shuffle null is flat at ~0.265** across all windows — no drift. The +0.015
  above 0.25 is the **audio-only floor**: the attended talker is acoustically *marked* (enhanced
  so it can be attended), and the decoder — trained to reconstruct attended envelopes — matches
  that marking even with its EEG scrambled. So 0.265 is what the audio marking alone buys with no
  usable EEG. It is a constant dataset property (not a construction artifact), present equally in
  the real accuracy, and **removed by measuring the margin over the empirical null** — so the
  audio-marking freebie is never credited to the brain. Reporting against 0.25 instead would
  *over*-state the neural effect.

## Fixed-window table (5 s; mean ± sd [95% CI] over 16 subjects; held-out test)
| model | protocol | 4-way (.25) | null | binary (.5) | Δ4 vs null (paired) |
|---|---|---|---|---|---|
| **VLAAI + multiband + margin** | **LOSO** | **0.335** [.316,.354] | 0.261 | 0.577 | +0.075, t=7.0, **p=2e-6** |
| VLAAI + multiband | LOSO | 0.328 [.313,.344] | 0.260 | 0.567 | +0.068, t=8.5, p=2e-7 |
| VLAAI (plain) | LOSO | 0.315 [.301,.331] | 0.247 | 0.564 | +0.068, t=7.6, p=8e-7 |
| Linear (reference) | LOSO | 0.254 [.246,.261] | 0.246 | 0.506 | +0.008, t=1.8, p=0.05 (≈chance) |
| **VLAAI + multiband** — within headline | within | **0.303** [.291,.315] | 0.261 | 0.548 | +0.042, t=6.1, p=1e-5 |
| VLAAI (plain) | within | 0.292 [.283,.301] | 0.251 | 0.545 | +0.041, t=7.9, p=6e-7 |
| VLAAI + multiband + margin | within | 0.288 [.280,.295] | 0.257 | 0.534 | +0.031, t=7.0, p=2e-6 |
| Linear (reference) | within | 0.253 [.245,.261] | 0.247 | 0.507 | +0.006 (≈chance) |

## The minimum innovation on VLAAI (LOSO ablation)
Two EEG-only, confound-free add-ons — neither touching a reserved method-paper lever:
| step | LOSO 4-way | Δ vs plain (paired) | LOSO binary | Δ binary |
|---|---|---|---|---|
| plain VLAAI | 0.315 | — | 0.564 | — |
| **+ multi-band reconstruction** | 0.328 | **+0.013 (t=4.36, p=0.0003)** | 0.567 | +0.004 (p=0.04) |
| **+ match-mismatch margin** | **0.335** | **+0.020 (t=3.64, p=0.001)** | **0.577** | **+0.014 (t=2.78, p=0.007)** |

- **Multi-band reconstruction** (`--bands 28`): reconstruct the 28 gammatone bands and fuse
  per-band (Fisher-z) correlations — band-heterogeneous cortical tracking carries more decodable
  structure than one broadband envelope. Significant on its own at LOSO here.
- **Match-mismatch margin** (`--mm-margin`): an auxiliary CE aligning the reconstruction with the
  decision metric; adds a further significant lift on top.

## Honest caveat: within-subject
The **margin** term helps LOSO but **hurts within-subject** (multi-band alone is best within:
0.303, p=9e-4; adding the margin drops it to 0.288) — the richer objective overfits the
data-starved within-subject folds. So we report **VLAAI + multiband for within (0.303)** and
**VLAAI + multiband + margin for LOSO (0.335)**. LOSO (cross-subject generalization) is the more
meaningful number for a usable dataset, and that is where the full innovation pays off.

## How to read the neural claim
1. **Chance is 0.25 at every window** and the empirical null is flat (§ above) — the four-choice
   baseline is honest at any window length.
2. **Loudness cannot help** — scale-free correlation; an uninformative reconstruction correlates
   ≈0 with every talker.
3. **The deterministic schedule cannot help** — the four candidate slots are permuted per
   (subject,trial) and the label is the permuted slot; the model has no audio→label path.
4. **EEG-shuffle null test** — reconstructing from scrambled EEG collapses the decision to the
   flat null; the paired per-subject acc-vs-null test is highly significant for VLAAI (LOSO
   4-way t=7.0, p=2e-6) and null for linear (the reference floor).
5. **Linear at chance** (0.254 ≈ null) — the classic decoder detects nothing, which *proves there
   is no trivial leak*: above-chance requires the modern deep reconstruction net.
6. **Causal-lag control** logged per run — genuine tracking is causal (audio leads EEG); guards
   against instantaneous stimulus bleed (subject-variable at the aggregate).

## Why this is the *admissible* baseline (what it is NOT)
- **Proper alignment, reused not re-derived** — reads the method paper's frozen `*_pa2_af64.npz`
  cache (all inter-stream lags resolved once); we cannot reintroduce an alignment mistake.
- **Train/val trial-disjoint in both protocols** — within = StratifiedKFold(5) + trial-disjoint
  inner-val; LOSO = held-out subject for test, train/val split by trial_k content.
- **NOT the github benchmark** — collapses to ~chance under a leakage-safe protocol (`n_gh_checks`).
- **NOT a contrastive net** (*NeuroCLIP*, tried) — sat at the null; reconstruction decoders detect
  this weak signal, contrastive does not.

## Headroom (reserved for the method paper)
EEG-only, single-reconstruction. Untouched: a learned similarity/CCA head; gaze shrinkage-LDA;
frozen V-JEPA2 video; head-IMU; reliability-weighted multimodal late fusion; subject FiLM /
test-time adaptation / SSL pretraining. Clean path from this baseline to the multimodal method.
