# source_azi — attended source as a latent azimuth (reliability-weighted observers)

## Motivation (from the data, not the architecture)
Probing the cached signal made two things clear:
1. **The 4 speakers are azimuth-ordered** (idx 0..3 = Left-outer, Left-inner,
   Right-inner, Right-outer). A flat 4/6-way softmax discards this; inner/outer
   stays at chance.
2. **Gaze reliability is wildly heterogeneous.** gaze-x by speaker (cached):
   - S2 `[0.378, 0.421, 0.525, 0.544]` — monotone → **full azimuth**.
   - S3 `[0.435, 0.437, 0.609, 0.607]` — splits hemisphere, flat within → **hemisphere only**.
   - S1 `[0.541, 0.515, 0.533, 0.526]` — flat → **dead** (no overt orienting; this is
     why S1 is at chance).
   A fixed-trust EEG+gaze fusion is wrong for two of the three subjects.

## What it does
Treats each modality as a **noisy observer of one shared latent azimuth**:
- **EEG observer** — multi-scale spectro-spatial (CSP) encoder → `(μ_e, log-prec_e)`.
- **Gaze observer** — learned per-subject affine self-calibration → conv over the
  gaze trajectory/stats → `(μ_g, log-prec_g)`.

Each emits a mean azimuth **and a per-sample precision (self-trust)**.
**Bayesian fusion**: `μ = Σ prec_i·μ_i / Σ prec_i`, `prec = Σ prec_i`. A dead gaze
observer (S1) learns low precision and is ignored; a full-azimuth one (S2) dominates
— learned per sample, not hand-set. **Azimuth-anchored readout**: 4 ordered anchors
at the speaker angles, `logit_k = -½·prec·(μ − anchor_k)²`.

## Losses
- 4-class CE on the anchored logits (top-level decision).
- **Per-observer heteroscedastic Gaussian NLL** over azimuth,
  `0.5·prec_i·(μ_i − anchor[att])² − 0.5·log_prec_i` — trains each observer to
  predict its **own reliability** (what makes the fusion subject-adaptive) and keeps
  the EEG observer individually supervised so gaze can't hijack the gradient.
  Gaze NLL is masked to present/kept windows.

## EEG-only vs EEG+gaze
The gaze-observer ablation (`prec_g → 0`) **is** the EEG-only condition — same model,
so `all` vs `no_gaze` in `evaluate()` is an exact comparison. Presence-gated gaze +
train-time `gaze_dropout` keep the EEG path honest.

## What this can't do alone
EEG's eccentricity signal is faint but **consistent across subjects** (corr ≈ +0.1 to
the inner/outer bit in all three). Single-subject data can't nail it — the queued next
lever is **cross-subject pretraining** of the EEG observer (needs datamodule support;
pays off most at n=16).

## Data / config
Reuses the `aad_spec` cache via `configs/data/aad_source.yaml` — no re-caching.
`configs/model/source_azi.yaml`. Smoke:
`python -m src.main mode=selftest model=source_azi data=aad_source`.
One-fold quick eval: add `runner.protocols=[within] runner.max_folds=1`.
