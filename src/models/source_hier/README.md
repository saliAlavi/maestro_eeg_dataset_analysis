# source_hier — geometry-factorised 4-way attended-source identification

## What it is
An upgrade of `source_net`. Keeps the backbone that works on this corpus —
multi-scale spectro-spatial EEG (`SpectralSpatialEncoder` branches at kernels
{15,33,65}) + conv gaze-trajectory encoder — but the **flat 6-way readout is
replaced by a factorised hemisphere × eccentricity head**. The lag-robust
content-match branch from `source_net` is **dropped**: this corpus's diagnostics
show envelope/content tracking is absent (its gate learned ≈0) while its 130-lag
backward dominated compute (~2 min/batch on CPU).

The 4 attendable speakers *are* the product of two binary geometric factors:

| speaker | hemisphere | eccentricity |
|---|---|---|
| 1 | Left  | Outer |
| 2 | Left  | Inner |
| 3 | Right | Inner |
| 4 | Right | Outer |

So the model predicts two scalar logits — `h` (Right>0, carried by EEG spatial
lateralisation) and `e` (Outer>0, carried by gaze azimuth) — from the fused
EEG+gaze embedding, and **composes** the per-speaker logits from this fixed
geometry. Each bit is also supervised directly with an auxiliary BCE
(`aux_weight=0.5`). The gated content branch is added on top and reports its gate.

## Why it should beat the flat head
- **Inductive bias:** the flat 6-way softmax must rediscover that the 4 classes
  factor, from only ~750 within-subject windows/fold. We hand it the geometry.
- **Sample efficiency:** both hemispheres' windows train the eccentricity bit and
  vice-versa, so each binary sub-decision sees the full dataset.
- **Fair EEG-only readout:** with gaze dropped, the hemisphere bit stays strong
  (EEG ~0.75) while eccentricity degrades to ~chance — the 4-way degrades
  gracefully instead of collapsing across hemispheres. The `all` vs `no_gaze`
  ablation in `evaluate()` is the EEG+gaze vs EEG-only comparison.

## What we use from gaze
Same as source_net: 8 window summary stats (mean/std x, y, pupil; mean/max speed)
+ the raw, non-z-scored subject-relative gaze x/y trajectory (`gaze_traj_len=32`,
the absolute-azimuth cue) through a 1-D conv. Gaze is presence-gated with train-time
modality dropout (`gaze_dropout=0.2`).

## Data / leakage
Reuses the `aad_spec` cache (broadband EEG + table-power envelopes + raw gaze) via
`configs/data/aad_source.yaml` — **no re-caching**. Intra-subject, trial-level
`chrono_forward` CV.

## Config / smoke
`configs/model/source_hier.yaml`. Smoke:
`python -m src.main mode=selftest model=source_hier data=aad_source`.
One-fold quick eval: add `runner.max_folds=1`.
