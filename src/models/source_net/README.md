# source_net — unified improved 4-way attended-source identification

## What it is
A strict upgrade of `source_fusion` (0.485 4-way). Three branches:

1. **Multi-scale spectro-spatial EEG** — parallel EEGNet/CSP branches at temporal
   kernels {15, 33, 65} (≈ beta / alpha / theta scales), concatenated log
   band-power. Sharper hemisphere (left/right) bit than a single-scale encoder.
2. **Conv gaze-trajectory encoder** — 1-D conv over the raw subject-relative gaze
   x/y sequence (`gaze_traj_len=32`, the azimuth cue), targeting the within-
   hemisphere (inner/outer) resolution that caps 4-way accuracy.
3. **Lag-robust content-match branch** (the `deep_match` mechanism) added as a
   **learned-gated** term (gate init ≈ 0.12, spatial-dominant). If any stimulus-
   tracking signal exists it contributes; if not (diagnostics say not), the gate
   learns ≈ 0 and the model falls back to spatial+gaze. So **source_net ≥
   source_fusion by construction**, with upside if content ever helps.

Joint fusion of spatial + gaze → base 6-way logits; gated content scores added;
4-class CE + label smoothing; presence-aware gaze + gaze dropout; `evaluate()`
reports `all` vs `no_gaze`. The `content_gate` is logged so we can read off whether
the envelope branch earned any weight.

## Why these levers
EEG gives the hemisphere bit but is at chance for inner/outer (eccentricity isn't
lateralised) — so the multi-scale EEG sharpens hemisphere while the conv gaze
encoder supplies the within-hemisphere azimuth. Content matching is the honest
long-shot, gated so it can't hurt.

## Data / leakage
Reuses the `aad_spec` cache (broadband EEG + table-power envelopes + raw gaze) via
`configs/data/aad_source.yaml` (`gaze_traj_len=32`) — **no re-caching**. Intra-
subject, trial-level `chrono_forward` CV.

## Config
`configs/model/source_net.yaml`. Smoke:
`python -m src.main mode=selftest model=source_net data=aad_source`.
