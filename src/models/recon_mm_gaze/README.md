# recon_mm_gaze — recon_mm + gaze fusion

## What it is
`recon_mm` (EEG → reconstructed attended envelope → correlation-based source
classification) **plus the eye-tracker gaze stream**, fused so we can measure —
against `recon_mm` as the EEG-only control — whether overt-orienting gaze
**helps or hurts** attended-source decoding on this dataset.

## Fusion strategies (config `fusion`)
- **`gated` (default) — late fusion.** Gaze → a spatial prior over the 6
  loudspeakers (gaze direction ≈ attended location). A learned, gaze-conditioned
  gate `g ∈ [0,1]` (forced to 0 when gaze is absent) sets how much of that prior
  is added to the EEG correlation scores: `logits = s_eeg + g · s_gaze`.
- **`film` — late fusion + feature modulation.** Additionally, the gaze embedding
  FiLM-conditions the EEG token sequence before reconstruction, letting gaze
  *steer* (not replace) the EEG decoder.

## Motivation / intuition — why this design
- **Presence-aware:** gaze is missing/invalid on some trials; the gate is forced
  to 0 there, and a zero gaze token is fed in, so absent gaze never injects noise.
- **Measurable contribution, not entanglement.** Unlike a black-box concat, the
  gate isolates the gaze term, and `evaluate()` reports **`all` (gaze on)** and
  **`no_gaze` (gate forced off)** from the *same* weights — a clean
  with/without-gaze readout, complementing the separate `recon_mm` model.
- **Gaze modality-dropout** during training (default 0.2) stops the model from
  leaning entirely on gaze, keeping the EEG path strong and the `no_gaze` readout
  fair. This guards against the project's known failure mode where overt
  orienting (gaze ~0.77) dominates and the EEG story is lost.

## Why a separate model (not just maestro masking)
The brief asks for two **distinct** models with **distinct wandb run names**
(`…__recon_mm__…` vs `…__recon_mm_gaze__…`) so the EEG-only vs EEG+gaze
comparison is a clean A/B between trained models, not a within-model mask.

## Leakage control
Identical to `recon_mm`: intra-subject, trial-level `chrono_forward` CV.

## Config
`configs/model/recon_mm_gaze.yaml` — `fusion` (`gated`|`film`), `gaze_dropout`,
`w_cls`, `w_recon`, `train.*`. CPU smoke test:
`python -m src.main mode=selftest model=recon_mm_gaze`.
