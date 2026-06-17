# eeg_spatial_gaze — eeg_spatial + gaze fusion

## What it is
`eeg_spatial` (EEG alpha-lateralisation attended-speaker classifier) **plus the
gaze stream**, fused so we can measure — against `eeg_spatial` as the EEG-only
control — whether overt-orienting gaze helps or hurts. Both modalities encode the
**spatial** locus of attention (EEG: alpha lateralisation; gaze: eye direction),
so this is the honest spatial multimodal decoder for this corpus, where envelope
tracking is unrecoverable (see `eeg_spatial`).

## Fusion
Presence-aware **gated late fusion** (same contract as `recon_mm_gaze`): gaze →
a 6-speaker spatial prior; a gaze-conditioned gate `g ∈ [0,1]`, forced to 0 when
gaze is absent, sets `logits = s_eeg + g · s_gaze`. `evaluate()` reports **`all`
(gaze on)** vs **`no_gaze` (gate off)** from one model; gaze modality-dropout
(0.2) keeps the EEG path honest so the `no_gaze` readout is fair.

## Why separate from eeg_spatial
Two distinct models → distinct wandb run names (`…__eeg_spatial__…` vs
`…__eeg_spatial_gaze__…`) → a clean A/B for the gaze contribution, not a
within-model mask. Expect gaze to add the most, since on this dataset overt
orienting is the strongest attention cue (~0.77).

## Config
`configs/model/eeg_spatial_gaze.yaml` (`F1`, `D`, `gaze_dropout`).
Data: `configs/data/aad_spec.yaml`. Smoke:
`python -m src.main mode=selftest model=eeg_spatial_gaze`.
