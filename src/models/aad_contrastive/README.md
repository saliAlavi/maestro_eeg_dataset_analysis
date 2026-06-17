# aad_contrastive — content-based attended-source identification (contrastive)

## What it is
Identifies **which audio source the EEG is tracking** by learning a shared space where
the EEG representation is **close to the attended speaker's envelope and far from the
unattended ones**. This is *content* matching — NOT spatial/direction decoding. The
encoder is position-agnostic and candidates are permuted, so the only usable signal is
the EEG↔envelope correspondence (neural envelope tracking).

## Biological priors (architecture)
- **Spatial filter** (1×1 conv over channels) → auditory-cortex source components
  (CSP/beamformer-like), on raw re-referenced EEG (BatchNorm whitens; `norm_eeg=false`
  so relative channel amplitudes survive).
- **TRF temporal filter** (~0–400 ms kernel) → the cortical envelope-tracking response,
  in the delta–theta band (cache is 1–10 Hz).
- **Similarity = lag-tolerant temporal correlation** (CCA-style), amplitude-invariant —
  the quantity the brain tracks; a ±~375 ms lag search absorbs residual jitter.

## Contrastive training (InfoNCE)
- **Positive:** the attended speaker's envelope.
- **Hard negatives:** the unattended speakers in the *same 6-speaker scene* (including
  the always-distractor speakers 5/6) — identical acoustics, only attention differs.
- **In-batch negatives:** attended envelopes of other trials in the batch.
- **Temperature**-scaled; **candidate order permuted** per sample → no positional cue.

Prediction: pick the attendable speaker (1–4) whose envelope is most similar to the EEG.

## Why this vs. the spatial models
`eeg_spatial`/`source_net` decode the *location* of attention (alpha lateralization +
gaze azimuth) — direction, not source content. This model is the honest test of whether
the attended *source* is identifiable from EEG↔envelope tracking, with the strongest
biologically-grounded, contrastive formulation. (Prior content matchers — CCA,
recon_mm, deep_match — were at chance; this adds proper contrastive metric learning,
TRF priors, and same-scene hard negatives.)

## Data / config
`configs/data/aad_match.yaml` (aad_recon cache, EEG 1–10 Hz, `norm_eeg=false`,
table-power-equalised envelopes), all 16 subjects, intra-subject `chrono_forward` CV.
`configs/model/aad_contrastive.yaml`. Smoke:
`python -m src.main mode=selftest model=aad_contrastive data=aad_match`.
