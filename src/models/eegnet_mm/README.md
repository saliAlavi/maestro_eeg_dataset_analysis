# eegnet_mm — EEGNet match-mismatch (EEG + audio only)

## What it is
An EEGNet-style encoder turns the EEG window into a query embedding; a
weight-shared 1-D conv encoder turns each of the six candidate speaker
envelopes into a key. The attended speaker is the candidate whose key best
matches the query (scaled dot-product → cross-entropy over candidates).

## Motivation / intuition
This is the contemporary neural standard for AAD (Accou/Vandecappelle/Francart,
ICASSP Auditory-EEG-Decoding). The **match-mismatch** framing is stimulus-aware:
the model scores candidates rather than memorising speaker identities, so it
generalises to unseen speech, and speakers 5/6 (never attended) act as free hard
negatives. Weight-sharing across candidates makes it permutation- and
identity-agnostic.

## Why we chose it (what mistakes led here)
A flat 4-class EEG classifier (our earlier iterations) can't exploit the audio
at all and isn't comparable to the AAD challenge literature. Match-mismatch
fixes both. It is also the controlled ablation for MAESTRO-Net: same EEG+audio
core, **no** gaze/IMU/video — so the gap between `eegnet_mm` and `maestro`
isolates exactly how much the orienting modalities add (the project's
"EEG ↔ each modality" thesis).

## Key hyperparameters (`configs/model/eegnet_mm.yaml`)
- `d_model`, `dropout`, and the shared `train.*` block (epochs, lr, batch_size)
