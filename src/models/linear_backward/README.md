# linear_backward — stimulus-reconstruction AAD (classical baseline)

## What it is
A single linear decoder `g` reconstructs the **attended** broadband speech
envelope from time-lagged EEG (forward lags 0–~250 ms). At decision time we
reconstruct the envelope for a window and correlate it against each of the six
candidate envelopes; the highest-correlation speaker, restricted to the four
attendable ones, is the prediction.

## Motivation / intuition
Cortical envelope tracking is the most established neural signature of auditory
attention (O'Sullivan et al., 2015). A backward (EEG→stimulus) linear model is
the field's reference decoder: cheap, interpretable, no GPU. It is the floor
that every neural model in this repo must clear to justify its complexity.

## Why we chose it (what mistakes led here)
Our own diagnostics showed gaze alone (~0.77) beats EEG-spectral features
(~0.72) and that motion-residualised EEG collapses to chance — so a fancy net
that "wins" could simply be exploiting overt orienting. We need a *purely
neural, stimulus-driven* baseline whose accuracy cannot come from gaze/IMU at
all. The backward decoder is exactly that anchor: it only ever sees EEG and the
audio envelopes, never the orienting modalities.

## Key hyperparameters (`configs/model/linear_backward.yaml`)
- `n_lags` (16 @ 64 Hz ≈ 250 ms decoder window)
- `alpha` ridge regularisation
- `max_train_samples` cap for the closed-form ridge solve
