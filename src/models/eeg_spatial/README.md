# eeg_spatial — spectral/spatial (lateralisation) attended-speaker decoder

## What it is
An intra-subject decoder that classifies the attended speaker (1–4, i.e. fixed
azimuth/hemisphere) **directly from EEG band-power**, with no audio, no envelope,
no reconstruction. An EEGNet/CSP spectro-spatial encoder (learned temporal band
filters → per-band spatial filters → **log-variance pooling**) extracts
lateralised band-power features; a small MLP maps them to speaker logits; plain
cross-entropy on the attended label.

## Motivation / intuition — the mistake that led here
Envelope-tracking AAD is **dead on this corpus**. The classical CCA backward
models (aad_v2/v3 ≈ 0.50, aad_v4 hemi 0.505) and our own `recon_mm` (hemi 0.54)
are all at chance, because audio↔EEG alignment is **software-timestamped with no
hardware trigger** (~100–255 ms variable playback latency; per-trial lags
scattered std ~600 ms — see `analysis/scripts/diag_lag_jitter.py`). Stimulus
reconstruction needs <~50 ms precision, so it cannot work here.

What *does* survive timing jitter is **spatial attention**: attending a
side/azimuth modulates parieto-occipital **alpha (8–12 Hz) lateralisation**, a
power feature that needs no audio timing. `eeg_spatial` targets exactly that — the
only EEG mechanism shown to beat chance on this data (spectral baseline ~0.58).

## Why CSP log-variance (not envelope match)
Log-variance of spatially-filtered band signals = band-power per spatial filter =
the CSP feature that captures left-vs-right alpha asymmetry. It is timing-robust
by construction. **The cache must keep alpha** (`eeg_lp_hz=0`, not the 10 Hz
envelope band) and must **not** per-channel z-score (`norm_eeg=false`), or the L–R
power asymmetry — the signal itself — is erased.

## Leakage control
Intra-subject, trial-level `chrono_forward` CV (train strictly earlier than test).

## Config
`configs/model/eeg_spatial.yaml` (`F1`, `D`, `d_model`, `dropout`).
Data: `configs/data/aad_spec.yaml`. Smoke: `python -m src.main mode=selftest model=eeg_spatial`.
