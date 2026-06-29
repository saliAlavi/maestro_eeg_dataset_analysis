# content_trf — lag-robust backward/TRF content decoder

## Why
The multipath content paths (env/w2v/sem) sit at chance because they score correlation over
5 s windows, assume a fixed EEG→audio lag (this corpus has per-trial software-sync jitter),
and use a pooled encoder that discards fine temporal structure. But content is **not** zero
here — a tuned backward model reached **0.60 binary** (attended vs unattended, t=3.79). This
model is built to capture that signal.

## What
- **TRF backward filter**: spatial mix → wide-lag temporal conv (~500 ms, `lag_taps`) →
  reconstructs the attended stimulus as **[broadband envelope, onset]** (onset = half-wave
  rectified envelope derivative; cortex tracks onsets well). Light + `weight_decay` ≈ ridge.
- **Train** on 5 s windows (sample count) with reconstruction (MSE + 1−corr) + match CE.
- **Score at the TRIAL level**: run the decoder over the full ~30 s trial and correlate the
  reconstruction with each candidate over the whole trial, with a **per-trial lag search**
  (max corr over ±`max_lag`) to absorb jitter. Decoupling scoring-window length from
  training-sample count is the key lever for envelope AAD.

Candidates are the 4 attendable talkers (permuted match task, loudness-equalised) → content
decision, no spatial/loudness shortcut.

## Metrics
Headline = trial-level **binary** (attended vs each unattended, chance 0.5) — comparable to the
0.60 backward benchmark — plus trial 4-class (chance 0.25) and hemisphere/inner-outer. `acc` in
per_split.parquet = trial 4-class; `binary` + trial collapses are in the run dir's `detail.json`.

## Levers exposed
`band` (broad vs delta-theta), `lag_taps` (TRF width), `max_lag` (jitter search), `w_recon`,
`weight_decay`. Run e.g.:
```bash
python -m src.main mode=train model=content_trf data=aad_multipath runner.protocols=[within] \
    runner.val_frac=0.2 model.band=broad model.tag=-broad
```
