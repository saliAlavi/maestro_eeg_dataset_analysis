# asad_mm — EEG + gaze attended-source detector (publication multimodal net)

## What it is
The same hardened EEG backbone as `asad_eeg` (multi-scale CSP + SE attention +
augmentation + multi-task aux), plus the **gaze fusion** the evidence on this corpus
supports:

- **Per-sample learned gaze reliability gate.** Cached-data probes show gaze quality
  is wildly heterogeneous across subjects — some carry full azimuth, some only
  hemisphere, some are dead (no overt orienting) — and it varies window-to-window. A
  scalar `r = sigmoid(head(gaze_emb)) ∈ [0,1]` scales the gaze contribution on top of
  the presence/dropout gate, so a dead-gaze window is down-weighted and a clean one
  trusted. (This is what let `source_rel` match/beat the flat fusion without
  regressing.)
- **EEG auxiliary hemisphere/eccentricity supervision** off the EEG embedding, so the
  EEG-only (`no_gaze`) path is supervised on both geometric axes.

## EEG-only vs EEG+gaze
`evaluate()` reports `all` (EEG+gaze) and `no_gaze` (EEG-only, gaze precision gated to
0). The `no_gaze` number is the exact gaze-ablation of *this* model, so the
EEG-only vs EEG+gaze comparison is apples-to-apples. (The dedicated EEG-only model is
`asad_eeg`.)

## Training
Within-subject `chrono_forward` CV with a chronological validation tail
(`runner.val_frac=0.15`) for best-model selection; all 5 folds. Gaze is presence-gated
with train-time `gaze_dropout=0.2` so the EEG path stays honest.

## Config / run
`configs/model/asad_mm.yaml`. Smoke:
`python -m src.main mode=selftest model=asad_mm data=aad_source`.
Train S1–3: `... mode=train model=asad_mm data=aad_source data.subjects=[1,2,3]
runner.protocols=[within] runner.val_frac=0.15`.
