# Multimodal AAD analysis suite

Publication-ready analyses for the OSU multimodal auditory-attention-decoding
(AAD) dataset. Prepared for a NeurIPS Datasets & Benchmarks submission.

## Layout

- `aad_utils/` — shared loaders, alignment, preprocessing, features, plotting.
- `01_data_audit.ipynb` — modality presence matrix, sampling stability, gaze
  validity, wall-clock alignment spot-check, example aligned-trial timeline.
- `02_behavioral.ipynb` — comprehension accuracy vs SNR / direction / order /
  demographics; psychometric curve; mixed-effects model.
- `03_eeg_signal_quality.ipynb` — PSDs, 1/f slope, bad-channel detection,
  ICA with auto-EOG ID, gaze regression, audio-onset ERP, alpha lateralization.
- `04_gaze_analysis.ipynb` — I-VT saccade detection, binocular vergence, pupil
  trajectory, IMU head motion, gaze-vs-attended-direction.
- `05_audio_features.ipynb` — Hilbert/gammatone/mel envelopes, spatial binaural
  reconstruction at the listener, linguistic surprisal (guarded, `RUN_HEAVY`).
- `06_eeg_audio_decoding.ipynb` — backward mTRF, forward TRF + topo, CCA,
  window-length sweep, LOSO, EEGNet + cross-modal attention (`RUN_DEEP`).
- `07_gaze_AAD.ipynb` — gaze-only 4-way AAD with LogReg + LightGBM, LOSO,
  SHAP feature importance, deep TCN stub.
- `08_cross_modal_predictability.ipynb` — pairwise CCA, mutual information,
  Granger, transfer entropy, CKA matrices.
- `09_multimodal_fusion.ipynb` — early/late/stacked fusion, Shapley
  modality contribution, deep cross-modal transformer stub, learning curves.
- `10_publication_figures.ipynb` — consolidated NeurIPS-ready figures
  (reads parquet artefacts from 01–09).
- `11_scene_video.ipynb` — motion energy, dense optical flow, gaze-contingent
  patches, face detection, video ↔ EEG/env correlation.

## Runtime guards

- **`RUN_DEEP=False`** — deep models (EEGNet, cross-modal attention, gaze-TCN)
  are implemented but not executed on the CPU node. Flip to `True` on a GPU.
- **`RUN_HEAVY=False`** (nb05) — GPT-2 surprisal is stubbed; flip to download
  `distilgpt2` when needed.

## Suggested run order

1. `01_data_audit` → exports `results/01_presence.parquet` etc.
2. `02_behavioral`
3. `03_eeg_signal_quality`
4. `04_gaze_analysis`
5. `05_audio_features` (populates `cache/audio_features/`)
6. `06_eeg_audio_decoding` (uses 05 cache, populates `cache/aad_pairs/`)
7. `07_gaze_AAD` → `results/07_gaze_features.parquet`
8. `08_cross_modal_predictability`
9. `09_multimodal_fusion`
10. `11_scene_video`
11. `10_publication_figures` (reads everyone's artefacts)

## Expanding defaults

For speed, several notebooks run on a subset (first 3–4 subjects, first 10–30
main trials). When ready to produce full results, widen `SUBJECTS[:3]`,
`range(6, 16)`, etc. to the full set.

## Artefacts

- Parquet tables: `results/*.parquet`
- Figures: `figures/*.{pdf,png}` (300 DPI, vector PDFs, colorblind-safe)
- Caches: `cache/audio_features/*.npz`, `cache/aad_pairs/*.npz`
