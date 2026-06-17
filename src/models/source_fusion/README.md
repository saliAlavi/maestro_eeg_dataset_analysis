# source_fusion — full 4-way attended-source identification (EEG + rich gaze)

## What it is
Identifies **which of the 4 attended sources** (speakers 1–4) the listener attends
— the 4-class decision as the headline, not the left/right direction collapse.

## Why this design
On this corpus the 4 sources are 4 fixed azimuths whose **voices rotate every
trial**, so there is no consistent voice signature to learn — the decodable signal
is spatial. EEG alpha-lateralisation cleanly gives the **hemisphere** (left/right)
bit but is at chance for **inner/outer** (eccentricity isn't lateralised), which
caps EEG-only 4-class near hemisphere level. Resolving the two sources *within* a
hemisphere needs the finer **eye-position / azimuth** cue, which gaze carries.

So:
- **EEG** → EEGNet/CSP spectro-spatial log-power encoder (hemisphere bit).
- **Gaze** → summary stats **+ the raw subject-relative gaze trajectory** (`gaze_traj`,
  the azimuth cue; not z-scored so absolute position survives), MLP-encoded;
  zeroed and gaze-dropped when absent so the EEG path stays honest.
- **Joint (concat) fusion** → MLP → 4-way source logits, letting the head use
  EEG×gaze interactions (gaze resolves which source on the EEG-indicated side).
  4-class cross-entropy + label smoothing.

The raw-gaze trajectory is computed at **materialise time** from the cached raw
gaze, so it needs **no re-caching** — reuse the `aad_spec` cache. EEG-only control
is the separate `eeg_spatial` model; `evaluate()` also reports `all` vs `no_gaze`.

## Leakage control
Intra-subject, trial-level `chrono_forward` CV.

## Config
`configs/model/source_fusion.yaml` (`F1`, `D`, `gaze_dropout`, `label_smoothing`).
Data: `configs/data/aad_spec.yaml` (`gaze_traj_len`). Smoke:
`python -m src.main mode=selftest model=source_fusion`.
