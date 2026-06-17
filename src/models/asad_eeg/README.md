# asad_eeg — EEG-only attended-source detector (publication backbone)

## What it is
A well-regularised, **multi-task spectro-spatial CNN** for 4-way attended-source
detection from EEG alone. This is the architecture the evidence on this corpus
supports — not a transformer.

- **Backbone** (`asad_common.EEGBackbone`): multi-scale CSP front-end
  (`SpectralSpatialEncoder` at temporal kernels {15,33,65} → α/β band-power via
  log-variance pooling) + **SE channel-attention** recalibration → `d_model` embedding.
- **Regularisation**: train-time **EEG augmentation** (per-sample contiguous time
  mask, channel mask, std-scaled Gaussian noise) + dropout + weight decay + label
  smoothing.
- **Heads**: speaker (4-way, main CE) + **hemisphere** + **inner/outer** (auxiliary
  BCE). The aux heads supervise the shared embedding on the two geometric axes
  (EEG decodes hemisphere ~0.75 well; inner/outer is the hard bit), which is what
  lifted the EEG-only path in prior iterations.

## Why this design (and not a bigger one)
Across this project, every attempt to add capacity or structural priors lost to a
plain flat spectro-spatial net on ~750 within-subject windows (see
`source_hier`, `source_azi`); only auxiliary supervision helped (`source_rel`).
So the top-notch move here is **regularisation + multi-task supervision + augmentation
+ proper validation**, not architectural size. The signal is α/β lateralisation, which
a compact CSP front-end captures directly.

## Training
Within-subject `chrono_forward` CV. The runner carves a **chronological validation
tail** (`runner.val_frac=0.15`) from train for **best-model selection** (the model
used at test is the best-on-val epoch, not the last). All 5 folds.

## Config / run
`configs/model/asad_eeg.yaml`. Smoke:
`python -m src.main mode=selftest model=asad_eeg data=aad_source`.
Train S1–3: `... mode=train model=asad_eeg data=aad_source data.subjects=[1,2,3]
runner.protocols=[within] runner.val_frac=0.15`.
