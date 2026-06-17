# MAESTRO — three-layer multimodal AAD framework

A clean, factory-driven framework for benchmarking auditory-attention-decoding
(AAD) models on the OSU multimodal corpus (EEG + gaze + IMU + video + audio).

## Architecture (three layers, constant interfaces)

```
src/
  common/      paths, registry/factory, logging, wandb, seeding
  data/        DATA LAYER   — download/preprocess/cache + the constant data view
  models/      MODEL LAYER  — one package per model: src/models/{name}/
  runner/      RUNNER LAYER — orchestrates data<->model, protocol sweep, agg, logs
  main.py      CLI entry point (Hydra)
configs/       Hydra config groups (data / model / runner)
slurm/         SLURM submit scripts (PAS2966, nextgen partition)
```

The runner only ever talks to `AbstractDataModule` and `AbstractModel`. A model
picks the data representation it needs from the **same** `AADView`:
`view.as_numpy()` (classical) or `view.as_torch_loader()` (neural). Changing how
data is cached, or adding a model, never touches the other layers.

### Adding a model
1. `src/models/{name}/model.py` — subclass `ClassicalModel` or `TorchModel`,
   decorate with `@MODEL_REGISTRY.register("{name}")`.
2. `src/models/{name}/README.md` — what it is, the intuition, and which earlier
   mistake motivated it.
3. `configs/model/{name}.yaml` — hyperparameters (`name: {name}` + a `train:` block
   for neural models).
4. Add `{name}` to `_PACKAGES` in `src/models/factory.py`. Done.

## Models
| name | layer | modalities | GPU | role |
|------|-------|-----------|-----|------|
| `linear_backward` | classical | EEG + audio | no | stimulus-reconstruction floor |
| `riemann_tangent` | classical | EEG | no | strong spatial-covariance baseline |
| `eegnet_mm`       | neural    | EEG + audio | yes | match-mismatch workhorse |
| `maestro`         | neural    | all 5 | yes | headline multimodal + leave-one-modality-out |

## Run
```bash
# build/cache the windowed dataset (per subject)
python -m src.main mode=prepare data.subjects=[1,2,3]

# train one model (within-subject CV + LOSO), logs to wandb, saves to scratch
python -m src.main mode=train model=maestro data.subjects=[1,2,3]

# full-stack CPU smoke on synthetic data (no corpus, wandb off)
python -m src.main mode=selftest
```

Outputs (scratch): per-split + aggregate parquet and `detail.json` under
`/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/runs/<run_name>/`;
checkpoints under `.../models/<run_name>/`. Run name = `multimodal_aad__{model}__{datetime}`.

## SLURM
```bash
bash slurm/submit_all.sh      # prepare(3 subjects) -> all 4 models in parallel
```
Account `PAS2966`, partition `nextgen`. wandb runs **offline** on compute nodes;
sync afterwards with `wandb sync <run-dir>`.

## Notes / caveats
- `aad_utils` path constants are stale (`audio_stimuli_data/` → `experiment_data/`);
  `data/aad_compat.py` patches them at runtime, non-invasively.
- Video is currently a zero context token (`present_video=0`) — the egocentric
  mp4 pipeline is expensive and mostly redundant with gaze/IMU; enable later.
- The central scientific tension (gaze ~0.77 > EEG) is built into the evaluation:
  `maestro` reports leave-one-modality-out and carries an adversarial gaze head.
