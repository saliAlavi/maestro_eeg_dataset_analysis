#!/bin/bash
# Train one spectral model (eeg_spatial | eeg_spatial_gaze) on the broadband-EEG
# spectral dataset, intra-subject CV. job-name/output/gres set by submit script.
# Required env: MODEL. Optional: SUBJECTS (default [1,2,3]), WANDB_MODE (offline).
set -euo pipefail
export PYTHONNOUSERSITE=1
export MNE_USE_NUMBA=false
export TOKENIZERS_PARALLELISM=false
export WANDB_DIR=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/runs
export WANDB_CACHE_DIR=/fs/scratch/PAS2301/alialavi/projects/cache/wandb
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python

MODEL=${MODEL:?set MODEL env var (eeg_spatial | eeg_spatial_gaze | source_fusion | deep_match)}
DATA=${DATA:-aad_spec}
SUBJECTS=${SUBJECTS:-[1,2,3]}
WANDB_MODE=${WANDB_MODE:-offline}
EXTRA=${EXTRA:-}

echo "[spec_train] model=$MODEL data=$DATA subjects=$SUBJECTS extra='$EXTRA' host=$(hostname) $(date)"
nvidia-smi -L 2>/dev/null | head -4 || echo "[spec_train] no GPU visible"

$PY -m src.main mode=train data="$DATA" model="$MODEL" \
    data.subjects="$SUBJECTS" wandb.mode="$WANDB_MODE" $EXTRA
echo "[spec_train] model=$MODEL DONE $(date)"
