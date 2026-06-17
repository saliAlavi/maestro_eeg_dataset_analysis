#!/bin/bash
# Train one model across the configured subjects (within-subject CV + LOSO).
# The variable bits (account/partition/gres/job-name/output) are passed on the
# sbatch CLI by slurm/submit_all.sh; this body just needs $MODEL.
#
# Required env: MODEL (config name). Optional: SUBJECTS (default [1,2,3]),
#               WANDB_MODE (default offline).
set -euo pipefail
export PYTHONNOUSERSITE=1
export MNE_USE_NUMBA=false
export TOKENIZERS_PARALLELISM=false
export WANDB_DIR=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/runs
export WANDB_CACHE_DIR=/fs/scratch/PAS2301/alialavi/projects/cache/wandb
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python

MODEL=${MODEL:?set MODEL env var}
SUBJECTS=${SUBJECTS:-[1,2,3]}
WANDB_MODE=${WANDB_MODE:-offline}
EXTRA=${EXTRA:-}            # extra hydra overrides, e.g. "data.window.mm_task=shifted"

echo "[train] model=$MODEL subjects=$SUBJECTS extra='$EXTRA' host=$(hostname) $(date)"
echo "[train] python: $($PY --version)"
nvidia-smi -L 2>/dev/null | head -4 || echo "[train] no GPU visible (CPU model)"

$PY -m src.main mode=train model="$MODEL" \
    data.subjects="$SUBJECTS" wandb.mode="$WANDB_MODE" $EXTRA
echo "[train] model=$MODEL DONE $(date)"
