#!/bin/bash
# Train one recon-family model (recon_mm | recon_mm_gaze) on the table-power /
# 10 Hz-band dataset, intra-subject CV. job-name/output/gres are set by the
# submit script; this body only needs $MODEL.
#
# Required env: MODEL. Optional: SUBJECTS (default [1,2,3]), WANDB_MODE (offline),
#               EXTRA (extra hydra overrides).
set -euo pipefail
export PYTHONNOUSERSITE=1
export MNE_USE_NUMBA=false
export TOKENIZERS_PARALLELISM=false
export WANDB_DIR=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/runs
export WANDB_CACHE_DIR=/fs/scratch/PAS2301/alialavi/projects/cache/wandb
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python

MODEL=${MODEL:?set MODEL env var (recon_mm | recon_mm_gaze)}
SUBJECTS=${SUBJECTS:-[1,2,3]}
WANDB_MODE=${WANDB_MODE:-offline}
EXTRA=${EXTRA:-}

echo "[recon_train] model=$MODEL subjects=$SUBJECTS extra='$EXTRA' host=$(hostname) $(date)"
echo "[recon_train] python: $($PY --version)"
nvidia-smi -L 2>/dev/null | head -4 || echo "[recon_train] no GPU visible"

$PY -m src.main mode=train data=aad_recon model="$MODEL" \
    data.subjects="$SUBJECTS" wandb.mode="$WANDB_MODE" $EXTRA
echo "[recon_train] model=$MODEL DONE $(date)"
