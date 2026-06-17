#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --partition=nextgen
#SBATCH --gres=gpu:1
#SBATCH --job-name=mix_train
#SBATCH --output=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/logs/mix_train_%j.out
#SBATCH --error=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/logs/mix_train_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail
export PYTHONNOUSERSITE=1 MNE_USE_NUMBA=false TOKENIZERS_PARALLELISM=false
export HF_HOME=/fs/scratch/PAS2301/alialavi/cache/hf
export WANDB_DIR=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/runs
export WANDB_CACHE_DIR=/fs/scratch/PAS2301/alialavi/projects/cache/wandb
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
SUBJECTS=${SUBJECTS:-[1,2,3]}
echo "[mix_train] recon_mix subjects=$SUBJECTS $(date)"; nvidia-smi -L | head -1
$PY -m src.main mode=train data=aad_mix model=recon_mix data.subjects="$SUBJECTS" wandb.mode="${WANDB_MODE:-offline}"
echo "[mix_train] DONE $(date)"
