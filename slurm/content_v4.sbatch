#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --partition=nextgen
#SBATCH --gres=gpu:1
#SBATCH --job-name=content_v4
#SBATCH --output=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/logs/content_v4_%j.out
#SBATCH --error=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/logs/content_v4_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
set -euo pipefail
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export SEEDS=${SEEDS:-0} SPACES=${SPACES:-spec,w2v,sem}
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
echo "[content_v4] $(date) SPACES=$SPACES"; nvidia-smi -L|head -1
$PY analysis/scripts/content_v4.py
echo "[content_v4] DONE $(date)"
