#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --partition=nextgen
#SBATCH --gres=gpu:1
#SBATCH --job-name=shifted_tssl
#SBATCH --output=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/logs/shifted_tssl_%j.out
#SBATCH --error=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/logs/shifted_tssl_%j.err
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
set -euo pipefail
export PYTHONNOUSERSITE=1
export SEEDS=${SEEDS:-0,1,2}
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
echo "[shifted_tssl] $(date)"; nvidia-smi -L | head -1
$PY analysis/scripts/content_shifted_ssl.py
echo "[shifted_tssl] DONE $(date)"
