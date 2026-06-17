#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --partition=nextgen
#SBATCH --gres=gpu:1
#SBATCH --job-name=content_v5
#SBATCH --output=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/logs/content_v5_%j.out
#SBATCH --error=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/logs/content_v5_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
set -euo pipefail
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export SEEDS=${SEEDS:-0} SPACES=${SPACES:-spec,w2v,sem} PHASE=${PHASE:-1} CCA=${CCA:-1}
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
echo "[content_v5] $(date) SPACES=$SPACES PHASE=$PHASE CCA=$CCA"; nvidia-smi -L|head -1
$PY analysis/scripts/content_v5.py
echo "[content_v5] DONE $(date)"
