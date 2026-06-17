#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --partition=nextgen
#SBATCH --gres=gpu:1
#SBATCH --job-name=content_cca
#SBATCH --output=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/logs/content_cca_%j.out
#SBATCH --error=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/logs/content_cca_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
set -euo pipefail
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export MODEL=${MODEL:-cca} SHUFFLE_EEG=${SHUFFLE_EEG:-0}
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
echo "[content_cca] $(date) MODEL=$MODEL SHUFFLE_EEG=$SHUFFLE_EEG"; nvidia-smi -L|head -1
$PY analysis/scripts/content_cca.py
echo "[content_cca] DONE $(date)"
