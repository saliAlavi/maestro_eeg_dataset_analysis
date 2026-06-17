#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --partition=nextgen
#SBATCH --gres=gpu:1
#SBATCH --job-name=transfer_ssl
#SBATCH --output=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/logs/transfer_ssl_%j.out
#SBATCH --error=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/logs/transfer_ssl_%j.err
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
set -euo pipefail
export PYTHONNOUSERSITE=1
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
echo "[transfer_ssl] $(date)"; nvidia-smi -L | head -1
export SEEDS=${SEEDS:-0,1,2}
$PY analysis/scripts/transfer_ssl.py
echo "[transfer_ssl] DONE $(date)"
