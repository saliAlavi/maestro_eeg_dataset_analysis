#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=merit_prep2
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/merit_prep2_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/merit_prep2_%A.err
#SBATCH --time=04:00:00
#SBATCH --partition=nextgen
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
# Materialise the 2 s window cache for the short-window modality sweep.
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
source ~/miniconda3/etc/profile.d/conda.sh; conda activate nips
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
echo "### building four-modality windows W=2s H=1s ###"
$PY -m src.main mode=prepare data=merit model=merit runner=merit \
    wandb.mode=disabled data.window_sec=2 data.hop_sec=1
