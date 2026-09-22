#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=credit_prep
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/credit_prep_%A_%a.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/credit_prep_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --partition=nextgen
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-3
# Materialise the four-modality window caches for the decision-window sweep.
# W=10 already exists; this builds 5/15/20/30 s at 50% hop.
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nips
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
W=(5 15 20 30); H=(2.5 7.5 10 15)
w=${W[$SLURM_ARRAY_TASK_ID]}; h=${H[$SLURM_ARRAY_TASK_ID]}
echo "### building four-modality windows W=${w}s H=${h}s ###"
$PY -m src.main mode=prepare data=merit model=merit runner=merit \
    wandb.mode=disabled data.window_sec=$w data.hop_sec=$h
