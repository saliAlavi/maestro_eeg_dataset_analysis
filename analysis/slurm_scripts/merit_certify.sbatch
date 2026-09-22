#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=credit_certify
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/credit_certify_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/credit_certify_%A.err
#SBATCH --time=03:00:00
#SBATCH --partition=nextgen
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
# Model-independent acceptance test: what can a linear reader take from the
# candidate audio alone, for each of the four constructions?  Run before any
# training; a result is only interpretable against the floor of its own set.
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nips
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
$PY -m src.main mode=certify data=merit model=merit runner=merit wandb.mode=disabled
