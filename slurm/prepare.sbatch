#!/bin/bash
# Build + cache the windowed multimodal dataset, one subject per array task.
# Submit: sbatch --array=1-3 slurm/prepare.sbatch
#SBATCH --account=PAS2966
#SBATCH --partition=nextgen
#SBATCH --job-name=aad_prep
#SBATCH --output=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/logs/prep_s%a_%A.out
#SBATCH --error=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/logs/prep_s%a_%A.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

set -euo pipefail
export PYTHONNOUSERSITE=1
export MNE_USE_NUMBA=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python

S=${SLURM_ARRAY_TASK_ID}
echo "[prepare] subject=$S host=$(hostname) $(date)"
echo "[prepare] python: $($PY --version)"
$PY -m src.main mode=prepare data.subjects=[$S] data.overwrite_cache=true
echo "[prepare] subject=$S DONE $(date)"
