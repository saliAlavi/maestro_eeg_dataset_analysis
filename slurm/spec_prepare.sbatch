#!/bin/bash
# Build + cache the spectral dataset variant (EEG broadband, alpha/beta kept),
# one subject per array task. Submit: sbatch slurm/spec_prepare.sbatch
#SBATCH --account=PAS2966
#SBATCH --partition=nextgen
#SBATCH --job-name=spec_prep
#SBATCH --array=1-16
#SBATCH --output=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/logs/spec_prep_s%a_%A.out
#SBATCH --error=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/logs/spec_prep_s%a_%A.err
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
echo "[spec_prepare] subject=$S host=$(hostname) $(date)"
$PY -m src.main mode=prepare data=aad_spec data.subjects=[$S] data.overwrite_cache=true
echo "[spec_prepare] subject=$S DONE $(date)"
