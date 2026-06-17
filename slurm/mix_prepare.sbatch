#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --partition=nextgen
#SBATCH --job-name=mix_prep
#SBATCH --array=1-16
#SBATCH --output=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/logs/mix_prep_s%a_%A.out
#SBATCH --error=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/logs/mix_prep_s%a_%A.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
set -euo pipefail
export PYTHONNOUSERSITE=1 MNE_USE_NUMBA=false OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export HF_HOME=/fs/scratch/PAS2301/alialavi/cache/hf
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
S=${SLURM_ARRAY_TASK_ID}
echo "[mix_prepare] subject=$S $(date)"
$PY -m src.main mode=prepare data=aad_mix data.subjects=[$S] data.overwrite_cache=true
echo "[mix_prepare] subject=$S DONE $(date)"
