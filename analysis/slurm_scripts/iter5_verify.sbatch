#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=iter5v
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/iter5v_s%a_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/iter5v_s%a_%A.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --array=1-16

set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nips

python scripts/eeg_spectral_features.py \
    --subject ${SLURM_ARRAY_TASK_ID} \
    --out results/eeg_spectral_verify/s${SLURM_ARRAY_TASK_ID}.parquet
