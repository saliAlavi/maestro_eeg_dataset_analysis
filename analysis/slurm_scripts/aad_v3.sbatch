#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=aad_v3
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/aad_v3_s%a_${FEATURES:-broadband}.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/aad_v3_s%a_${FEATURES:-broadband}.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=1-16

set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nips

FEATURES="${FEATURES:-broadband}"
TARGET="${TARGET:-derivative}"
OUT=results/aad_v3/s${SLURM_ARRAY_TASK_ID}_${FEATURES}_${TARGET}.parquet

python scripts/aad_v3.py --subject ${SLURM_ARRAY_TASK_ID} \
    --features ${FEATURES} --target ${TARGET} --out ${OUT}
