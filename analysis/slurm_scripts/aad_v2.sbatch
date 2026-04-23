#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=aad_v2
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/aad_v2_s%a_c${CORRECT:-0}.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/aad_v2_s%a_c${CORRECT:-0}.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=1-16

set -euo pipefail

cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nips

SUBJECT=${SLURM_ARRAY_TASK_ID}
CORRECT_FLAG=""
TAG="all"
if [[ "${CORRECT:-0}" == "1" ]]; then
    CORRECT_FLAG="--correct-only"
    TAG="correct"
fi

NO_ICA_FLAG=""
if [[ "${NO_ICA:-0}" == "1" ]]; then
    NO_ICA_FLAG="--no-ica"
    TAG="${TAG}_noica"
fi

TARGET="${TARGET_MODE:-derivative}"
OUT=results/aad_v2/s${SUBJECT}_${TAG}_${TARGET}.parquet

python scripts/aad_v2.py --subject ${SUBJECT} ${CORRECT_FLAG} ${NO_ICA_FLAG} \
    --target-mode ${TARGET} --out ${OUT}
