#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=iter4
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/iter4_s%a_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/iter4_s%a_%A.err
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=1-16

set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nips

S=${SLURM_ARRAY_TASK_ID}
EXP="${EXP:?set EXP env var: 4class | predictability | gaze_residualised}"

case "$EXP" in
    4class)
        python scripts/aad_v4_4class.py --subject $S \
            --out results/aad_v4_4class/s${S}.parquet
        ;;
    predictability)
        python scripts/eeg_gaze_predictability.py --subject $S \
            --out results/predictability/s${S}.parquet
        ;;
    gaze_residualised)
        python scripts/aad_gaze_residualized.py --subject $S \
            --out results/gaze_residualised/s${S}.parquet
        ;;
    *)
        echo "unknown EXP=$EXP"; exit 1 ;;
esac
