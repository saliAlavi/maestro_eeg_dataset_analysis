#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=iter6
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/iter6_s%a_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/iter6_s%a_%A.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --array=1-16

set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nips

S=${SLURM_ARRAY_TASK_ID}
EXP="${EXP:?set EXP: imu | video | clean_all}"

case "$EXP" in
    imu)
        python scripts/imu_features.py --subject $S \
            --out results/imu_aad/s${S}.parquet
        ;;
    video)
        python scripts/video_features.py --subject $S \
            --out results/video_aad/s${S}.parquet
        ;;
    clean_all)
        for COND in raw ica_eog regress_gaze regress_gaze_imu drop_eog_emg; do
            python scripts/eeg_clean_aad.py --subject $S --condition $COND \
                --out results/eeg_clean/s${S}_${COND}.parquet
        done
        ;;
    *) echo "unknown EXP=$EXP"; exit 1 ;;
esac
