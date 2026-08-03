#!/bin/bash
# n_gh_checks — within-subject (chrono-forward 5-fold) training, one subject/task.
# Trains all 9 classification singles (hemisphere/eccentricity/speaker4 x eeg/gaze/imu)
# + 7 reconstruction combos for the subject given by the array index.
#SBATCH --account=PAS2966
#SBATCH --job-name=ngh_within
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/ngh_within_%A_%a.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/ngh_within_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --partition=nextgen
#SBATCH --array=1-16

set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/n_gh_checks
NIPS=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
S=${SLURM_ARRAY_TASK_ID}
echo "host=$(hostname) subject=$S gpu=$(nvidia-smi -L 2>/dev/null | head -1)"
$NIPS train_gh.py --protocol within --subject "$S" --epochs 50
