#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=gaze_video
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/gaze_video_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/gaze_video_%A.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --partition=debug-nextgen
#SBATCH --requeue
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/baseline_model/models
NIPS=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
echo "host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null|head -1)"
echo "=== GAZE -> DIRECTION ==="; $NIPS gaze_dir.py
echo "=== VIDEO -> DIRECTION + CONTENT ==="; $NIPS video_dir_content.py
echo "GAZE_VIDEO DONE"
