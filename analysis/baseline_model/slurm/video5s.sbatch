#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=video5s
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/video5s_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/video5s_%A.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --partition=debug-nextgen
#SBATCH --requeue
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/baseline_model/models
NIPS=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
echo "host=$(hostname)"
echo "=== VIDEO 5s (windowed) ==="; $NIPS video_dir_content.py --win5
echo "=== VIDEO whole (re-confirm) ==="; $NIPS video_dir_content.py
echo "VIDEO5S DONE"
