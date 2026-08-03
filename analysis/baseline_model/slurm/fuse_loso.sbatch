#!/bin/bash
# LOSO fusion (VLAAI content 5-seed+margin, posterior-alpha spatial). One held-out test subject/task.
#SBATCH --account=PAS2966
#SBATCH --job-name=fuse_loso
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/fuse_loso_%A_%a.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/fuse_loso_%A_%a.err
#SBATCH --time=01:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=88G
#SBATCH --gpus-per-node=1
#SBATCH --partition=preemptible-nextgen
#SBATCH --requeue
#SBATCH --array=1-16
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/baseline_model/models
NIPS=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
echo "host=$(hostname) test_subject=$SLURM_ARRAY_TASK_ID gpu=$(nvidia-smi -L 2>/dev/null|head -1)"
$NIPS fuse_loso_test.py "$SLURM_ARRAY_TASK_ID"
echo "FUSE LOSO s$SLURM_ARRAY_TASK_ID DONE"
