#!/bin/bash
# Strict-regime lever confirmation: env vs onset envelope target, content-disjoint within+loso,
# EEG-shuffle null. One subject per array task.
#SBATCH --account=PAS2966
#SBATCH --job-name=strimp
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/strimp_%A_%a.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/strimp_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus-per-node=1
#SBATCH --partition=preemptible-nextgen
#SBATCH --requeue
#SBATCH --array=1-16
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/baseline_model/models
NIPS=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
S=${SLURM_ARRAY_TASK_ID}
echo "host=$(hostname) subject=$S gpu=$(nvidia-smi -L 2>/dev/null|head -1)"
CONFIGS="env,onset" NSEED=5 PROTOCOL=within $NIPS strict_improve.py "$S"
CONFIGS="env,onset" NSEED=5 PROTOCOL=loso   $NIPS strict_improve.py "$S"
