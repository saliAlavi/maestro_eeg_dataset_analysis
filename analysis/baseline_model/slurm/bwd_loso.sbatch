#!/bin/bash
# Backward-model baselines (linear + VLAAI) — loso, one subject per array task.
#SBATCH --account=PAS2966
#SBATCH --job-name=bwd_loso
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/bwd_loso_%A_%a.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/bwd_loso_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --partition=preemptible-nextgen
#SBATCH --requeue
#SBATCH --array=1-16
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/baseline_model/models
NIPS=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
S=${SLURM_ARRAY_TASK_ID}
echo "host=$(hostname) subject=$S gpu=$(nvidia-smi -L 2>/dev/null|head -1)"
$NIPS train_bwd.py --model linear --protocol loso --subject "$S" --seeds 0 1 2 --skip-existing
$NIPS train_bwd.py --model vlaai  --protocol loso --subject "$S" --seeds 0 1 2 --skip-existing
