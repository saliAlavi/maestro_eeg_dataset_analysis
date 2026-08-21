#!/bin/bash
# K-shot LOSO calibration curve (fine-tune LOSO base on K held-out-subject trials). One subj/task.
#SBATCH --account=PAS2966
#SBATCH --job-name=kshot
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/kshot_%A_%a.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/kshot_%A_%a.err
#SBATCH --time=01:30:00
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
NSEED=5 $NIPS kshot_loso.py "$S"
