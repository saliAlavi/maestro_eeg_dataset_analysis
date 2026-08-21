#!/bin/bash
# Learned similarity head vs fixed correlation on OUR baseline, content-disjoint, within+loso,
# each readout with its EEG-shuffle null. One subject per array task.
#SBATCH --account=PAS2966
#SBATCH --job-name=lrnhead
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/lrnhead_%A_%a.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/lrnhead_%A_%a.err
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
PROTOCOL=within NSEED=5 $NIPS learned_head_ablation.py "$S"
PROTOCOL=loso   NSEED=5 $NIPS learned_head_ablation.py "$S"
