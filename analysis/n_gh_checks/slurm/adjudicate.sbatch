#!/bin/bash
# Eval-only: EEG-shuffle null + causal-lag controls on the strict-LOSO GitHub AADModel.
#SBATCH --account=PAS2966
#SBATCH --job-name=gh_adjud
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/gh_adjud_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/gh_adjud_%A.err
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --partition=debug-nextgen
#SBATCH --requeue
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/n_gh_checks
NIPS=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
echo "host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null|head -1)"
$NIPS adjudicate_gh.py
echo "GH ADJUDICATE DONE"
