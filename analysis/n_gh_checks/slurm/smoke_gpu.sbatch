#!/bin/bash
# Fast GPU validation of the full path on a real compute node before the big arrays.
#SBATCH --account=PAS2966
#SBATCH --job-name=ngh_smoke
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/ngh_smoke_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/ngh_smoke_%A.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --partition=debug-nextgen

set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/n_gh_checks
NIPS=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
echo "host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null | head -1)"
# 1) full model+trainer smoke on GPU
$NIPS train_gh.py --smoke
# 2) one real within subject (fast on GPU) to exercise the deploy path + checkpoints
$NIPS train_gh.py --protocol within --subject 1 --epochs 8
echo "SMOKE_GPU_OK"
