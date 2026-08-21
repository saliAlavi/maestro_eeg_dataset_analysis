#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=fix_smoke
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/fix_smoke_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/fix_smoke_%A.err
#SBATCH --time=00:55:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --partition=debug-nextgen

set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/n_gh_checks/fixed
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
nvidia-smi --query-gpu=name --format=csv,noheader || true
$PY -u train_v2.py --subjects 1,2,3,4 --configs A4,A5,A6,A5r,M-full \
  --folds 0 --epochs 3 --val_shuffles 2 --test_shuffles 5 \
  --out /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/n_gh_checks/results/fixed/smoke.json
