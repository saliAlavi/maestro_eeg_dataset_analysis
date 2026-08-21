#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=fix_ladder
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/fix_ladder_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/fix_ladder_%A.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus-per-node=1
#SBATCH --partition=preemptible-nextgen

set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/n_gh_checks/fixed

PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
CONFIGS=${CONFIGS:-A0,A0b,A1,A2,A3,A4,A5}
SPLIT=${SPLIT:-within}
FOLDS=${FOLDS:-all}
EPOCHS=${EPOCHS:-50}
TAG=${TAG:-ladder}

echo "configs=$CONFIGS split=$SPLIT folds=$FOLDS epochs=$EPOCHS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

$PY -u train_v2.py \
  --configs "$CONFIGS" \
  --split_setting "$SPLIT" \
  --folds "$FOLDS" \
  --epochs "$EPOCHS" \
  --out /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/n_gh_checks/results/fixed/${TAG}_${SPLIT}.json
