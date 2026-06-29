#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=mp_train
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/mp_train_%x_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/mp_train_%x_%A.err
#SBATCH --time=08:00:00
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Parameterised by positional args (avoids --export comma-splitting on the path list):
#   $1 = PATHS -- hydra list of active paths, e.g. "[env,w2v,sem,dir]" or "[dir]"
#   $2 = TAG   -- run-name suffix, e.g. "-full" or "-dir"
# Usage: sbatch --job-name=mp_dir multipath_train.sbatch "[dir]" "-dir"
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nips
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python

PATHS="${1:-[env,w2v,sem,dir]}"
TAG="${2:--full}"
EXTRA="${3:-}"                               # optional extra hydra overrides
export WANDB_MODE="${WANDB_MODE:-offline}"   # compute nodes have no internet; sync after

echo "multipath_mm  paths=${PATHS}  tag=${TAG}  extra=${EXTRA}"
$PY -m src.main mode=train \
    model=multipath_mm data=aad_multipath \
    runner.protocols=[within] runner.val_frac=0.2 \
    model.paths="${PATHS}" model.tag="${TAG}" ${EXTRA}
