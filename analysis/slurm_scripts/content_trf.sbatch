#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=ctrf
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/ctrf_%x_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/ctrf_%x_%A.err
#SBATCH --time=06:00:00
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# $1 = band (broad|dt) ; $2 = tag ; $3 = shuffle_eeg (true|false, NULL control)
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nips
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
BAND="${1:-broad}"
TAG="${2:--broad}"
SHUF="${3:-false}"
export WANDB_MODE="${WANDB_MODE:-offline}"
echo "content_trf band=${BAND} tag=${TAG} shuffle_eeg=${SHUF}"
$PY -m src.main mode=train model=content_trf data=aad_multipath \
    runner.protocols=[within] runner.val_frac=0.2 \
    data.window.overlap=0.875 \
    model.band="${BAND}" model.tag="${TAG}" model.shuffle_eeg="${SHUF}"
