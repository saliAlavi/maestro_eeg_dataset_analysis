#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=credit_win
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/credit_win_%A_%a.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/credit_win_%A_%a.err
#SBATCH --time=16:00:00
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-4
# Decision-window sweep: contribution against window length, all stream sets.
# Needs credit_prepare.sbatch to have built the 5/15/20/30 s caches.
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nips
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
W=(5 10 15 20 30); H=(2.5 5 7.5 10 15)
w=${W[$SLURM_ARRAY_TASK_ID]}; h=${H[$SLURM_ARRAY_TASK_ID]}
$PY -m src.main data=merit model=merit runner=merit_windows wandb.mode=offline \
    data.window_sec=$w data.hop_sec=$h model.tag=__w${w}
