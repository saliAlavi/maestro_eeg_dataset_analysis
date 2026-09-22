#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=merit_h2l
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/merit_h2l_%A_%a.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/merit_h2l_%A_%a.err
#SBATCH --time=16:00:00
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-1
# LOSO half of the clean hinge 2x2, split in two because all four arms over
# sixteen held-out listeners does not fit one walltime.
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
source ~/miniconda3/etc/profile.d/conda.sh; conda activate nips
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
CROP="data.window_sec=30 data.hop_sec=15 data.crop_len=640 data.n_train_crops=8 data.n_eval_crops=5"
$PY -m src.main data=merit model=merit wandb.mode=offline $CROP \
    runner=merit_hinge2loso${SLURM_ARRAY_TASK_ID} data.orient_feats=true \
    data.euclidean_align=true data.audio_feats=env_onset model.tag=__v2hinge2loso
