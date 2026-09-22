#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=merit_rel
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/merit_rel_%A_%a.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/merit_rel_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-2
# Retrain the models the paper calls MERIT with save_models on, so the weights
# can be released.  0: fused, within listeners; 1: fused, unseen listeners;
# 2: EEG-only, within listeners.  Same data settings as the reported runs.
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
source ~/miniconda3/etc/profile.d/conda.sh; conda activate nips
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
CROP="data.window_sec=30 data.hop_sec=15 data.crop_len=640 data.n_train_crops=8 data.n_eval_crops=5"
case $SLURM_ARRAY_TASK_ID in
  0) $PY -m src.main data=merit model=merit wandb.mode=offline $CROP runner=merit_release \
         data.euclidean_align=true data.audio_feats=env_onset model.tag=__release ;;
  1) $PY -m src.main data=merit model=merit wandb.mode=offline $CROP runner=merit_release \
         data.euclidean_align=true data.audio_feats=env_onset runner.protocols=[loso] \
         model.tag=__releaseloso ;;
  2) $PY -m src.main data=merit model=merit wandb.mode=offline $CROP runner=merit_release_eeg \
         data.audio_feats=env_onset model.tag=__releaseeeg ;;
esac
