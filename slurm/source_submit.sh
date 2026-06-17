#!/bin/bash
# Submit source-identification training. Reuses the EXISTING aad_spec cache (raw
# gaze is already cached; the gaze trajectory is computed at materialise time), so
# NO prepare job is needed. Trains source_fusion (EEG + rich gaze, 4-way source ID)
# on one GPU. eeg_spatial is the EEG-only control (already trained).
set -euo pipefail
ROOT=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
LOG=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/logs
mkdir -p "$LOG"
export SUBJECTS=${SUBJECTS:-[1,2,3]} WANDB_MODE=offline
cd "$ROOT"

JID=$(sbatch --parsable \
  --account=PAS2966 --partition=nextgen --gres=gpu:1 \
  --job-name=source_fusion --time=04:00:00 --cpus-per-task=8 --mem=64G \
  --output="$LOG/source_train_source_fusion_%j.out" \
  --error="$LOG/source_train_source_fusion_%j.err" \
  --export=ALL,MODEL=source_fusion \
  slurm/spec_train.sbatch)
echo "source_fusion=$JID"
echo "monitor: squeue -u alialavi"
