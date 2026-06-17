#!/bin/bash
# Submit source_net (unified improved 4-way source ID). Reuses the EXISTING
# aad_spec cache via data=aad_source (gaze_traj_len bumped at materialise time),
# so NO prepare job is needed.
set -euo pipefail
ROOT=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
LOG=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/logs
mkdir -p "$LOG"
export SUBJECTS=${SUBJECTS:-[1,2,3]} WANDB_MODE=offline DATA=aad_source
cd "$ROOT"

JID=$(sbatch --parsable \
  --account=PAS2966 --partition=nextgen --gres=gpu:1 \
  --job-name=source_net --time=06:00:00 --cpus-per-task=8 --mem=64G \
  --output="$LOG/sourcenet_train_source_net_%j.out" \
  --error="$LOG/sourcenet_train_source_net_%j.err" \
  --export=ALL,MODEL=source_net,DATA=aad_source \
  slurm/spec_train.sbatch)
echo "source_net=$JID  (data=aad_source, reuses aad_spec cache)"
echo "monitor: squeue -u alialavi"
