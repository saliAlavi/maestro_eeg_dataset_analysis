#!/bin/bash
# Submit the spectral experiment: prepare broadband-EEG cache (S1-3 array) ->
# train eeg_spatial (EEG-only) and eeg_spatial_gaze (+gaze) on one GPU each.
# NOTE: comma-containing values (SUBJECTS list) go via the ENVIRONMENT, never the
# --export list (SLURM splits --export on commas).
set -euo pipefail
ROOT=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
LOG=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/logs
mkdir -p "$LOG"
ACC=PAS2966; PART=nextgen
export SUBJECTS=${SUBJECTS:-[1,2,3]} WANDB_MODE=offline
cd "$ROOT"

PREP=$(sbatch --parsable slurm/spec_prepare.sbatch)
echo "PREPARE(array 1-3)=$PREP"

submit_model () {
  local MODEL=$1 TIME=$2
  sbatch --parsable \
    --account="$ACC" --partition="$PART" --gres=gpu:1 \
    --dependency=afterok:"$PREP" --kill-on-invalid-dep=yes \
    --job-name="spec_$MODEL" --time="$TIME" --cpus-per-task=8 --mem=64G \
    --output="$LOG/spec_train_${MODEL}_%j.out" \
    --error="$LOG/spec_train_${MODEL}_%j.err" \
    --export=ALL,MODEL="$MODEL" \
    slurm/spec_train.sbatch
}

echo "eeg_spatial=$(submit_model eeg_spatial 04:00:00)"
echo "eeg_spatial_gaze=$(submit_model eeg_spatial_gaze 04:00:00)"
echo "submitted. monitor: squeue -u alialavi"
