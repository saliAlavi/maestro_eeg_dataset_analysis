#!/bin/bash
# Submit the recon-family experiment: prepare the table-power/10 Hz-band cache
# (S1-3 array) -> train BOTH models (recon_mm = EEG+audio, recon_mm_gaze =
# +gaze) on one GPU each, depending on the prepare array. Prints job IDs and
# exits. wandb runs offline (sync later: wandb sync <run-dir>).
set -euo pipefail
ROOT=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
LOG=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/logs
mkdir -p "$LOG"
ACC=PAS2966
PART=nextgen
SUBJECTS=${SUBJECTS:-[1,2,3]}
cd "$ROOT"

PREP=$(sbatch --parsable slurm/recon_prepare.sbatch)
echo "PREPARE(array 1-3)=$PREP"

submit_model () {
  local MODEL=$1 TIME=$2
  # NOTE: SUBJECTS/WANDB_MODE go through the *environment* (carried by --export=ALL),
  # never inside the --export list -- a list like [1,2,3] contains commas that
  # SLURM's --export parser would split on (truncating to '[1' and breaking Hydra).
  export SUBJECTS WANDB_MODE=offline
  sbatch --parsable \
    --account="$ACC" --partition="$PART" --gres=gpu:1 \
    --dependency=afterok:"$PREP" --kill-on-invalid-dep=yes \
    --job-name="recon_$MODEL" \
    --time="$TIME" --cpus-per-task=8 --mem=64G \
    --output="$LOG/recon_train_${MODEL}_%j.out" \
    --error="$LOG/recon_train_${MODEL}_%j.err" \
    --export=ALL,MODEL="$MODEL" \
    slurm/recon_train.sbatch
}

echo "recon_mm=$(submit_model recon_mm 06:00:00)"
echo "recon_mm_gaze=$(submit_model recon_mm_gaze 06:00:00)"
echo "submitted. monitor: squeue -u alialavi"
