#!/bin/bash
# Submit the full first-pass sweep: prepare (3 subjects) -> all 4 models in
# parallel, each depending on the prepare array. Prints job IDs and exits;
# nothing blocks. Classical models run CPU-only on nextgen; neural models grab
# one A100. wandb runs offline (sync later with: wandb sync <run-dir>).
set -euo pipefail
ROOT=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
LOG=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/logs
mkdir -p "$LOG"
ACC=PAS2966
PART=nextgen
cd "$ROOT"

PREP=$(sbatch --parsable --array=1-3 slurm/prepare.sbatch)
echo "PREPARE=$PREP"

submit_model () {
  local MODEL=$1 GRES=$2 TIME=$3 MMTASK=$4
  local extra=""
  [ -n "$GRES" ] && extra="--gres=$GRES"
  local EXP="WANDB_MODE=offline"
  [ -n "$MMTASK" ] && EXP="$EXP,EXTRA=data.window.mm_task=$MMTASK"
  sbatch --parsable \
    --account="$ACC" --partition="$PART" $extra \
    --dependency=afterok:"$PREP" --kill-on-invalid-dep=yes \
    --job-name="aad_$MODEL" \
    --time="$TIME" --cpus-per-task=8 --mem=64G \
    --output="$LOG/train_${MODEL}_%j.out" \
    --error="$LOG/train_${MODEL}_%j.err" \
    --export=ALL,MODEL="$MODEL",$EXP \
    slurm/train.sbatch
}

# classical decoders stay on the speaker task (they're EEG-mediated, not confounded);
# neural match-mismatch models use the EEG-honest shifted-negative task.
echo "linear_backward=$(submit_model linear_backward ''      02:00:00)"
echo "riemann_tangent=$(submit_model riemann_tangent ''      02:00:00)"
echo "eegnet_mm=$(submit_model eegnet_mm        gpu:1   06:00:00 shifted)"
echo "maestro=$(submit_model maestro            gpu:1   08:00:00 shifted)"
