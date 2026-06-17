#!/bin/bash
# Submit source_hier (geometry-factorised 4-way source ID) one-fold quick eval on
# subjects 1-3. Reuses the EXISTING aad_spec cache (raw gaze cached; gaze trajectory
# computed at materialise time), so NO prepare job is needed. within protocol,
# fold 0 only (runner.max_folds=1) for a fast EEG-only vs EEG+gaze comparison.
set -euo pipefail
ROOT=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
LOG=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/logs
mkdir -p "$LOG"
export SUBJECTS=${SUBJECTS:-[1,2,3]} WANDB_MODE=${WANDB_MODE:-offline}
export MODEL=source_hier DATA=aad_source
export EXTRA=${EXTRA:-"runner.protocols=[within] runner.max_folds=1"}
cd "$ROOT"

ACCT=${ACCT:-PAS2966}
JID=$(sbatch --parsable \
  --account="$ACCT" --partition=nextgen --gres=gpu:1 \
  --job-name=source_hier --time=02:00:00 --cpus-per-task=8 --mem=64G \
  --output="$LOG/source_train_source_hier_%j.out" \
  --error="$LOG/source_train_source_hier_%j.err" \
  --export=ALL,MODEL,DATA,SUBJECTS,WANDB_MODE,EXTRA \
  slurm/spec_train.sbatch)
echo "source_hier=$JID  (acct=$ACCT, fold0, subjects=$SUBJECTS)"
echo "monitor: squeue -u alialavi ; logs: $LOG/source_train_source_hier_${JID}.out"
