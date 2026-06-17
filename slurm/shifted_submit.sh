#!/bin/bash
# Confound-free EEG-tracking test: aad_contrastive on the SHIFTED match-mismatch task
# (same-talker time-shifted negatives), S1-3, within-subject. Eval self-reports the
# EEG-shuffle control (shuf_acc). Reuses the aad_recon cache via aad_shifted.
set -euo pipefail
ROOT=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
LOG=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/logs
mkdir -p "$LOG"
export SUBJECTS="[1,2,3]" WANDB_MODE=offline DATA=aad_shifted EXTRA="runner.protocols=[within]"
cd "$ROOT"
JID=$(sbatch --parsable --account=PAS2966 --partition=nextgen --gres=gpu:1 \
  --job-name=aadc_shift --time=06:00:00 --cpus-per-task=8 --mem=64G \
  --output="$LOG/aadc_shift_%j.out" --error="$LOG/aadc_shift_%j.err" \
  --export=ALL,MODEL=aad_contrastive,DATA=aad_shifted,EXTRA="runner.protocols=[within]" \
  slurm/spec_train.sbatch)
echo "aad_contrastive(shifted, S1-3, within)=$JID"
echo "monitor: squeue -u alialavi"
