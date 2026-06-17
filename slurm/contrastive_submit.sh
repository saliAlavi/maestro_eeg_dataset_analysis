#!/bin/bash
# Submit aad_contrastive (content-based contrastive EEG<->envelope source ID).
# Reuses the EXISTING 16-subject aad_recon cache via data=aad_match -> NO prepare.
set -euo pipefail
ROOT=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
LOG=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/logs
mkdir -p "$LOG"
export SUBJECTS=${SUBJECTS:-[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]} WANDB_MODE=offline
export DATA=aad_match EXTRA="runner.protocols=[within]"
cd "$ROOT"

JID=$(sbatch --parsable \
  --account=PAS2966 --partition=nextgen --gres=gpu:1 \
  --job-name=aad_contrastive --time=12:00:00 --cpus-per-task=8 --mem=64G \
  --output="$LOG/contrastive_aad_contrastive_%j.out" \
  --error="$LOG/contrastive_aad_contrastive_%j.err" \
  --export=ALL,MODEL=aad_contrastive,DATA=aad_match,EXTRA="runner.protocols=[within]" \
  slurm/spec_train.sbatch)
echo "aad_contrastive=$JID  (16 subj, within, aad_recon cache)"
echo "monitor: squeue -u alialavi"
