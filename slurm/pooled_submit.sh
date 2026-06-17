#!/bin/bash
# Cross-subject POOLED training of aad_contrastive: one model trained on all 16
# subjects' (chronologically-early) trials pooled, tested on the held-out late
# trials. ~16x the per-subject data. Reuses the aad_recon cache via aad_match.
set -euo pipefail
ROOT=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
LOG=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/logs
mkdir -p "$LOG"
export SUBJECTS="[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]" WANDB_MODE=offline
export DATA=aad_match EXTRA="runner.protocols=[pooled]"
cd "$ROOT"

JID=$(sbatch --parsable --account=PAS2966 --partition=nextgen --gres=gpu:1 \
  --job-name=aadc_pooled --time=12:00:00 --cpus-per-task=8 --mem=80G \
  --output="$LOG/aadc_pooled_%j.out" --error="$LOG/aadc_pooled_%j.err" \
  --export=ALL,MODEL=aad_contrastive,DATA=aad_match,EXTRA="runner.protocols=[pooled]" \
  slurm/spec_train.sbatch)
echo "aad_contrastive(pooled, 16 subj)=$JID"
echo "monitor: squeue -u alialavi"
