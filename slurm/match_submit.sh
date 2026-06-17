#!/bin/bash
# Submit lag-robust deep match-mismatch source ID. Reuses the EXISTING aad_recon
# cache (norm_eeg flipped at materialise time), so NO prepare job is needed.
set -euo pipefail
ROOT=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
LOG=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/logs
mkdir -p "$LOG"
export SUBJECTS=${SUBJECTS:-[1,2,3]} WANDB_MODE=offline DATA=aad_match
cd "$ROOT"

JID=$(sbatch --parsable \
  --account=PAS2966 --partition=nextgen --gres=gpu:1 \
  --job-name=deep_match --time=05:00:00 --cpus-per-task=8 --mem=64G \
  --output="$LOG/match_train_deep_match_%j.out" \
  --error="$LOG/match_train_deep_match_%j.err" \
  --export=ALL,MODEL=deep_match,DATA=aad_match \
  slurm/spec_train.sbatch)
echo "deep_match=$JID  (data=aad_match, reuses aad_recon cache)"
echo "monitor: squeue -u alialavi"
