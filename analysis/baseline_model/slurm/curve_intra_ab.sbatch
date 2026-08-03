#!/bin/bash
# A/B of "smarter network" levers to push WITHIN-subject 4-way at 5 s (no window growth, no
# reserved lever). One config per array task, screened on subjects 1-4 (5-fold, 5-seed ensemble,
# eval fixed at 5 s / 0.5 overlap). Read the [curve|within|sN|w5] lines to compare.
#SBATCH --account=PAS2966
#SBATCH --job-name=curveW_ab
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/curveW_ab_%A_%a.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/curveW_ab_%A_%a.err
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --partition=preemptible-nextgen
#SBATCH --requeue
#SBATCH --array=1-6
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/baseline_model/models
NIPS=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python

declare -A CFG
CFG[1]="--model vlaai  --bands 28 --overlap 0.875"                                      # denser windows
CFG[2]="--model vlaai  --bands 28 --aug-chdrop 0.15 --aug-noise 0.10"                   # augmentation
CFG[3]="--model vlaai  --bands 28 --overlap 0.875 --aug-chdrop 0.15 --aug-noise 0.10"   # both
CFG[4]="--model vlaai2 --bands 28"                                                      # new encoder alone
CFG[5]="--model vlaai2 --bands 28 --overlap 0.875 --aug-chdrop 0.15 --aug-noise 0.10"   # new encoder + both
CFG[6]="--model vlaai2 --bands 28 --hidden 64 --overlap 0.875 --aug-chdrop 0.15 --aug-noise 0.10"  # smaller encoder + both

C="${CFG[$SLURM_ARRAY_TASK_ID]}"
echo "host=$(hostname) cfg=[$C] gpu=$(nvidia-smi -L 2>/dev/null | head -1)"
for S in 1 2 3 4; do
  $NIPS train_bwd.py $C --curve --protocol within --dec-wins 5 --n-seeds 5 --subject "$S" --skip-existing
done
echo "AB CONFIG $SLURM_ARRAY_TASK_ID DONE"
