#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=ncl_diagb
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/ncl_diagb_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/ncl_diagb_%A.err
#SBATCH --time=00:55:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --partition=debug-nextgen
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/baseline_model/models
NIPS=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
echo "host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null|head -1)"
echo "=== BINARY match-mismatch (chance 0.5): loso s1,s2,s3 ==="
for S in 1 2 3; do $NIPS train.py --protocol loso --subject $S --seeds 0 1 --epochs 60 2>&1 | grep -E 'loso\|.*seed'; done
echo "NCL_DIAGB_OK"
