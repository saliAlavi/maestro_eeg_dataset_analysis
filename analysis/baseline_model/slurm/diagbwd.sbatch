#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=ncl_diagbw
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/ncl_diagbw_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/ncl_diagbw_%A.err
#SBATCH --time=00:58:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --partition=debug-nextgen
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/baseline_model/models
NIPS=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
echo "host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null|head -1)"
echo "=== LINEAR backward, LOSO s1,s2,s3 (binary chance 0.5) ==="
for S in 1 2 3; do $NIPS train_bwd.py --model linear --protocol loso --subject $S --seeds 0 --epochs 50; done
echo "=== VLAAI backward, LOSO s1,s2 ==="
for S in 1 2; do $NIPS train_bwd.py --model vlaai --protocol loso --subject $S --seeds 0 --epochs 50; done
echo "NCL_DIAGBW_OK"
