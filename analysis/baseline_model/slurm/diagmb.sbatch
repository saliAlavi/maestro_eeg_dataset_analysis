#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=ncl_diagmb
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/ncl_diagmb_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/ncl_diagmb_%A.err
#SBATCH --time=00:58:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --partition=debug-nextgen
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/baseline_model/models
NIPS=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
echo "host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null|head -1)"
echo "=== VLAAI multi-band (28), LOSO s1,s2,s3 ==="
for S in 1 2 3; do $NIPS train_bwd.py --model vlaai --bands 28 --protocol loso --subject $S --seeds 0 1 2; done
echo "=== VLAAI multi-band + margin, LOSO s1,s2,s3 ==="
for S in 1 2 3; do $NIPS train_bwd.py --model vlaai --bands 28 --mm-margin 0.3 --protocol loso --subject $S --seeds 0 1 2; done
echo "NCL_DIAGMB_OK"
