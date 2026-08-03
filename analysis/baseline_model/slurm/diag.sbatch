#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=ncl_diag
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/ncl_diag_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/ncl_diag_%A.err
#SBATCH --time=00:55:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --partition=debug-nextgen
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/baseline_model/models
NIPS=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
echo "host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null|head -1)"
echo "=== content-disjoint LOSO s1 (null MUST drop to ~0.25 if fix works) ==="
$NIPS train.py --protocol loso --subject 1 --seeds 0 --epochs 30
echo "=== within s1 (confirm clean) ==="
$NIPS train.py --protocol within --subject 1 --seeds 0 --epochs 30
echo "NCL_DIAG_OK"
