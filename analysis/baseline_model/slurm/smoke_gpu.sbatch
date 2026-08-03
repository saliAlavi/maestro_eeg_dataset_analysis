#!/bin/bash
# Fast GPU validation of NeuroCLIP-AAD before the arrays.
#SBATCH --account=PAS2966
#SBATCH --job-name=ncl_smoke
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/ncl_smoke_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/ncl_smoke_%A.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --partition=debug-nextgen

set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/baseline_model/models
NIPS=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
echo "host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null|head -1)"
$NIPS train.py --smoke
# one real within subject at full epochs (fast on GPU) to exercise the deploy path
$NIPS train.py --protocol within --subject 1 --seeds 0 --epochs 40
echo "NCL_SMOKE_OK"
