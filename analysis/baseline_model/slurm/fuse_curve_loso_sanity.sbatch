#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=fuseL_san
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/fuseL_san_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/fuseL_san_%A.err
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus-per-node=1
#SBATCH --partition=debug-nextgen
#SBATCH --requeue
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/baseline_model/models
NIPS=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
echo "host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null|head -1)"
$NIPS fuse_curve_loso.py 1
echo "FUSEL SANITY DONE"
