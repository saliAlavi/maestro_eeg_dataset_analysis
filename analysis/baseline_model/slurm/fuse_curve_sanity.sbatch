#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=fusewin_san
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/fusewin_san_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/fusewin_san_%A.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --partition=debug-nextgen
#SBATCH --requeue
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/baseline_model/models
NIPS=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
echo "host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null|head -1)"
$NIPS fuse_curve.py 1
echo "FUSEWIN SANITY DONE"
