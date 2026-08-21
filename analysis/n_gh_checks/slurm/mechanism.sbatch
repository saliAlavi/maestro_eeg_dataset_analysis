#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=mech
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/mech_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/mech_%A.err
#SBATCH --time=00:50:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --partition=debug-nextgen
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/n_gh_checks/fixed
/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python -u mechanism_trace.py
