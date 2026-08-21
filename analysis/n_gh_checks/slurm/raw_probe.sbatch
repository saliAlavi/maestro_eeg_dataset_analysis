#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=rawprobe
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/rawprobe_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/rawprobe_%A.err
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=preemptible-nextgen
#SBATCH --gpus-per-node=1
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/n_gh_checks
/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python raw_probe.py
