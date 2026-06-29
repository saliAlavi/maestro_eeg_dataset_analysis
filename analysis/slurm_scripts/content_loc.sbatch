#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=cloc
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/cloc_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/cloc_%A.err
#SBATCH --time=04:00:00
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nips
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
echo "########## content-ONLY (no location) ##########"
CFEAT=spec USE_LOC=0 $PY scripts/content_loc.py
echo "########## content + LOCATION ##########"
CFEAT=spec USE_LOC=1 $PY scripts/content_loc.py
echo "########## content + LOCATION -- INPUT-SHUFFLE NULL ##########"
CFEAT=spec USE_LOC=1 SHUFFLE_INPUT=1 $PY scripts/content_loc.py
