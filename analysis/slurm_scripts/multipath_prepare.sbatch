#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=mp_prep
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/mp_prep_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/mp_prep_%A.err
#SBATCH --time=04:00:00
#SBATCH --partition=nextgen
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nips
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python

# Build the broadband-EEG + audio-feature record cache for all 16 subjects ONCE
# (avoids the GPU jobs racing to build the same cache files).
$PY -m src.main mode=prepare data=aad_multipath
