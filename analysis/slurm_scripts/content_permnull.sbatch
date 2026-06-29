#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=cpnull
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/cpnull_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/cpnull_%A.err
#SBATCH --time=12:00:00
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nips
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
export PYTHONNOUSERSITE=1

# Permutation NULL DISTRIBUTION (expensive: NPERM full within-subject passes with FUSE=rel). Replaces
# the old single-draw input-shuffle with a real null -> empirical one-sided p for fused/cfused/sfused.
# NPERM=50 -> min reportable p = 1/51 ~= 0.02.
echo "########## PERMUTATION NULL  +VIS DIR=lda FUSE=rel  NPERM=50 ##########"
PERM_NULL=1 NPERM=50 VIS=1 DIRMODE=lda FUSE=rel $PY scripts/content_best.py
