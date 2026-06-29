#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=cmsctrl
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/cmsctrl_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/cmsctrl_%A.err
#SBATCH --time=00:40:00
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

# CONTROL: MS+recon encoder, NO video alignment (VALIGN off). Isolates whether the content drop
# (cfused 0.44 -> 0.31 under VALIGN) is the ENCODER SWAP vs the ALIGN LOSS. If this MSREC-only
# cfused ~= the +VALIGN cfused (~0.31), the align loss is confirmed NEUTRAL for content. chance 0.25.
echo "########## MSREC-only control (no VALIGN): DIR=lda FUSE=rel ##########"
MSREC=1 DIRMODE=lda FUSE=rel $PY scripts/content_best.py
