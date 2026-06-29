#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=cbest
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/cbest_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/cbest_%A.err
#SBATCH --time=08:00:00
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

# baselines for reference (validated): avg=0.563, rel=0.565, null(avg)=0.282
echo "########## content_best REAL -- avg fusion (reference baseline) ##########"
FUSE=avg $PY scripts/content_best.py

# ===================== FUSION-ONLY GAINS (baseline FeatNet encoder) =====================
echo "########## FeatNet | FUSE=stack  (fusion-only gain) ##########"
FUSE=stack $PY scripts/content_best.py
echo "########## FeatNet | FUSE=relv   (validity-gated dir) ##########"
FUSE=relv  $PY scripts/content_best.py

# ===================== ENCODER GAIN (multi-scale + recon aux) ===========================
echo "########## MS+recon | FUSE=avg   (encoder-only gain vs avg=0.563) ##########"
MSREC=1 FUSE=avg   $PY scripts/content_best.py
echo "########## MS+recon | FUSE=relv  ##########"
MSREC=1 FUSE=relv  $PY scripts/content_best.py
echo "########## MS+recon | FUSE=stack (combined best candidate) ##########"
MSREC=1 FUSE=stack $PY scripts/content_best.py

# ===================== MANDATORY NULL (everything on -> must be ~chance) =================
echo "########## NULL input-shuffle | MS+recon | FUSE=stack ##########"
SHUFFLE_INPUT=1 MSREC=1 FUSE=stack $PY scripts/content_best.py
