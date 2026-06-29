#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=cvisloso
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/cvisloso_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/cvisloso_%A.err
#SBATCH --time=16:00:00
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

# Fill the LOSO column of the results table (audio-window-trimmed V-JEPA cache). chance 0.25.
# Each run prints the grouped block: cfused (content) / sfused (orienting). The already-have LOSO
# run is "+VIS FOVEA_ONLY DIR=lda FUSE=rel" (content branches, gaze-LDA, video fovea, content-only,
# orienting+video, full+video-rel). These 5 add the remaining "--" cells.

echo "########## LOSO BASELINE (no video): DIR=lda FUSE=rel  -> orienting(gaze), full-no-video ##########"
PROTOCOL=loso DIRMODE=lda FUSE=rel $PY scripts/content_best.py

echo "########## LOSO gaze-MLP head: DIR=mlp FUSE=rel  -> Gaze MLP (spatial) ##########"
PROTOCOL=loso DIRMODE=mlp FUSE=rel $PY scripts/content_best.py

echo "########## LOSO +VIS scene+fovea: DIR=lda FUSE=rel  -> Video scene+fovea (spatial) ##########"
PROTOCOL=loso VIS=1 DIRMODE=lda FUSE=rel $PY scripts/content_best.py

echo "########## LOSO +VIS FOVEA_ONLY: DIR=lda FUSE=relv  -> Full +video (relv) ##########"
PROTOCOL=loso VIS=1 FOVEA_ONLY=1 DIRMODE=lda FUSE=relv $PY scripts/content_best.py

echo "########## LOSO NULL input-shuffle: +VIS FOVEA_ONLY +VALIGN  -> Null full / content-only ##########"
PROTOCOL=loso SHUFFLE_INPUT=1 VIS=1 FOVEA_ONLY=1 VALIGN=1 VALIGN_TGT=scene DIRMODE=lda FUSE=rel $PY scripts/content_best.py
