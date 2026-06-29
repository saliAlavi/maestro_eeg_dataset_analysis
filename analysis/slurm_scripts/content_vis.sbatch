#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=cvis
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/cvis_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/cvis_%A.err
#SBATCH --time=10:00:00
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

# "What does video add to attended-speaker decoding?" -- re-run on the AUDIO-WINDOW-TRIMMED V-JEPA2
# cache, with the VALIGN fix (mean-centered target + warmup + low weight) and fovea-only ablations.
# Every run now also prints a GROUPED block: cfused (env/spec/sem ONLY = the honest content number,
# report under S1) vs sfused (dir/vis orienting = report under S3). chance = 0.25.
# Best validated content+orienting baseline (no video) = DIR=lda FUSE=rel.

echo "########## BASELINE (no video): DIR=lda FUSE=rel  [ref cfused~0.43, fused~0.616] ##########"
DIRMODE=lda FUSE=rel  $PY scripts/content_best.py

# ---- Tier 1: V-JEPA2 spatial/foveation branch (S3 orienting, NOT content) ----
echo "########## +VIS (scene+fovea) | DIR=lda FUSE=rel ##########"
VIS=1 DIRMODE=lda FUSE=rel  $PY scripts/content_best.py
echo "########## +VIS FOVEA_ONLY (drop near-constant scene half) | DIR=lda FUSE=rel ##########"
VIS=1 FOVEA_ONLY=1 DIRMODE=lda FUSE=rel  $PY scripts/content_best.py
echo "########## +VIS FOVEA_ONLY | DIR=lda FUSE=relv (validity-gated dir) ##########"
VIS=1 FOVEA_ONLY=1 DIRMODE=lda FUSE=relv $PY scripts/content_best.py

# ---- Tier 2: FIXED honest content test (video as a train-time regularizer, dropped at test) ----
echo "########## +VALIGN tgt=scene (FIXED: mean-center+warmup) | DIR=lda FUSE=rel ##########"
VALIGN=1 VALIGN_TGT=scene DIRMODE=lda FUSE=rel $PY scripts/content_best.py
echo "########## +VALIGN tgt=fovea (orienting target) | DIR=lda FUSE=rel ##########"
VALIGN=1 VALIGN_TGT=fovea DIRMODE=lda FUSE=rel $PY scripts/content_best.py

# ---- Both: fovea spatial branch + scene content regularizer ----
echo "########## +VIS FOVEA_ONLY +VALIGN tgt=scene | DIR=lda FUSE=rel ##########"
VIS=1 FOVEA_ONLY=1 VALIGN=1 VALIGN_TGT=scene DIRMODE=lda FUSE=rel $PY scripts/content_best.py

# ---- LOSO with the fovea vis branch (cross-subject; vis per-subject z-scored) ----
echo "########## LOSO +VIS FOVEA_ONLY | DIR=lda FUSE=rel ##########"
PROTOCOL=loso VIS=1 FOVEA_ONLY=1 DIRMODE=lda FUSE=rel $PY scripts/content_best.py

# ---- MANDATORY NULL: break EEG/gaze/video <-> label -> must collapse to ~chance (incl. cfused) ----
echo "########## NULL input-shuffle | +VIS FOVEA_ONLY +VALIGN | DIR=lda FUSE=rel ##########"
SHUFFLE_INPUT=1 VIS=1 FOVEA_ONLY=1 VALIGN=1 VALIGN_TGT=scene DIRMODE=lda FUSE=rel $PY scripts/content_best.py
