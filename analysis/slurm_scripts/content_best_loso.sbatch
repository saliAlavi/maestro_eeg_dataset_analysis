#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=cbloso
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/cbloso_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/cbloso_%A.err
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
# Leave-one-subject-out (train 15 subjects, test 1). Per-subject unsupervised gaze z-score
# (calibration) applied. Tests whether the intra-subject DIR=lda win survives cross-subject.

echo "########## LOSO | DIR=lda | FUSE=avg ##########"
PROTOCOL=loso DIRMODE=lda FUSE=avg  $PY scripts/content_best.py
echo "########## LOSO | DIR=mlp | FUSE=avg  (baseline dir) ##########"
PROTOCOL=loso DIRMODE=mlp FUSE=avg  $PY scripts/content_best.py
echo "########## LOSO | DIR=lda | FUSE=rel ##########"
PROTOCOL=loso DIRMODE=lda FUSE=rel  $PY scripts/content_best.py
echo "########## LOSO NULL input-shuffle | DIR=lda | FUSE=avg (must be ~chance) ##########"
PROTOCOL=loso SHUFFLE_INPUT=1 DIRMODE=lda FUSE=avg $PY scripts/content_best.py
