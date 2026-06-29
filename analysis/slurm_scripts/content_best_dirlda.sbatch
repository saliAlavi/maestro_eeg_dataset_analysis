#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=cbdirlda
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/cbdirlda_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/cbdirlda_%A.err
#SBATCH --time=03:00:00
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
# Reference (MLP dir): avg=0.563, rel=0.565. Diagnostic: gaze-only shrinkage-LDA dir = 0.559
# vs MLP dir 0.527 (+0.032). This A/B tests whether the better dir branch lifts the FUSED number.

echo "########## DIR=lda | FUSE=avg  (better dir + param-free fusion) ##########"
DIRMODE=lda FUSE=avg  $PY scripts/content_best.py
echo "########## DIR=lda | FUSE=rel  ##########"
DIRMODE=lda FUSE=rel  $PY scripts/content_best.py
echo "########## DIR=lda | FUSE=relv (validity-gated dir) ##########"
DIRMODE=lda FUSE=relv $PY scripts/content_best.py
echo "########## NULL input-shuffle | DIR=lda | FUSE=avg (must be ~chance) ##########"
SHUFFLE_INPUT=1 DIRMODE=lda FUSE=avg $PY scripts/content_best.py
