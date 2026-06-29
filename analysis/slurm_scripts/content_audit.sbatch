#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=caudit
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/caudit_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/caudit_%A.err
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

# Audit-driven runs. Headline config = +VIS DIR=lda FUSE=rel. Every run now prints inferential stats
# (Wilcoxon vs chance + bootstrap 95% CI + paired Wilcoxon) and the cfused/sfused content/spatial split.
# NSEED=3 averages 3 outer-CV seeds so deltas <=0.02 aren't split noise.

echo "############################## P0: TRIAL-ORDER (S3) CONTROL ##############################"
echo "##### headline +VIS, raw (within-subject, T3 4-class) #####"
VIS=1 DIRMODE=lda FUSE=rel NSEED=3 $PY scripts/content_best.py
echo "##### +VIS, ORDER_RESID=1 -> the honest S3 motion/order-residualised orienting test #####"
VIS=1 DIRMODE=lda FUSE=rel NSEED=3 ORDER_RESID=1 $PY scripts/content_best.py
echo "##### LOSO +VIS, ORDER_RESID=1 #####"
PROTOCOL=loso VIS=1 DIRMODE=lda FUSE=rel ORDER_RESID=1 $PY scripts/content_best.py

echo "############################## P1: T1 (hemisphere) & T2 (inner/outer) ##############################"
echo "##### T1 hemisphere (chance 0.5) +VIS #####"
TASK=hemi  VIS=1 DIRMODE=lda FUSE=rel NSEED=3 $PY scripts/content_best.py
echo "##### T2 inner/outer (chance 0.5) +VIS #####"
TASK=inout VIS=1 DIRMODE=lda FUSE=rel NSEED=3 $PY scripts/content_best.py
echo "##### T1 hemisphere, ORDER_RESID (does orienting survive at the coarse task?) #####"
TASK=hemi  VIS=1 DIRMODE=lda FUSE=rel NSEED=3 ORDER_RESID=1 $PY scripts/content_best.py

# NOTE: the permutation-null distribution (expensive: NPERM x full rel) runs in content_permnull.sbatch.
