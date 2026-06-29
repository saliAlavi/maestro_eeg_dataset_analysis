#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=c2pre
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/c2pre_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/c2pre_%A.err
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

# #2: cross-subject supervised pretrain of CONTENT branches + within-subject fine-tune,
# combined with #1 reliability-weighted fusion.
echo "########## PRETRAIN+REL: content+dir ##########"
PRETRAIN=1 FUSE=rel FEATS=env,spec,w2v,sem,dir $PY scripts/content_2stage.py
echo "########## PRETRAIN+REL: content-only (pure content lift) ##########"
PRETRAIN=1 FUSE=rel FEATS=env,spec,w2v,sem $PY scripts/content_2stage.py
