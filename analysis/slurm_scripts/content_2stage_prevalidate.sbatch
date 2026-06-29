#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=c2pv
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/c2pv_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/c2pv_%A.err
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

echo "########## PRETRAIN NULL (SHUFFLE_INPUT) -- does pretrain inflate w2v null? ##########"
PRETRAIN=1 SHUFFLE_INPUT=1 FUSE=rel FEATS=env,spec,w2v,sem,dir $PY scripts/content_2stage.py
echo "########## PRETRAIN REAL, NO w2v: clean content+dir ##########"
PRETRAIN=1 FUSE=rel FEATS=env,spec,sem,dir $PY scripts/content_2stage.py
echo "########## PRETRAIN REAL, NO w2v: clean content-only ##########"
PRETRAIN=1 FUSE=rel FEATS=env,spec,sem $PY scripts/content_2stage.py
