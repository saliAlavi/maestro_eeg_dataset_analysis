#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=c2rel
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/c2rel_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/c2rel_%A.err
#SBATCH --time=04:00:00
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nips
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python

echo "########## REL FUSION: content+dir ##########"
FEATS=env,spec,w2v,sem,dir FUSE=rel $PY scripts/content_2stage.py
echo "########## REL FUSION: content-only ##########"
FEATS=env,spec,w2v,sem FUSE=rel $PY scripts/content_2stage.py
echo "########## REL FUSION: clean content (env,spec) + dir ##########"
FEATS=env,spec,dir FUSE=rel $PY scripts/content_2stage.py
