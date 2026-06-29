#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=c2stage
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/c2stage_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/c2stage_%A.err
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

echo "########## REAL: content(env,spec,w2v,sem)+dir, avg fusion ##########"
FEATS=env,spec,w2v,sem,dir FUSE=avg $PY scripts/content_2stage.py

echo "########## CONTENT-ONLY (no dir): clean neural content number ##########"
FEATS=env,spec,w2v,sem FUSE=avg $PY scripts/content_2stage.py

echo "########## FULL-PIPELINE NULL (SHUFFLE_INPUT=1) -- must be chance ##########"
SHUFFLE_INPUT=1 FEATS=env,spec,w2v,sem,dir FUSE=avg $PY scripts/content_2stage.py
