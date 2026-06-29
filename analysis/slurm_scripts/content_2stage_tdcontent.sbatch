#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=c2tdc
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/c2tdc_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/c2tdc_%A.err
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
echo "########## TRIAL-DISJOINT PRETRAIN -- CONTENT-ONLY clean (env,spec,sem) REAL ##########"
PRETRAIN=1 FUSE=avg FEATS=env,spec,sem $PY scripts/content_2stage.py
echo "########## TRIAL-DISJOINT PRETRAIN -- CONTENT-ONLY clean (env,spec,sem) NULL ##########"
PRETRAIN=1 SHUFFLE_INPUT=1 FUSE=avg FEATS=env,spec,sem $PY scripts/content_2stage.py
echo "########## TRIAL-DISJOINT PRETRAIN -- CONTENT+w2v (env,spec,w2v,sem) REAL ##########"
PRETRAIN=1 FUSE=avg FEATS=env,spec,w2v,sem $PY scripts/content_2stage.py
