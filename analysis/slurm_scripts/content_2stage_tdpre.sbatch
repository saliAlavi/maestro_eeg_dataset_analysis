#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=c2td
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/c2td_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/c2td_%A.err
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
echo "########## TRIAL-DISJOINT PRETRAIN -- REAL (avg fusion) ##########"
PRETRAIN=1 FUSE=avg FEATS=env,spec,w2v,sem,dir $PY scripts/content_2stage.py
echo "########## TRIAL-DISJOINT PRETRAIN -- NULL (SHUFFLE_INPUT) ##########"
PRETRAIN=1 SHUFFLE_INPUT=1 FUSE=avg FEATS=env,spec,w2v,sem,dir $PY scripts/content_2stage.py
