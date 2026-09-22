#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=credit_smoke
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/credit_smoke_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/credit_smoke_%A.err
#SBATCH --time=01:00:00
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nips
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "### all seven stream variants, 4 listeners, 3 epochs, 1 fold ###"
time $PY -m src.main data=merit model=merit runner=merit_streams \
    wandb.mode=disabled runner.certify=false runner.max_folds=1 \
    'data.limit_subjects=[1,2,3,4]' data.limit_content=40 \
    model.train.epochs=3 model.train.val_shuffles=2 model.train.test_shuffles=5 \
    model.tag=__smoke
echo "### full-size timing probe: one variant, all listeners, 2 epochs ###"
time $PY -m src.main data=merit model=merit runner=merit \
    wandb.mode=disabled runner.certify=false runner.max_folds=1 \
    model.train.epochs=2 model.train.val_shuffles=2 model.train.test_shuffles=5 \
    model.tag=__timing
