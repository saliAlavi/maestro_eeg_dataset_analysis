#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=credit_loso
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/credit_loso_%A_%a.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/credit_loso_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-15
# Leave-one-listener-out: one array task per held-out listener, all seven stream
# sets inside it (they share the data build and the split).  The held-out
# listener is novel in identity AND in stimulus content.
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nips
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
F=${SLURM_ARRAY_TASK_ID}
$PY -m src.main data=merit model=merit runner=merit_streams wandb.mode=offline \
    runner.protocols=[loso] "runner.folds=[$F]" model.tag=__loso_f${F}
