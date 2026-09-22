#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=merit_ex5
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/merit_ex5_%A_%a.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/merit_ex5_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --array=0-3
# KU Leuven and DTU at a 5 s window, both splits, so the comparison with
# NeuroToken (which reports 5 s segments) is made at the same window length.
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
source ~/miniconda3/etc/profile.d/conda.sh; conda activate nips
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
C=(kul kul dtu dtu); P=(within loso within loso)
c=${C[$SLURM_ARRAY_TASK_ID]}; p=${P[$SLURM_ARRAY_TASK_ID]}
echo "########## corpus=$c protocol=$p window=5s ##########"
$PY -m src.main data=external model=merit runner=merit_external wandb.mode=offline \
    data.corpus=$c data.window_sec=5 data.hop_sec=2.5 runner.protocols=[$p] \
    model.tag=__ext${c}${p}5
