#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=merit_wm
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/merit_wm_%A_%a.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/merit_wm_%A_%a.err
#SBATCH --time=16:00:00
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --array=0-7
# Short-window modality sweep: eleven stream sets at W=5s and W=2s, within-subject,
# per-window decisions, 50% hop -- the same protocol as the w5..w30 sweep, so the
# rows extend that table downward and are comparable to the benchmark's per-window
# numbers.  Groups only exist so each array task fits its walltime; every group
# shares one data build and one set of folds, and all write the same model.tag so
# the rows merge into a single w2/w5 group in the macro generator.
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
source ~/miniconda3/etc/profile.d/conda.sh; conda activate nips
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
W=(5 5 5 5 2 2 2 2); H=(2.5 2.5 2.5 2.5 1 1 1 1); G=(0 1 2 3 0 1 2 3)
w=${W[$SLURM_ARRAY_TASK_ID]}; h=${H[$SLURM_ARRAY_TASK_ID]}; g=${G[$SLURM_ARRAY_TASK_ID]}
echo "### W=${w}s H=${h}s  modality group ${g} ###"
$PY -m src.main data=merit model=merit runner=merit_winmod${g} \
    wandb.mode=offline data.window_sec=$w data.hop_sec=$h model.tag=__w${w}
