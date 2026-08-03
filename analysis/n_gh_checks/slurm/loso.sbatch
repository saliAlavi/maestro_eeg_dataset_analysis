#!/bin/bash
# n_gh_checks — leave-one-subject-out training. Array index = held-out TEST subject;
# trains on the other 15 subjects. Same task/mode matrix as within.sbatch.
#SBATCH --account=PAS2966
#SBATCH --job-name=ngh_loso
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/ngh_loso_%A_%a.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/ngh_loso_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --partition=nextgen
#SBATCH --array=1-16

set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/n_gh_checks
NIPS=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
S=${SLURM_ARRAY_TASK_ID}
echo "host=$(hostname) test_subject=$S gpu=$(nvidia-smi -L 2>/dev/null | head -1)"
$NIPS train_gh.py --protocol loso --subject "$S" --epochs 50 --skip-existing
