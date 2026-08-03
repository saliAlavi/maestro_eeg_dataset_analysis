#!/bin/bash
# LOSO on the preemptible partition — schedules on idle A100 cycles, bypassing the
# priority queue. --requeue + --skip-existing make preemption cheap: on restart the
# array task skips configs whose result JSON already exists and resumes.
#SBATCH --account=PAS2966
#SBATCH --job-name=ngh_loso_pe
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/ngh_loso_pe_%A_%a.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/ngh_loso_pe_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --partition=preemptible-nextgen
#SBATCH --requeue
#SBATCH --array=1-16

set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/n_gh_checks
NIPS=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
S=${SLURM_ARRAY_TASK_ID}
echo "host=$(hostname) test_subject=$S gpu=$(nvidia-smi -L 2>/dev/null | head -1)"
$NIPS train_gh.py --protocol loso --subject "$S" --epochs 50 --skip-existing
