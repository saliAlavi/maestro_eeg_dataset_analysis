#!/bin/bash
# GITHUB-METHOD LOSO (train_loso_hot-style: subject-disjoint, val==test, best-epoch,
# 30s windows) for the 4-class task — reproduces the repo's LOSO column.
# Array index = held-out TEST subject.
#SBATCH --account=PAS2966
#SBATCH --job-name=ngh_ghloso
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/ngh_ghloso_%A_%a.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/ngh_ghloso_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --partition=preemptible-nextgen
#SBATCH --requeue

set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/n_gh_checks
NIPS=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
# github-LOSO iterates ALL 16 held-out subjects internally in one pass, so this is
# a single job (not an array); --subject is required by the CLI but ignored here.
echo "host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null|head -1)"
$NIPS train_pooled.py --data-method github --protocol loso --subject 1 --epochs 50 --skip-existing
