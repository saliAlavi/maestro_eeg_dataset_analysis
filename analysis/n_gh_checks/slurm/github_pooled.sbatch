#!/bin/bash
# GITHUB-METHOD reproduction (pooled StratifiedKFold, val==test, best-epoch, 30s
# windows) — reproduces the repo's hemisphere/eccentricity/4-class/reconstruction
# headline numbers. Array index selects the task.
#SBATCH --account=PAS2966
#SBATCH --job-name=ngh_ghpool
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/ngh_ghpool_%A_%a.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/ngh_ghpool_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --partition=preemptible-nextgen
#SBATCH --requeue
#SBATCH --array=1-4

set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/n_gh_checks
NIPS=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
TASKS=(hemisphere eccentricity pooled reconstruction)   # script names
SCRIPT=${TASKS[$((SLURM_ARRAY_TASK_ID-1))]}
echo "host=$(hostname) task=$SCRIPT gpu=$(nvidia-smi -L 2>/dev/null|head -1)"
$NIPS train_${SCRIPT}.py --data-method github --protocol pooled --epochs 50 --skip-existing
# late fusion for the classification tasks (skips reconstruction)
if [ "$SCRIPT" != "reconstruction" ]; then
  $NIPS late_fusion.py --data-method github --protocol pooled || echo "WARN fusion"
fi
