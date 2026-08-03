#!/bin/bash
# WITHIN-subject four-way decision-window curve (train 5s, eval {5,10,15,20,30}s, 5-seed
# reconstruction-ensemble, per-subject 5-fold pooled). Multiband, NO margin (margin overfits
# the data-starved within folds). One subject per array task.
#SBATCH --account=PAS2966
#SBATCH --job-name=curveW
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/curveW_%A_%a.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/curveW_%A_%a.err
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --partition=preemptible-nextgen
#SBATCH --requeue
#SBATCH --array=1-16
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/baseline_model/models
NIPS=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
S=${SLURM_ARRAY_TASK_ID}
echo "host=$(hostname) subject=$S gpu=$(nvidia-smi -L 2>/dev/null | head -1)"
$NIPS train_bwd.py --model vlaai --bands 28 --curve --protocol within --n-seeds 5 --subject "$S" --skip-existing
