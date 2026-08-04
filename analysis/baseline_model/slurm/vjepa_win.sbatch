#!/bin/bash
# Windowed (5 s) V-JEPA2 scene+fovea extraction, one subject per array task.
#SBATCH --account=PAS2966
#SBATCH --job-name=vjepaW
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/vjepaW_%A_%a.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/vjepaW_%A_%a.err
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --partition=preemptible-nextgen
#SBATCH --requeue
#SBATCH --array=1-16
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis
NIPS=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
echo "host=$(hostname) subject=$SLURM_ARRAY_TASK_ID gpu=$(nvidia-smi -L 2>/dev/null|head -1)"
$NIPS scripts/extract_vjepa_windowed.py --subject "$SLURM_ARRAY_TASK_ID"
echo "VJEPAW s$SLURM_ARRAY_TASK_ID DONE"
