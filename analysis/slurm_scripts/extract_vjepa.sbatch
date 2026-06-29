#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=vjepa
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/vjepa_%A_%a.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/vjepa_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=1-16
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nips
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
export PYTHONNOUSERSITE=1
export HF_HOME=/fs/scratch/PAS2301/alialavi/cache/hf
export HF_HUB_DISABLE_TELEMETRY=1

S=${SLURM_ARRAY_TASK_ID}
echo "########## V-JEPA2 extraction subject $S (audio-window-trimmed) ##########"
$PY scripts/extract_vjepa.py --subject $S --model facebook/vjepa2-vitl-fpc64-256 --frames 64

echo "########## foveal-crop sanity montage subject $S ##########"
$PY scripts/vjepa_crop_check.py --subject $S --trials 1,25,50,75,100 --per-trial 3 || true
