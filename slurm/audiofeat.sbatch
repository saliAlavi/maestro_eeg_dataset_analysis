#!/bin/bash
# Extract rich per-speaker audio features (HuBERT layer-9 + GPT-2 surprisal) for
# all 100 main trials, shared across subjects. Submit: sbatch slurm/audiofeat.sbatch
#SBATCH --account=PAS2966
#SBATCH --partition=nextgen
#SBATCH --gres=gpu:1
#SBATCH --job-name=audiofeat
#SBATCH --output=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/logs/audiofeat_%j.out
#SBATCH --error=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/logs/audiofeat_%j.err
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail
export PYTHONNOUSERSITE=1
export HF_HOME=/fs/scratch/PAS2301/alialavi/cache/hf
export TORCH_HOME=/users/PAS2301/alialavi/.cache/torch
export TOKENIZERS_PARALLELISM=false
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
echo "[audiofeat] host=$(hostname) $(date)"; nvidia-smi -L 2>/dev/null | head -1 || true
$PY -m src.data.audio_features --trials 1-100
echo "[audiofeat] DONE $(date)"
