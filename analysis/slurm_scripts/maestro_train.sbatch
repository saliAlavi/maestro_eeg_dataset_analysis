#!/bin/bash
# Stage 2 (single GPU): train + evaluate MAESTRO-Net on the full corpus and
# emit the protocol table (T1-T3 x within/LOSO x leave-one-modality-out).
#   sbatch maestro_train.sbatch
#
# NOTE: confirm the GPU partition/account for your OSC allocation. On Cardinal/
# Pitzer this typically also needs `--partition=<gpu-partition>`; add it below.
#SBATCH --account=PAS2966
#SBATCH --job-name=maestro_train
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/maestro_train_%j.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/maestro_train_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1

set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nips

# Sanity: a CPU smoke test first proves the graph before burning GPU time.
python scripts/maestro_net.py --smoke

python scripts/maestro_net.py --train \
    --subjects $(seq 1 16) \
    --epochs 40 --win-s 5.0 \
    --out results/maestro/protocol_table.parquet
