#!/bin/bash
# Stage 1 (CPU array): precompute frozen-encoder + flow video embeddings.
# The image encoder is frozen, so this is a one-off pass cached to
# results/video_embeddings/. Run before maestro_train.sbatch.
#   sbatch maestro_video.sbatch
#SBATCH --account=PAS2966
#SBATCH --job-name=maestro_vid
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/maestro_vid_s%a_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/maestro_vid_s%a_%A.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --array=1-16

set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nips

S=${SLURM_ARRAY_TASK_ID}
BACKBONE="${BACKBONE:-resnet18}"   # resnet18 | clip | none
python scripts/video_embeddings.py --subject $S --backbone "$BACKBONE" \
    --gaze-crop --win-s 5.0 --hop-s 1.0 --video-dim 16
