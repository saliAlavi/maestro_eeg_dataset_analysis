#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=aad_loso
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/aad_loso_s%a_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/aad_loso_s%a_%A.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=1-16

set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nips

S=${SLURM_ARRAY_TASK_ID}
OUT=results/loso_cca_mel28/s${S}.parquet

python scripts/aad_loso.py --test-subject ${S} --n-components 5 --out ${OUT}
