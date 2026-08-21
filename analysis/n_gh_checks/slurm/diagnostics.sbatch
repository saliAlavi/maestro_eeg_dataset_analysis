#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=diag
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/diag_%x_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/diag_%x_%A.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=preemptible-nextgen
#SBATCH --gpus-per-node=1
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/n_gh_checks
NIPS=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
MODE=${MODE:-eeg}
WITHIN=${WITHIN:-1}
echo "=== diagnostics mode=$MODE within=$WITHIN ==="
$NIPS shuffle_diagnostics.py --mode "$MODE" --tag w10_h5 --within "$WITHIN"
echo "=== done ==="
