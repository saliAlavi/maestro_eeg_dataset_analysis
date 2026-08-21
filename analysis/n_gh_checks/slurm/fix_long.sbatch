#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=fix_dbg
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/fix_dbg_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/fix_dbg_%A.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --gpus-per-node=1
#SBATCH --partition=preemptible-nextgen

set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/n_gh_checks/fixed
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
W=${WINDOW:-10}; H=${HOP:-5}; SPLIT=${SPLIT:-within}
echo "configs=${CONFIGS:-} tag=$TAG split=$SPLIT window=$W hop=$H"
if [ "${MODE:-ladder}" = "ridge" ]; then
  $PY -u ridge_baseline.py --split_setting "$SPLIT" --window_sec "$W" --hop_sec "$H" \
    --out /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/n_gh_checks/results/fixed/${TAG}.json
else
  $PY -u train_v2.py --configs "$CONFIGS" --split_setting "$SPLIT" \
    --folds "${FOLDS:-all}" --epochs "${EPOCHS:-50}" \
    --window_sec "$W" --hop_sec "$H" \
    --out /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/n_gh_checks/results/fixed/${TAG}.json
fi
