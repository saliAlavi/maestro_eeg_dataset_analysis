#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=ncl_sweep
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/ncl_sweep_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/ncl_sweep_%A.err
#SBATCH --time=00:58:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --partition=debug-nextgen
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/baseline_model/models
NIPS=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
echo "host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null|head -1)"
B="--model vlaai --bands 28 --mm-margin 0.3 --protocol loso"
for S in 1 2; do
  echo "== win8 s$S =="; $NIPS train_bwd.py $B --win-s 8 --subject $S --seeds 0 1 2
  echo "== ens5 s$S =="; $NIPS train_bwd.py $B --ensemble 5 --subject $S
  echo "== combined(win8+ens5+h256+four) s$S =="; $NIPS train_bwd.py $B --win-s 8 --ensemble 5 --hidden 256 --early four --subject $S
done
echo "NCL_SWEEP_OK"
