#!/bin/bash
# Fast GPU end-to-end sanity of the rebuilt (4-real-talker) backward pipeline before arrays.
#SBATCH --account=PAS2966
#SBATCH --job-name=bwd_sanity
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/bwd_sanity_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/bwd_sanity_%A.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --partition=debug-nextgen
#SBATCH --requeue
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/baseline_model/models
NIPS=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
echo "host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null | head -1)"
# 1) canonical smoke (linear + vlaai, within s1, 3 ep) — checks shapes + null≈0.25
$NIPS train_bwd.py --smoke
# 2) one real within fold, both innovations, to confirm 4-way null is 0.25 at 5s
echo "== within s1 vlaai (bands28+margin), full epochs =="
$NIPS train_bwd.py --model vlaai --bands 28 --mm-margin 0.3 --protocol within --subject 1 --seeds 0
# 3) the decision-window curve for one subject — confirms null 0.25 at EVERY window
echo "== curve s1 (train 5s, eval 5..30s), null must be ~0.25 at all windows =="
$NIPS train_bwd.py --model vlaai --bands 28 --mm-margin 0.3 --curve --n-seeds 2 --subject 1 --dec-wins 5 10 20 30
echo "SANITY DONE"
