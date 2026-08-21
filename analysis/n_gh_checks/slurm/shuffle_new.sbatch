#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=shuf_new
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/shuf_new_%x_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/shuf_new_%x_%A.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=preemptible-nextgen
#SBATCH --gpus-per-node=1

set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/n_gh_checks
NIPS=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python

MODE=${MODE:-eeg}
TAGS=${TAGS:-w5_h2.5,w10_h5,w15_h7.5,w20_h10,w30_h15}
PROTOCOLS=${PROTOCOLS:-loso,within}

echo "=== shuffle-null on updated MAESTRO repo | mode=$MODE tags=$TAGS protocols=$PROTOCOLS ==="
$NIPS shuffle_new_repo.py --mode "$MODE" --tags "$TAGS" --protocols "$PROTOCOLS" --n_shuffle 20
echo "=== done ==="
