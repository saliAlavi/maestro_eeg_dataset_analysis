#!/bin/bash
# n_gh_checks — post-processing: late fusion + SNR/error analyses + aggregation.
# Submit with a dependency on the within + loso array jobs:
#   sbatch --dependency=afterok:<within_jobid>:<loso_jobid> post.sbatch
#SBATCH --account=PAS2966
#SBATCH --job-name=ngh_post
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/ngh_post_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/ngh_post_%A.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --partition=preemptible-nextgen
#SBATCH --requeue

set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/n_gh_checks
NIPS=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
echo "host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null | head -1)"

# tolerate individual step failures so the aggregate table is always produced
$NIPS late_fusion_gh.py --protocol within || echo "WARN: within fusion failed"
$NIPS late_fusion_gh.py --protocol loso   || echo "WARN: loso fusion failed"
$NIPS analyze_snr_gh.py --protocol within || echo "WARN: snr analysis failed"
$NIPS analyze_error_gh.py --protocol within || echo "WARN: error analysis failed"
$NIPS aggregate.py
