#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=merit_ext
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/merit_ext_%A_%a.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/merit_ext_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --array=0-5
# MERIT's coupling branch on the public two-talker corpora.  Tasks 0-2 are the
# within-listener protocol, 3-5 leave-one-listener-out.  Each task runs the
# centred / causal / acausal lag bands, so the directionality control travels
# with the result rather than being a separate claim.
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
source ~/miniconda3/etc/profile.d/conda.sh; conda activate nips
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
C=(nju kul dtu nju kul dtu)
P=(within within within loso loso loso)
c=${C[$SLURM_ARRAY_TASK_ID]}; p=${P[$SLURM_ARRAY_TASK_ID]}
echo "########## corpus=$c protocol=$p ##########"
$PY -m src.main data=external model=merit runner=merit_external \
    wandb.mode=offline data.corpus=$c runner.protocols=[$p] \
    model.tag=__ext${c}${p}
