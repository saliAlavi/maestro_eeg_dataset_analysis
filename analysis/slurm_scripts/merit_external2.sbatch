#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=merit_ex2
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/merit_ex2_%A_%a.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/merit_ex2_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --array=0-4
# Tasks 0-1: KU Leuven / DTU leave-one-listener-out (the first attempt emptied
#   every fold -- these corpora share all content across listeners).
# Tasks 2-4: the SAME-TALKER construction on all three corpora.  Both candidates
#   are the attended talker at different times, so talker identity, spatial
#   position and any EEG-side lateralisation carry no information about which
#   is correct.  On KU Leuven and DTU the lag-band control did not discriminate;
#   this is the test that decides whether their accuracy is temporal tracking.
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
source ~/miniconda3/etc/profile.d/conda.sh; conda activate nips
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
C=(kul dtu kul dtu nju)
P=(loso loso within within within)
M=(qmatch qmatch shifted_qm shifted_qm shifted_qm)
c=${C[$SLURM_ARRAY_TASK_ID]}; p=${P[$SLURM_ARRAY_TASK_ID]}; m=${M[$SLURM_ARRAY_TASK_ID]}
tag=ext${c}${p}; [ "$m" = shifted_qm ] && tag=ext${c}${p}st
echo "########## corpus=$c protocol=$p candidates=$m ##########"
$PY -m src.main data=external model=merit runner=merit_external \
    wandb.mode=offline data.corpus=$c data.cand_mode=$m \
    runner.protocols=[$p] model.tag=__${tag}
