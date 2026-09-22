#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=merit_v2
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/merit_v2_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/merit_v2_%A.err
#SBATCH --time=16:00:00
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#
# v2 improvements. EXP=:
#   ladder  EEG-only improvement ladder, with and without Euclidean alignment
#   best    full stack across stream sets (orienting features + EA + listener adaptation)
#   onset   the same two, with the rectified-envelope-derivative audio channel
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
source ~/miniconda3/etc/profile.d/conda.sh; conda activate nips
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
EXP=${EXP:-ladder}
# whole-trial storage + 10 s random crops; the eval grid reproduces the w10/h5 windows
CROP="data.window_sec=30 data.hop_sec=15 data.crop_len=640 data.n_train_crops=8 data.n_eval_crops=5"
BASE="data=merit model=merit wandb.mode=offline $CROP"

case "$EXP" in
  ladder)
    echo "########## EEG ladder, no Euclidean alignment ##########"
    $PY -m src.main $BASE runner=merit_ladder2 model.tag=__v2ladder
    echo "########## EEG ladder, + Euclidean alignment ##########"
    $PY -m src.main $BASE runner=merit_ladder2 data.euclidean_align=true \
        model.tag=__v2ladderEA ;;
  best)
    echo "########## full stack: orienting features + EA + listener adaptation ##########"
    $PY -m src.main $BASE runner=merit_best data.orient_feats=true \
        data.euclidean_align=true model.tag=__v2best
    echo "########## same, leave-one-listener-out ##########"
    $PY -m src.main $BASE runner=merit_best data.orient_feats=true \
        data.euclidean_align=true runner.protocols=[loso] model.tag=__v2bestloso ;;
  onset)
    echo "########## EEG ladder with the onset channel ##########"
    $PY -m src.main $BASE runner=merit_ladder2 data.audio_feats=env_onset \
        model.tag=__v2onset
    echo "########## full stack with the onset channel ##########"
    $PY -m src.main $BASE runner=merit_best data.orient_feats=true \
        data.euclidean_align=true data.audio_feats=env_onset model.tag=__v2bestonset ;;
  onsetloso)
    echo "########## full stack with the onset channel, leave-one-listener-out ##########"
    $PY -m src.main $BASE runner=merit_best data.orient_feats=true \
        data.euclidean_align=true data.audio_feats=env_onset \
        runner.protocols=[loso] model.tag=__v2bestonsetloso ;;
  hinge2)
    echo "########## clean hinge 2x2 at the v2 operating point ##########"
    $PY -m src.main $BASE runner=merit_hinge2 data.orient_feats=true \
        data.euclidean_align=true data.audio_feats=env_onset model.tag=__v2hinge2
    echo "########## the two main arms, leave-one-listener-out ##########"
    $PY -m src.main $BASE runner=merit_hinge2 data.orient_feats=true \
        data.euclidean_align=true data.audio_feats=env_onset \
        runner.protocols=[loso] model.tag=__v2hinge2loso ;;
  *) echo "unknown EXP=$EXP"; exit 1 ;;
esac
