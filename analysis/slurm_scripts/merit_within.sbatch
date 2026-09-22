#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --job-name=credit_within
#SBATCH --output=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/credit_within_%A.out
#SBATCH --error=/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/slurm_logs/credit_within_%A.err
#SBATCH --time=20:00:00
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#
# Within-subject (content-disjoint 5-fold) experiments.  Pick one with EXP=:
#   EXP=streams     per-stream credit for seven stream sets            (Table: streams)
#   EXP=hinge       contribution-hinge x selection-criterion cross     (Table: the punchline)
#   EXP=ladder_raw  the ablation ladder on confounded candidates     (Table: ladder)
#   EXP=ladder_qm   the same ladder on distribution-matched candidates
#   EXP=lag         causal vs acausal lag band                         (Table: directionality)
#   EXP=orientfree  same-talker negatives, K=2 and K=3                 (Table: orienting-free)
set -euo pipefail
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nips
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
EXP=${EXP:-streams}
BASE="data=merit model=merit wandb.mode=offline"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

case "$EXP" in
  streams)
    $PY -m src.main $BASE runner=merit_streams model.tag=__streams ;;
  hinge)
    $PY -m src.main $BASE runner=merit_hinge model.tag=__hinge ;;
  ladder_raw)
    $PY -m src.main $BASE runner=merit_ladder data.cand_mode=raw model.tag=__ladder_raw ;;
  ladder_qm)
    $PY -m src.main $BASE runner=merit_ladder data.cand_mode=qmatch model.tag=__ladder_qm ;;
  lag)
    echo "### lag band, four-way matched candidates ###"
    $PY -m src.main $BASE runner=merit_lag data.cand_mode=qmatch data.K=4 model.tag=__lag_qm
    echo "### lag band, binary same-talker candidates (exact 1/K floor) ###"
    $PY -m src.main $BASE runner=merit_lag data.cand_mode=shifted_qm data.K=2 model.tag=__lag_mm ;;
  orientfree)
    echo "### same-talker temporal negatives: K=3 then K=2 ###"
    $PY -m src.main $BASE runner=merit_streams data.cand_mode=shifted_qm data.K=3 model.tag=__mm3
    $PY -m src.main $BASE runner=merit_streams data.cand_mode=shifted_qm data.K=2 model.tag=__mm2 ;;
  *) echo "unknown EXP=$EXP"; exit 1 ;;
esac
