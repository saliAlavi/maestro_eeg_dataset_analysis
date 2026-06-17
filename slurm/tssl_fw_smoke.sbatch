#!/bin/bash
#SBATCH --account=PAS2966
#SBATCH --partition=nextgen
#SBATCH --gres=gpu:1
#SBATCH --job-name=tssl_fw
#SBATCH --output=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/logs/tssl_fw_%j.out
#SBATCH --error=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/logs/tssl_fw_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
set -euo pipefail
export PYTHONNOUSERSITE=1 HF_HOME=/fs/scratch/PAS2301/alialavi/cache/hf
export WANDB_DIR=/fs/scratch/PAS2301/alialavi/projects/multimodal_aad/runs
mkdir -p "$WANDB_DIR"
cd /users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu
PY=/users/PAS2301/alialavi/miniconda3/envs/nips/bin/python
$PY -m src.main mode=train model=transfer_ssl data=aad_source runner.name=transfer_ssl \
  data.subjects=[1,2,3] 'model.seeds=[0]' model.ssl_epochs=5 model.pretrain_epochs=5 \
  model.finetune_epochs=5 model.scratch_epochs=5 wandb.mode=offline
echo "[tssl_fw] DONE"
