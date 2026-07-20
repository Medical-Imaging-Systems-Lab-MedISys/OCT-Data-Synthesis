#!/bin/bash
#SBATCH --job-name=RETFound_NR206_Frozen_NoWM_Aug
#SBATCH --output=logs/retfound_nr206_frozen_nw_aug_%j.out
#SBATCH --error=logs/retfound_nr206_frozen_nw_aug_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00

echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"

# Define Directories
SOURCE_DATA="/data/vds/mmk/Codes/oct_data_synthesis/DATA/NR206"
WEIGHTS_SOURCE="/data/vds/mmk/Codes/oct_data_synthesis/layer_segmentation/checkpoints/RETFound_mae_natureOCT.pth"
export LOCAL_DATA_DIR="/tmp/NR206_frozen_nw_aug_${SLURM_JOB_ID}"

# Secure Node /tmp Transfer
echo "Creating local temporary directory at ${LOCAL_DATA_DIR}..."
mkdir -p ${LOCAL_DATA_DIR}

echo "Rsyncing dataset from persistent storage to node /tmp for faster I/O..."
/data/vds/env_tools/bin/rsync -aq ${SOURCE_DATA}/ ${LOCAL_DATA_DIR}/
echo "Rsyncing RETFound weights to local /tmp..."
/data/vds/env_tools/bin/rsync -aq ${WEIGHTS_SOURCE} ${LOCAL_DATA_DIR}/

# Execute Training
echo "Starting Frozen RETFound Segmentation Training on NR206 (No WM removal, with Augmentations)..."
cd /data/vds/mmk/Codes/oct_data_synthesis/layer_segmentation

/data/vds/env_pt/bin/python train_nr206.py \
    --data_dir ${LOCAL_DATA_DIR} \
    --weights_path "${LOCAL_DATA_DIR}/RETFound_mae_natureOCT.pth" \
    --freeze_backbone \
    --no_watermark_removal \
    --use_augmentations \
    --run_name "RETFound_NR206_Frozen_NoWM_Aug"

echo "Training process finished. Keeping local data on node."
