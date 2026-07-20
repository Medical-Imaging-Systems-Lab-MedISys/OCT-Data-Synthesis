#!/bin/bash
set -e

# 1. Environment Initialization
source activate /data/vds/env_pt

# Ensure working directory is repo root
cd /data/vds/mmk/Codes/oct_data_synthesis/

# 2. Stage dataset to local NVMe SSD (/tmp)
export LOCAL_SCRATCH="/tmp/${USER}_job_controlnet"
echo "Staging dataset to local SSD scratch: $LOCAL_SCRATCH"
mkdir -p "$LOCAL_SCRATCH"
cp -r /data/vds/mmk/Codes/oct_data_synthesis/DATA/combined_synthesis_data "$LOCAL_SCRATCH/"
cp -r /data/vds/mmk/Codes/oct_data_synthesis/DATA/val_comparison_set "$LOCAL_SCRATCH/"
export TRAIN_REAL="$LOCAL_SCRATCH/combined_synthesis_data/real"
export TRAIN_LABELS="$LOCAL_SCRATCH/combined_synthesis_data/masks"
export VAL_REAL="$LOCAL_SCRATCH/val_comparison_set/real"
export VAL_LABELS="$LOCAL_SCRATCH/val_comparison_set/labels"

# 3. Execution Configuration
CHECKPOINT_PATH="./ControlNet/models/control_sd15_ini.ckpt"
echo "Starting ControlNet OCT 300 Epochs Pretrained Fine-Tuning..."
python -u ControlNet/train_controlnet_oct.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --batch_size 4 \
    --image_size 256 \
    --epochs 300

# 4. Post-Run Cleanup
echo "Cleaning up SSD scratch..."
rm -rf "$LOCAL_SCRATCH"
echo "ControlNet training completed successfully!"
