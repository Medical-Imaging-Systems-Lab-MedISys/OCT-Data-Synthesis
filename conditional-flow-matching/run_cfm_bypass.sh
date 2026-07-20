#!/bin/bash
module purge
module load Miniforge3/26.1.1-3

# Initialize shell integration for conda
eval "$(conda shell.bash hook)"

# Activate environment
conda activate /data/vds/env_pt

# Stage dataset to node local SSD
export LOCAL_SCRATCH="/tmp/mohan.manepalli_job_cfm_bypass"
echo "Staging dataset to local SSD scratch: $LOCAL_SCRATCH"
mkdir -p "$LOCAL_SCRATCH"
mkdir -p "$LOCAL_SCRATCH/NR206"
cp -r /data/vds/mmk/Codes/oct_data_synthesis/DATA/combined_synthesis_data/real "$LOCAL_SCRATCH/NR206/train"
cp -r /data/vds/mmk/Codes/oct_data_synthesis/DATA/combined_synthesis_data/masks "$LOCAL_SCRATCH/NR206/train_labels"
cp -r /data/vds/mmk/Codes/oct_data_synthesis/DATA/val_comparison_set/real "$LOCAL_SCRATCH/NR206/test"
cp -r /data/vds/mmk/Codes/oct_data_synthesis/DATA/val_comparison_set/labels "$LOCAL_SCRATCH/NR206/test_labels"
export LOCAL_DATA_DIR="$LOCAL_SCRATCH/NR206"

export CUDA_VISIBLE_DEVICES=1
echo "Starting CFM training on GPU 1..."
python /data/vds/mmk/Codes/oct_data_synthesis/conditional-flow-matching/train_val.py

# Cleanup
echo "Restoring configuration file and cleaning up SSD scratch..."
rm -rf "$LOCAL_SCRATCH"
echo "Cleanup completed successfully!"
