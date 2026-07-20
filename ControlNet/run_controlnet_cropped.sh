#!/bin/bash
#SBATCH --job-name=run_controlnet
#SBATCH --nodes=1
#SBATCH --nodelist=n1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=normal
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/controlnet_oct_%A.out
#SBATCH --error=logs/controlnet_oct_%A.err
#SBATCH --chdir=/data/vds/mmk/Codes/oct_data_synthesis/

# 1. Environment Initialization
module purge
module load Miniforge3/26.1.1-3
source activate /data/vds/env_pt

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
# Define whether to train from scratch or use a fine-tuned pretrained model
CHECKPOINT_PATH="./ControlNet/models/control_sd15_ini.ckpt"
#     --train_from_scratch \
echo "Starting ControlNet OCT training..."
srun python ControlNet/train_controlnet_oct.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --batch_size 4 \
    --image_size 256 \
    --epochs 100

# 4. Post-Run Cleanup
echo "Cleaning up SSD scratch..."
rm -rf "$LOCAL_SCRATCH"
echo "Cleanup completed successfully!"
