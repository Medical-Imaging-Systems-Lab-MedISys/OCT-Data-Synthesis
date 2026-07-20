#!/bin/bash
#SBATCH --job-name=RETFound_Duke
#SBATCH --output=logs/retfound_duke_%j.out
#SBATCH --error=logs/retfound_duke_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1         # Increase this (e.g., gpu:2) if using Multi-GPU
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# 1. Load Environment Modules
module load Miniforge3/26.1.1-3
# module load cuda/11.8 # Uncomment if CUDA still needs explicit loading

echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"

# 2. Define Directories
SOURCE_DATA="/data/vds/mmk/Codes/oct_data/Duke_2015"
WEIGHTS_SOURCE="/data/vds/mmk/Codes/oct_data_synthesis/layer_segmentation/checkpoints/RETFound_mae_natureOCT.pth"
export LOCAL_DATA_DIR="/tmp/Duke_${SLURM_JOB_ID}"

# 3. Secure Node /tmp Transfer
echo "Creating local temporary directory at ${LOCAL_DATA_DIR}..."
mkdir -p ${LOCAL_DATA_DIR}

echo "Rsyncing dataset from persistent storage to node /tmp for faster I/O..."
/data/vds/env_tools/bin/rsync -aq ${SOURCE_DATA}/ ${LOCAL_DATA_DIR}/
echo "Rsyncing RETFound weights to local /tmp..."
/data/vds/env_tools/bin/rsync -aq ${WEIGHTS_SOURCE} ${LOCAL_DATA_DIR}/

# 4. Execute Training
echo "Starting RETFound Segmentation Training..."

# Navigate to the correct directory (SLURM executes from a spool dir by default)
cd /data/vds/mmk/Codes/oct_data_synthesis/layer_segmentation

# Update the --weights_path argument
/data/vds/env_pt/bin/python train.py \
    --data_dir ${LOCAL_DATA_DIR} \
    --weights_path "${LOCAL_DATA_DIR}/RETFound_mae_natureOCT.pth"

# 5. Node Storage Cleanup (Disabled as requested to keep codes/checkpoints on node)
echo "Training process finished. Keeping local data on node."