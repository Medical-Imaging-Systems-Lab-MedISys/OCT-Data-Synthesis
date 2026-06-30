#!/bin/bash
#SBATCH --job-name=RETFound_OCT5k
#SBATCH --output=logs/retfound_oct5k_%j.out
#SBATCH --error=logs/retfound_oct5k_%j.err
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
SOURCE_DATA="/data/vds/mmk/Codes/oct_data_synthesis/DATA/OCT5k/OCT5k_split"
WEIGHTS_SOURCE="/data/vds/mmk/Codes/weights/RETFound_oct_weights.pth" # Keep this as is assuming the weights are still there
export LOCAL_DATA_DIR="/tmp/OCT5k_${SLURM_JOB_ID}"

# 3. Secure Node /tmp Transfer
echo "Creating local temporary directory at ${LOCAL_DATA_DIR}..."
mkdir -p ${LOCAL_DATA_DIR}

echo "Rsyncing dataset from persistent storage to node /tmp for faster I/O..."
/data/vds/env_tools/bin/rsync -aq ${SOURCE_DATA}/ ${LOCAL_DATA_DIR}/
echo "Copying RETFound weights to local /tmp..."
cp ${WEIGHTS_SOURCE} ${LOCAL_DATA_DIR}/

# Activate your specific PyTorch environment
conda activate /data/vds/env_pt

# 4. Execute Training
echo "Starting RETFound Segmentation Training on OCT5k..."

python train_oct5k.py \
    --data_dir ${LOCAL_DATA_DIR} \
    --weights_path "${LOCAL_DATA_DIR}/RETFound_oct_weights.pth"

# 5. Node Storage Cleanup
echo "Training process finished. Initiating cleanup of ${LOCAL_DATA_DIR}..."
rm -rf ${LOCAL_DATA_DIR}
echo "Cleanup completed successfully."