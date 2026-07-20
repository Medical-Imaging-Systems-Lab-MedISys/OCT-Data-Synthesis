#!/bin/bash
#SBATCH --job-name=cGAN_Cropped
#SBATCH --output=logs/cgan_cropped_%j.out
#SBATCH --error=logs/cgan_cropped_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00

LOSS_TYPE="l1"

# 1. Load Environment Modules
module load Miniforge3/26.1.1-3
# module load cuda/11.8 # Uncomment if CUDA still needs explicit loading alongside Miniforge

echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"

# 2. Define Directories
SOURCE_DATA="/data/vds/mmk/Codes/oct_data_synthesis/DATA/NR206"
export LOCAL_DATA_DIR="/tmp/NR206_${SLURM_JOB_ID}"

# 3. Secure Node /tmp Transfer
echo "Creating local temporary directory at ${LOCAL_DATA_DIR}..."
mkdir -p ${LOCAL_DATA_DIR}

echo "Rsyncing data from persistent storage to node /tmp..."
# Directly targeting the rsync binary inside the specified conda env avoids needing to activate it
/data/vds/env_tools/bin/rsync -aq ${SOURCE_DATA}/ ${LOCAL_DATA_DIR}/

# 4. Execute Training
#export LOCAL_DATA_DIR="/tmp/NR206_${SLURM_JOB_ID}/NR206"
echo "Starting cGAN Cropped Training with ${LOSS_TYPE^^} Loss..."
/data/vds/env_pt/bin/python models/cgan_linear/train_cgan_cropped.py --loss_type ${LOSS_TYPE}

# 5. Node Storage Cleanup (Disabled as requested to keep codes/checkpoints on node)
echo "Training process finished. Keeping local data on node."
