#!/bin/bash
#SBATCH --job-name=run_cfm_2
#SBATCH --nodes=1
#SBATCH --nodelist=n1              # Target RTX Pro 6000 node n1
#SBATCH --gres=gpu:1               # Request 1 GPU resource per array task
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=normal
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/cfm_oct_%A.out
#SBATCH --error=logs/cfm_oct_%A.err
#SBATCH --chdir=/data/vds/mmk/Codes/oct_data_synthesis/   # Ensure job starts from the correct directory

# 1. Environment Initialization
module purge
module load Miniforge3/26.1.1-3

# Activate your cluster conda environment
source activate /data/vds/env_pt

# 3. Stage dataset to local NVMe SSD (/tmp) for high-performance I/O
export LOCAL_SCRATCH="/tmp/${USER}_job"
echo "Staging dataset to local SSD scratch: $LOCAL_SCRATCH"
mkdir -p "$LOCAL_SCRATCH"
mkdir -p "$LOCAL_SCRATCH/NR206"
cp -r /data/vds/mmk/Codes/oct_data_synthesis/DATA/combined_synthesis_data/real "$LOCAL_SCRATCH/NR206/train"
cp -r /data/vds/mmk/Codes/oct_data_synthesis/DATA/combined_synthesis_data/masks "$LOCAL_SCRATCH/NR206/train_labels"
cp -r /data/vds/mmk/Codes/oct_data_synthesis/DATA/val_comparison_set/real "$LOCAL_SCRATCH/NR206/test"
cp -r /data/vds/mmk/Codes/oct_data_synthesis/DATA/val_comparison_set/labels "$LOCAL_SCRATCH/NR206/test_labels"
export LOCAL_DATA_DIR="$LOCAL_SCRATCH/NR206"

srun python -u conditional-flow-matching/train_val.py

# 5. Post-Run Cleanup
echo "Restoring configuration file and cleaning up SSD scratch..."
rm -rf "$LOCAL_SCRATCH"
echo "Cleanup completed successfully!" My bash script
