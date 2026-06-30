#!/bin/bash
#SBATCH --job-name=cgan_oct_array
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --array=5
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --output=logs/cgan_oct_%A_%a.out
#SBATCH --error=logs/cgan_oct_%A_%a.err
#SBATCH --chdir=/data/vds/mmk/Codes/oct_data_synthesis/

module purge
module load Miniforge3/26.1.1-3
source activate /data/vds/env_pt

LOCAL_SCRATCH="/tmp/${USER}_job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Staging dataset to local SSD scratch: $LOCAL_SCRATCH"
mkdir -p "$LOCAL_SCRATCH"
cp -r ./DATA/NR206 "$LOCAL_SCRATCH/"

# Check if individual config files exist, else use default logic
CONFIG_FILE="models/cgan_linear/config_exp${SLURM_ARRAY_TASK_ID}.json"
CONFIG_BACKUP="models/cgan_linear/config_exp${SLURM_ARRAY_TASK_ID}_backup.json"

if [ -f "$CONFIG_FILE" ]; then
    cp "$CONFIG_FILE" "$CONFIG_BACKUP"
    sed -i "s|\"./DATA/NR206|\"$LOCAL_SCRATCH/NR206|g" "$CONFIG_FILE"
    srun python models/cgan_linear/conditional-GAN-generating-NR206.py --config "$CONFIG_FILE"
    mv "$CONFIG_BACKUP" "$CONFIG_FILE"
else
    echo "Config file $CONFIG_FILE not found, running with array ID as arg or default"
    # Modify as needed to support your argparse
    srun python models/cgan_linear/conditional-GAN-generating-NR206.py
fi

rm -rf "$LOCAL_SCRATCH"
echo "Cleanup completed successfully!"
