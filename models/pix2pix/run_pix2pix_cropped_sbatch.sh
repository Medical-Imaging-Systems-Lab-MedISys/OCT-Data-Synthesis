#!/bin/bash
#SBATCH --job-name=pix2pix_oct
#SBATCH --nodes=1
#SBATCH --nodelist=n1              # Target RTX Pro 6000 node n1
#SBATCH --gres=gpu:1               # Request 1 GPU resource per array task
#SBATCH --array=5,10               # Run config 5 and config 10 in parallel
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --output=logs/pix2pix_oct_%A_%a.out
#SBATCH --error=logs/pix2pix_oct_%A_%a.err
#SBATCH --chdir=/data/vds/mmk/Codes/oct_data_synthesis/   # Ensure job starts from the correct directory

# 1. Environment Initialization
module purge
module load Miniforge3/26.1.1-3

# Activate your cluster conda environment
source activate /data/vds/env_pt



# 3. Stage dataset to local NVMe SSD (/tmp) for high-performance I/O
LOCAL_SCRATCH="/tmp/${USER}_job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Staging dataset to local SSD scratch: $LOCAL_SCRATCH"
mkdir -p "$LOCAL_SCRATCH"
cp -r ./DATA/NR206 "$LOCAL_SCRATCH/"

# Backup specific config and rewrite dataset paths to point to SSD scratch
CONFIG_FILE="models/pix2pix/config_exp${SLURM_ARRAY_TASK_ID}.json"
CONFIG_BACKUP="models/pix2pix/config_exp${SLURM_ARRAY_TASK_ID}_backup.json"

cp "$CONFIG_FILE" "$CONFIG_BACKUP"
sed -i "s|\"./DATA/NR206|\"$LOCAL_SCRATCH/NR206|g" "$CONFIG_FILE"

# 4. Execute Training
srun python models/pix2pix/train_pix2pix_cropped.py --config "$CONFIG_FILE"

# 5. Post-Run Cleanup
echo "Restoring configuration file and cleaning up SSD scratch..."
mv "$CONFIG_BACKUP" "$CONFIG_FILE"
rm -rf "$LOCAL_SCRATCH"
echo "Cleanup completed successfully!"
