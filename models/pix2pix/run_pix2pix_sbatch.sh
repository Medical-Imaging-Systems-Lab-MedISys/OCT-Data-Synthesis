#!/bin/bash
#SBATCH --job-name=pix2pix_oct
#SBATCH --nodes=1
#SBATCH --nodelist=n1              # Target RTX Pro 6000 node n1
#SBATCH --gres=gpu:1               # Request 1 GPU resource
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --output=pix2pix_oct_%j.out
#SBATCH --error=pix2pix_oct_%j.err
#SBATCH --chdir=/data/vds/mmk/Codes/oct_data_synthesis/   # Ensure job starts from the correct directory

# 1. Environment Initialization
module purge
module load Miniforge3/26.1.1-3

# Activate your cluster conda environment
source activate /data/vds/env_pt

# 2. Stage dataset to local NVMe SSD (/tmp) for high-performance I/O
LOCAL_SCRATCH="/tmp/${USER}_job_${SLURM_JOB_ID}"
echo "Staging dataset to local SSD scratch: $LOCAL_SCRATCH"
mkdir -p "$LOCAL_SCRATCH"
cp -r ./NR206 "$LOCAL_SCRATCH/"

# Backup config_pix2pix.json and rewrite dataset paths to point to SSD scratch
cp models/pix2pix/config_pix2pix.json models/pix2pix/config_pix2pix_backup.json
sed -i "s|\"./NR206|\"$LOCAL_SCRATCH/NR206|g" models/pix2pix/config_pix2pix.json

# 3. Execute Training
srun python models/pix2pix/train_pix2pix.py --config models/pix2pix/config_pix2pix.json

# 4. Post-Run Cleanup
echo "Restoring configuration file and cleaning up SSD scratch..."
mv models/pix2pix/config_pix2pix_backup.json models/pix2pix/config_pix2pix.json
rm -rf "$LOCAL_SCRATCH"
echo "Cleanup completed successfully!"
