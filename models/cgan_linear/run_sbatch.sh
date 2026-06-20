#!/bin/bash
#SBATCH --job-name=cgan_nr206
#SBATCH --nodes=1
#SBATCH --nodelist=n1              # Target RTX Pro 6000 node n1
#SBATCH --gres=gpu:4               # Request GPU resources (Matches 'num_gpus' in config.json)
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --output=logs/cgan_nr206_%j.out
#SBATCH --error=logs/cgan_nr206_%j.err
#SBATCH --chdir=/data/vds/mmk/Codes/oct_data_synthesis/   # Ensure job starts from the correct directory

# 1. Environment Initialization (LMod + Conda)
module purge
module load Miniforge3/26.1.1-3

# Activate your cluster conda environment
source activate /data/vds/env_pt



# 3. Optional: Stage dataset to local NVMe /tmp for I/O Optimization (highly recommended on this cluster)
# To use, uncomment the lines below and ensure your config.json references the staged directories:
#
# LOCAL_SCRATCH="/tmp/${USER}_job_${SLURM_JOB_ID}"
# mkdir -p "$LOCAL_SCRATCH"
# cp -r ./NR206 "$LOCAL_SCRATCH/"
#
# # Backup config.json and point to scratch path
# cp models/cgan_linear/config.json models/cgan_linear/config_backup.json
# sed -i "s|\"./NR206|\"$LOCAL_SCRATCH/NR206|g" models/cgan_linear/config.json

# 4. Execute Training
NUM_GPUS=4
srun python models/cgan_linear/conditional-GAN-generating-NR206.py --num_gpus $NUM_GPUS

# 5. Optional: Staging Cleanup (Mandatory if staging was enabled)
# mv models/cgan_linear/config_backup.json models/cgan_linear/config.json
# rm -rf "$LOCAL_SCRATCH"
