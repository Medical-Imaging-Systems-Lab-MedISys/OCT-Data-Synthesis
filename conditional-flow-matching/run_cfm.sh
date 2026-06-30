#!/bin/bash
#SBATCH --job-name=cfm_oct
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/cfm_oct_%A_%a.out
#SBATCH --error=logs/cfm_oct_%A_%a.err
#SBATCH --chdir=/data/vds/mmk/Codes/oct_data_synthesis/

module purge
module load Miniforge3/26.1.1-3
source activate /data/vds/env_pt

LOCAL_SCRATCH="/tmp/${USER}_job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Staging dataset to local SSD scratch: $LOCAL_SCRATCH"
mkdir -p "$LOCAL_SCRATCH"
cp -r ./DATA/NR206 "$LOCAL_SCRATCH/"

srun python conditional-flow-matching/train_val.py

rm -rf "$LOCAL_SCRATCH"
echo "Cleanup completed successfully!"
