#!/bin/bash
#SBATCH --job-name=Segment_AMD_SD
#SBATCH --output=logs/segment_amd_sd_%j.out
#SBATCH --error=logs/segment_amd_sd_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"

cd /data/vds/mmk/Codes/oct_data_synthesis/layer_segmentation

/data/vds/env_pt/bin/python segment_new_datasets.py \
    --dataset_dir "/data/vds/mmk/Codes/oct_data_synthesis/DATA/AMD-SD" \
    --dataset_type "ucsd" \
    --run_id "a8d99dbe233442e48fc391c7b02c5b74" \
    --model_type "nr206" \
    --output_dir "/data/vds/mmk/Codes/oct_data_synthesis/layer_segmentation/predictions_amd_sd/NR206_Finetune_Aug_256/" \
    --run_name "Segment_AMD_SD_NR206_Finetune_Aug_256" \
    --img_size 256

echo "Inference finished."
