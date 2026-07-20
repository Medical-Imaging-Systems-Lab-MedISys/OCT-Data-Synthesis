#!/bin/bash
#SBATCH --job-name=Segment_New_Datasets
#SBATCH --output=logs/segment_new_%j.out
#SBATCH --error=logs/segment_new_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

cd /data/vds/mmk/Codes/oct_data_synthesis/layer_segmentation

# THOCT1800
/data/vds/env_pt/bin/python segment_new_datasets.py \
    --dataset_dir "/data/vds/mmk/Codes/oct_data_synthesis/DATA/THOCT1800" \
    --dataset_type "ucsd" \
    --run_id "a8d99dbe233442e48fc391c7b02c5b74" \
    --model_type "nr206" \
    --output_dir "/data/vds/mmk/Codes/oct_data_synthesis/layer_segmentation/predictions_thoct1800/NR206/" \
    --run_name "Segment_THOCT1800_NR206" \
    --img_size 256

# 2014_BOE_Srinivasan
/data/vds/env_pt/bin/python segment_new_datasets.py \
    --dataset_dir "/data/vds/mmk/Codes/oct_data_synthesis/DATA/2014_BOE_Srinivasan" \
    --dataset_type "ucsd" \
    --run_id "a8d99dbe233442e48fc391c7b02c5b74" \
    --model_type "nr206" \
    --output_dir "/data/vds/mmk/Codes/oct_data_synthesis/layer_segmentation/predictions_boe2014/NR206/" \
    --run_name "Segment_BOE2014_NR206" \
    --img_size 256

echo "Inference finished."
