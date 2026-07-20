#!/bin/bash
#SBATCH --job-name=Segment_MAT_Datasets
#SBATCH --output=logs/segment_mats_%j.out
#SBATCH --error=logs/segment_mats_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

cd /data/vds/mmk/Codes/oct_data_synthesis/layer_segmentation

# 2011_IOVS_Chiu
/data/vds/env_pt/bin/python segment_new_datasets.py \
    --dataset_dir "/data/vds/mmk/Codes/oct_data_synthesis/DATA/2011_IOVS_Chiu/png_extracted" \
    --dataset_type "ucsd" \
    --run_id "a8d99dbe233442e48fc391c7b02c5b74" \
    --model_type "nr206" \
    --output_dir "/data/vds/mmk/Codes/oct_data_synthesis/layer_segmentation/predictions_2011_iovs/NR206/" \
    --run_name "Segment_2011_IOVS_NR206" \
    --img_size 256

# 2015_BOE_Chiu2
/data/vds/env_pt/bin/python segment_new_datasets.py \
    --dataset_dir "/data/vds/mmk/Codes/oct_data_synthesis/DATA/2015_BOE_Chiu2/png_extracted" \
    --dataset_type "ucsd" \
    --run_id "a8d99dbe233442e48fc391c7b02c5b74" \
    --model_type "nr206" \
    --output_dir "/data/vds/mmk/Codes/oct_data_synthesis/layer_segmentation/predictions_2015_boe/NR206/" \
    --run_name "Segment_2015_BOE_NR206" \
    --img_size 256

# kmader_eye_oct (Heidelberg)
/data/vds/env_pt/bin/python segment_new_datasets.py \
    --dataset_dir "/data/vds/mmk/Codes/oct_data_synthesis/DATA/kmader_eye_oct/heiderlberg_oct/png_extracted" \
    --dataset_type "ucsd" \
    --run_id "a8d99dbe233442e48fc391c7b02c5b74" \
    --model_type "nr206" \
    --output_dir "/data/vds/mmk/Codes/oct_data_synthesis/layer_segmentation/predictions_kmader_heidelberg/NR206/" \
    --run_name "Segment_Kmader_Heidelberg_NR206" \
    --img_size 256

echo "Inference finished."
