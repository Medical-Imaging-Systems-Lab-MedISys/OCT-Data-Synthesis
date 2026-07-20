#!/bin/bash
#SBATCH --job-name=Segment_New_Zips
#SBATCH --output=logs/segment_new_zips_%j.out
#SBATCH --error=logs/segment_new_zips_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

cd /data/vds/mmk/Codes/oct_data_synthesis/layer_segmentation

DATASETS=(
    "AMD"
    "Age-related Macular Degeneration Retinal OCT images"
    "Central serous retinopathy retinal OCT images"
    "Control"
    "Diabetic Retinopathy Retinal OCT Images"
    "Macular Hole Retinal OCT images"
    "Normal Retinal OCT images"
)

for DS in "${DATASETS[@]}"; do
    DS_NAME_CLEAN=$(echo "$DS" | sed -r 's/[^a-zA-Z0-9]+/_/g')
    /data/vds/env_pt/bin/python segment_new_datasets.py \
        --dataset_dir "/data/vds/mmk/Codes/oct_data_synthesis/DATA/$DS" \
        --dataset_type "ucsd" \
        --run_id "a8d99dbe233442e48fc391c7b02c5b74" \
        --model_type "nr206" \
        --output_dir "/data/vds/mmk/Codes/oct_data_synthesis/layer_segmentation/predictions_${DS_NAME_CLEAN}/NR206/" \
        --run_name "Segment_${DS_NAME_CLEAN}_NR206" \
        --img_size 256
done

# Manual Segmentation with GT
/data/vds/env_pt/bin/python segment_new_datasets.py \
    --dataset_dir "/data/vds/mmk/Codes/oct_data_synthesis/DATA/Manual Segmentation/images" \
    --dataset_type "ucsd" \
    --run_id "a8d99dbe233442e48fc391c7b02c5b74" \
    --model_type "nr206" \
    --output_dir "/data/vds/mmk/Codes/oct_data_synthesis/layer_segmentation/predictions_Manual_Segmentation/NR206/" \
    --run_name "Segment_Manual_Segmentation_NR206" \
    --gt_dir "/data/vds/mmk/Codes/oct_data_synthesis/DATA/Manual Segmentation/masks_gt" \
    --img_size 256

echo "Inference finished."
