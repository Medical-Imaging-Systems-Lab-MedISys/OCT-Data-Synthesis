#!/bin/bash
# Master script to retrain all 8 models with 256 input size concurrently on reserved GPUs
# n1: 2 GPUs via Job 2316 (GPU 0 + GPU 1)
# n2: 6 GPUs via Jobs 2318/2319/2320 (GPU 0 + GPU 1 of each job)
# All 8 runs start simultaneously in parallel.

cd /data/vds/mmk/Codes/oct_data_synthesis/layer_segmentation

mkdir -p logs

NR206_DIR="/data/vds/mmk/Codes/oct_data_synthesis/DATA/NR206"
OCT5K_DIR="/data/vds/mmk/Codes/oct_data_synthesis/DATA/OCT5k/OCT5k_split"

echo "=== Launching all 8 training runs in parallel ==="

# Node n1 Job 2316 – GPU 0: NR206 Finetune 256
nohup env CUDA_VISIBLE_DEVICES=0 srun --jobid=2316 --overlap /data/vds/env_pt/bin/python train_nr206.py \
    --data_dir ${NR206_DIR} \
    --weights_path "checkpoints/RETFound_mae_natureOCT.pth" \
    --img_size 256 --gpu 0 \
    --run_name "RETFound_NR206_Finetune_WM_Removed_256" > logs/train_nr206_ft_256.log 2>&1 &

# Node n1 Job 2316 – GPU 1: NR206 Frozen 256
nohup env CUDA_VISIBLE_DEVICES=1 srun --jobid=2316 --overlap /data/vds/env_pt/bin/python train_nr206.py \
    --data_dir ${NR206_DIR} \
    --weights_path "checkpoints/RETFound_mae_natureOCT.pth" \
    --img_size 256 --gpu 1 --freeze_backbone \
    --run_name "RETFound_NR206_Frozen_WM_Removed_256" > logs/train_nr206_frozen_256.log 2>&1 &

# Node n2 Job 2320 – GPU 0: NR206 Finetune Aug 256
nohup env CUDA_VISIBLE_DEVICES=0 srun --jobid=2320 --overlap /data/vds/env_pt/bin/python train_nr206.py \
    --data_dir ${NR206_DIR} \
    --weights_path "checkpoints/RETFound_mae_natureOCT.pth" \
    --img_size 256 --gpu 0 --use_augmentations \
    --run_name "RETFound_NR206_Finetune_WM_Removed_Aug_256" > logs/train_nr206_ft_aug_256.log 2>&1 &

# Node n2 Job 2320 – GPU 1: NR206 Frozen Aug 256
nohup env CUDA_VISIBLE_DEVICES=1 srun --jobid=2320 --overlap /data/vds/env_pt/bin/python train_nr206.py \
    --data_dir ${NR206_DIR} \
    --weights_path "checkpoints/RETFound_mae_natureOCT.pth" \
    --img_size 256 --gpu 1 --freeze_backbone --use_augmentations \
    --run_name "RETFound_NR206_Frozen_WM_Removed_Aug_256" > logs/train_nr206_frozen_aug_256.log 2>&1 &

# Node n2 Job 2319 – GPU 0: OCT5k Finetune 256
nohup env CUDA_VISIBLE_DEVICES=0 srun --jobid=2319 --overlap /data/vds/env_pt/bin/python train_oct5k.py \
    --data_dir ${OCT5K_DIR} \
    --weights_path "checkpoints/RETFound_mae_natureOCT.pth" \
    --img_size 256 --gpu 0 \
    --run_name "RETFound_OCT5k_Finetune_256" > logs/train_oct5k_ft_256.log 2>&1 &

# Node n2 Job 2319 – GPU 1: OCT5k Frozen 256
nohup env CUDA_VISIBLE_DEVICES=1 srun --jobid=2319 --overlap /data/vds/env_pt/bin/python train_oct5k.py \
    --data_dir ${OCT5K_DIR} \
    --weights_path "checkpoints/RETFound_mae_natureOCT.pth" \
    --img_size 256 --gpu 1 --freeze_backbone \
    --run_name "RETFound_OCT5k_Frozen_256" > logs/train_oct5k_frozen_256.log 2>&1 &

# Node n2 Job 2318 – GPU 0: OCT5k Finetune Aug 256
nohup env CUDA_VISIBLE_DEVICES=0 srun --jobid=2318 --overlap /data/vds/env_pt/bin/python train_oct5k.py \
    --data_dir ${OCT5K_DIR} \
    --weights_path "checkpoints/RETFound_mae_natureOCT.pth" \
    --img_size 256 --gpu 0 --use_augmentations \
    --run_name "RETFound_OCT5k_Finetune_Aug_256" > logs/train_oct5k_ft_aug_256.log 2>&1 &

# Node n2 Job 2318 – GPU 1: OCT5k Frozen Aug 256
nohup env CUDA_VISIBLE_DEVICES=1 srun --jobid=2318 --overlap /data/vds/env_pt/bin/python train_oct5k.py \
    --data_dir ${OCT5K_DIR} \
    --weights_path "checkpoints/RETFound_mae_natureOCT.pth" \
    --img_size 256 --gpu 1 --freeze_backbone --use_augmentations \
    --run_name "RETFound_OCT5k_Frozen_Aug_256" > logs/train_oct5k_frozen_aug_256.log 2>&1 &

echo "All 8 training runs launched in parallel."
