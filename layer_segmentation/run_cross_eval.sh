#!/bin/bash
#SBATCH --job-name=Cross_Eval_Segmentation
#SBATCH --output=logs/cross_eval_%j.out
#SBATCH --error=logs/cross_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1:00:00

echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"

cd /data/vds/mmk/Codes/oct_data_synthesis/layer_segmentation

# Define dataset roots on node
NR206_DIR="/data/vds/mmk/Codes/oct_data_synthesis/DATA/NR206"
OCT5K_DIR="/data/vds/mmk/Codes/oct_data_synthesis/DATA/OCT5k/OCT5k_split"

# Run 1: OCT5k Finetune Aug on NR206
echo "--------------------------------------------------"
echo "Running OCT5k Finetune Aug on NR206..."
/data/vds/env_pt/bin/python cross_eval.py \
    --run_id "1ccf1f6a197e4775a308ca76399d2cf4" \
    --model_type finetune \
    --source_dataset oct5k \
    --target_dataset nr206 \
    --data_dir ${NR206_DIR} \
    --output_dir "predictions/OCT5k_models_on_NR206/finetune" \
    --run_name "OCT5k_Finetune_Aug_on_NR206"

# Run 2: OCT5k Frozen Aug on NR206
echo "--------------------------------------------------"
echo "Running OCT5k Frozen Aug on NR206..."
/data/vds/env_pt/bin/python cross_eval.py \
    --run_id "9429c89b393d4032b3b8e6560bf95cc1" \
    --model_type frozen \
    --source_dataset oct5k \
    --target_dataset nr206 \
    --data_dir ${NR206_DIR} \
    --output_dir "predictions/OCT5k_models_on_NR206/frozen" \
    --run_name "OCT5k_Frozen_Aug_on_NR206"

# Run 3: NR206 Finetune WM Removed Aug on OCT5k
echo "--------------------------------------------------"
echo "Running NR206 Finetune WM Removed Aug on OCT5k..."
/data/vds/env_pt/bin/python cross_eval.py \
    --run_id "f61a994d99964b56bcad74323c25bcf2" \
    --model_type finetune \
    --source_dataset nr206 \
    --target_dataset oct5k \
    --data_dir ${OCT5K_DIR} \
    --output_dir "predictions/NR206_models_on_OCT5k/finetune" \
    --run_name "NR206_Finetune_WM_Removed_Aug_on_OCT5k"

# Run 4: NR206 Frozen WM Removed Aug on OCT5k
echo "--------------------------------------------------"
echo "Running NR206 Frozen WM Removed Aug on OCT5k..."
/data/vds/env_pt/bin/python cross_eval.py \
    --run_id "dfdd81e5464e48389318c3b96de000fc" \
    --model_type frozen \
    --source_dataset nr206 \
    --target_dataset oct5k \
    --data_dir ${OCT5K_DIR} \
    --output_dir "predictions/NR206_models_on_OCT5k/frozen" \
    --run_name "NR206_Frozen_WM_Removed_Aug_on_OCT5k"

echo "--------------------------------------------------"
echo "Cross evaluation complete."
