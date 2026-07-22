#!/bin/bash
#SBATCH --job-name=cfm_all_aug_n1
#SBATCH --nodes=1
#SBATCH --nodelist=n1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=normal
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/cfm_all_aug_%j.out
#SBATCH --error=logs/cfm_all_aug_%j.err
#SBATCH --chdir=/home/mmk/Codes/oct_data_synthesis

module purge
module load Miniforge3/26.1.1-3 2>/dev/null || true
source activate /data/vds/env_pt 2>/dev/null || source /data/vds/env_pt/bin/activate || source activate pytorch-env 2>/dev/null

export PYTHONUNBUFFERED=1
mkdir -p logs

echo "=========================================================="
echo "RUN 1: L1 Loss with Spatial Weights"
echo "=========================================================="
python -u conditional-flow-matching/train_on_augmented.py --loss_type l1

echo "=========================================================="
echo "RUN 2: L1 Loss without Spatial Weights"
echo "=========================================================="
python -u conditional-flow-matching/train_on_augmented.py --loss_type l1 --no_spatial_weighting

echo "=========================================================="
echo "RUN 3: L2 Loss with Spatial Weights"
echo "=========================================================="
python -u conditional-flow-matching/train_on_augmented.py --loss_type l2

echo "=========================================================="
echo "RUN 4: L2 Loss without Spatial Weights"
echo "=========================================================="
python -u conditional-flow-matching/train_on_augmented.py --loss_type l2 --no_spatial_weighting

echo "All 4 training runs completed successfully!"
