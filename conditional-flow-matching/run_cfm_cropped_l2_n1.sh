#!/bin/bash
#SBATCH --job-name=cfm_l2_n1
#SBATCH --nodes=1
#SBATCH --nodelist=n1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=normal
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/cfm_cropped_l2_%j.out
#SBATCH --error=logs/cfm_cropped_l2_%j.err
#SBATCH --chdir=/data/vds/mmk/Codes/oct_data_synthesis/

module purge
module load Miniforge3/26.1.1-3 2>/dev/null || true
source activate /data/vds/env_pt 2>/dev/null || source /data/vds/env_pt/bin/activate

export PYTHONUNBUFFERED=1
mkdir -p logs
python -u conditional-flow-matching/train_cropped_multidataset.py --loss_type l2 --pseudo_weight 0.5
