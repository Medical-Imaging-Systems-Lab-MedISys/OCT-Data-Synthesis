#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

PORT=$1
if [ -z "$PORT" ]; then
    PORT=8888
fi

echo "Starting Jupyter Lab on port $PORT"
echo "Running on node: $SLURMD_NODENAME"

# Activate environment
source /data/vds/env_pt/bin/activate || true

# Run jupyter lab
/data/vds/env_pt/bin/jupyter lab --ip=0.0.0.0 --port=$PORT --no-browser --ServerApp.token='' --ServerApp.password='' --ServerApp.allow_origin='*'
