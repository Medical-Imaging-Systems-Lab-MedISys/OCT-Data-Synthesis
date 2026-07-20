#!/bin/bash
echo "Starting test Jupyter Lab deployment..."
# 1. Submit 4 jobs
ssh tanuh "cd /data/vds/mmk/Codes/oct_data_synthesis && sbatch -J loraFT_1 tools/launch_jupyter.sh 8881 && sbatch -J loraFT_2 tools/launch_jupyter.sh 8882 && sbatch -J loraFT_3 tools/launch_jupyter.sh 8883 && sbatch -J loraFT_4 tools/launch_jupyter.sh 8884"

echo "Waiting 20 seconds for node allocation..."
sleep 20

# 2. Get nodes
NODE1=$(ssh tanuh "squeue -u \$USER -n loraFT_1 -h -o %N")
NODE2=$(ssh tanuh "squeue -u \$USER -n loraFT_2 -h -o %N")
NODE3=$(ssh tanuh "squeue -u \$USER -n loraFT_3 -h -o %N")
NODE4=$(ssh tanuh "squeue -u \$USER -n loraFT_4 -h -o %N")
echo "Nodes: 1=$NODE1, 2=$NODE2, 3=$NODE3, 4=$NODE4"

# 3. Setup local forwarding from tanuh nodes to localhost
pkill -f "8881:.*:8881" || true
ssh -N -f -L 8881:${NODE1}:8881 -L 8882:${NODE2}:8882 -L 8883:${NODE3}:8883 -L 8884:${NODE4}:8884 tanuh

# 4. Setup remote forwarding to mohankumar@10.72.38.239
echo "Setting up remote forwarding to mohankumar@10.72.38.239..."
ssh -N -f -R 8881:localhost:8881 -R 8882:localhost:8882 -R 8883:localhost:8883 -R 8884:localhost:8884 mohankumar@10.72.38.239

echo "Test jobs running and ports forwarded. Sleeping for 10 minutes..."
sleep 600

echo "10 minutes passed. Canceling jobs..."
ssh tanuh "scancel -n loraFT_1 -n loraFT_2 -n loraFT_3 -n loraFT_4"
pkill -f "8881:localhost:8881" || true
pkill -f "8881:.*:8881" || true
echo "Test complete and cleaned up."
