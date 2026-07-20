#!/bin/bash
DAY=$(date +%a)
echo "Starting Morning Jupyter Lab deployment for $DAY..."

# Clean up old morning forwards
pkill -f "8881:.*:8881" || true
pkill -f "8882:.*:8882" || true
pkill -f "8881:localhost:8881" || true
pkill -f "8882:localhost:8882" || true

# 1. Submit 2 jobs to n1
ssh tanuh "cd /data/vds/mmk/Codes/oct_data_synthesis && sbatch -w n1 -J loraFT_${DAY}_1 tools/launch_jupyter.sh 8881 && sbatch -w n1 -J loraFT_${DAY}_2 tools/launch_jupyter.sh 8882"

echo "Waiting 30 seconds for node allocation..."
sleep 30

# 2. Get nodes
NODE1=$(ssh tanuh "squeue -u \$USER -n loraFT_${DAY}_1 -h -o %N")
NODE2=$(ssh tanuh "squeue -u \$USER -n loraFT_${DAY}_2 -h -o %N")
echo "Nodes: 1=$NODE1, 2=$NODE2"

# 3. Setup local forwarding from tanuh nodes to localhost
ssh -N -f -L 8881:${NODE1}:8881 -L 8882:${NODE2}:8882 tanuh

# 4. Setup remote forwarding to mohankumar
ssh -N -f -R 8881:localhost:8881 -R 8882:localhost:8882 mohankumar@10.72.38.239

echo "Morning deployment completed successfully."
