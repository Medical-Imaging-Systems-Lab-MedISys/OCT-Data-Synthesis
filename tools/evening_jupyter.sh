#!/bin/bash
DAY=$(date +%a)
echo "Starting Evening Jupyter Lab deployment on node n2 for $DAY..."

# Clean up old evening forwards
pkill -f "8891:.*:8891" || true
pkill -f "8892:.*:8892" || true
pkill -f "8891:localhost:8891" || true
pkill -f "8892:localhost:8892" || true

# 1. Submit 2 jobs to node n2
ssh tanuh "cd /data/vds/mmk/Codes/oct_data_synthesis && sbatch -w n2 -J loraFT_PM_${DAY}_1 tools/launch_jupyter.sh 8891 && sbatch -w n2 -J loraFT_PM_${DAY}_2 tools/launch_jupyter.sh 8892"

echo "Waiting 30 seconds for node allocation..."
sleep 30

# 2. Get nodes
NODE1=$(ssh tanuh "squeue -u \$USER -n loraFT_PM_${DAY}_1 -h -o %N")
NODE2=$(ssh tanuh "squeue -u \$USER -n loraFT_PM_${DAY}_2 -h -o %N")
echo "Nodes: 1=$NODE1, 2=$NODE2"

# 3. Setup local forwarding from tanuh nodes to localhost
ssh -N -f -L 8891:${NODE1}:8891 -L 8892:${NODE2}:8892 tanuh

# 4. Setup remote forwarding to mohankumar
ssh -N -f -R 8891:localhost:8891 -R 8892:localhost:8892 mohankumar@10.72.38.239

echo "Evening deployment completed successfully."
