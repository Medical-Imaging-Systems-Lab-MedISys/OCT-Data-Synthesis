#!/bin/bash

echo "Starting automated Jupyter Lab deployment on tanuh..."

# 1. Submit 4 jobs via SSH
ssh tanuh "cd /data/vds/mmk/Codes/oct_data_synthesis && sbatch -J loraFT_1 tools/launch_jupyter.sh 8881 && sbatch -J loraFT_2 tools/launch_jupyter.sh 8882 && sbatch -J loraFT_3 tools/launch_jupyter.sh 8883 && sbatch -J loraFT_4 tools/launch_jupyter.sh 8884"

echo "Jobs submitted. Waiting 30 seconds for SLURM node allocation..."
sleep 30

# 2. Retrieve the assigned compute nodes
NODE1=$(ssh tanuh "squeue -u \$USER -n loraFT_1 -h -o %N")
NODE2=$(ssh tanuh "squeue -u \$USER -n loraFT_2 -h -o %N")
NODE3=$(ssh tanuh "squeue -u \$USER -n loraFT_3 -h -o %N")
NODE4=$(ssh tanuh "squeue -u \$USER -n loraFT_4 -h -o %N")

echo "Nodes allocated: 1=$NODE1, 2=$NODE2, 3=$NODE3, 4=$NODE4"

# Validate that we successfully got node names (not empty and not 'PD' meaning pending)
if [[ -z "$NODE1" || "$NODE1" == *"PD"* || "$NODE1" == *"("* ]]; then
    echo "Warning: Jobs are still pending or no node assigned. Port forwarding may fail or need to be done manually later."
    # We could implement a loop to wait, but for now we'll just try to forward if possible
fi

# 3. Terminate any existing port forwards for these ports to avoid conflicts
pkill -f "8881:.*:8881"
pkill -f "8882:.*:8882"
pkill -f "8883:.*:8883"
pkill -f "8884:.*:8884"

# 4. Set up the local SSH tunnels dynamically based on the nodes assigned
echo "Setting up SSH Tunnels to the assigned compute nodes..."
ssh -N -f -L 8881:${NODE1}:8881 -L 8882:${NODE2}:8882 -L 8883:${NODE3}:8883 -L 8884:${NODE4}:8884 tanuh

echo "Done! The notebooks should now be accessible locally on ports 8881-8884."
