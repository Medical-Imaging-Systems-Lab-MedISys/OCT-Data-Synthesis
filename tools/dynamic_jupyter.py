import subprocess
import re
import time
import sys
import os

LOG_FILE = "/home/mmk/Codes/oct_data_synthesis/tools/dynamic_jupyter.log"

def log(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(formatted + "\n")
    except Exception as e:
        print(f"Failed to write to log file: {e}")

def check_online():
    try:
        # Check SSH connectivity to the server
        res = subprocess.run(
            ["ssh", "-q", "-o", "ConnectTimeout=10", "tanuh", "echo online"],
            capture_output=True, text=True, timeout=15
        )
        return res.returncode == 0
    except Exception:
        return False

def get_node_gpus():
    try:
        res = subprocess.run(
            ["ssh", "tanuh", "sinfo -o '%N %G'"],
            capture_output=True, text=True, check=True
        )
        lines = res.stdout.strip().split('\n')[1:] # Skip header
        nodes = {}
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                node_name = parts[0]
                gres = parts[1]
                # Extract GPU count
                match = re.search(r'gpu:\w+:(\d+)|gpu:(\d+)', gres)
                if match:
                     count = int(match.group(1) or match.group(2))
                     nodes[node_name] = count
                else:
                     nodes[node_name] = 0
        return nodes
    except Exception as e:
        log(f"Error querying sinfo: {e}")
        return {}

def main():
    log("Starting scheduled dynamic Jupyter GPU reservation...")
    
    # 1. Wait for server to be online (retry every 15 minutes)
    while not check_online():
        log("Server 10.16.63.40 (tanuh) is offline. Retrying in 15 minutes...")
        time.sleep(15 * 60)
        
    log("Server 10.16.63.40 (tanuh) is online!")
    
    # 2. Query node GPU configurations
    nodes = get_node_gpus()
    log(f"Detected GPU counts: {nodes}")
    
    # 3. Determine reservation details
    reservations = {'n1': 2, 'n2': 2}
    
    # Check for other nodes (e.g. n3, n4)
    other_nodes = [node for node in nodes if node not in ('n1', 'n2') and nodes[node] > 0]
    
    if other_nodes:
        # Lock 2 on n1, 2 on n2, and 1 on the first new node (e.g. n3)
        target_other = other_nodes[0]
        reservations[target_other] = 1
        log(f"Found new node '{target_other}'. Setting reservations: n1=2, n2=2, {target_other}=1")
    else:
        n1_total = nodes.get('n1', 0)
        n2_total = nodes.get('n2', 0)
        
        if n1_total >= 8 and n2_total >= 8:
            reservations['n1'] = 3
            reservations['n2'] = 3
            log("Both n1 and n2 have >=8 GPUs. Setting reservations: n1=3, n2=3")
        elif n1_total >= 8:
            reservations['n1'] = 3
            # n2 has fewer GPUs, keep at 2
            reservations['n2'] = 2
            log("n1 has >=8 GPUs. Setting reservations: n1=3, n2=2")
        elif n2_total >= 8:
            # n2 has 8 GPUs (H200 NVL) — always use 3 for 3 parallel training slots
            reservations['n1'] = 2
            reservations['n2'] = 3
            log("n2 has >=8 GPUs (H200 NVL). Setting reservations: n1=2, n2=3")
        else:
            reservations['n1'] = 2
            reservations['n2'] = 2
            log("n1 and n2 both have <=4 GPUs. Setting default reservations: n1=2, n2=2")

    # 4. Clean up old Jupyter/loraFT SLURM jobs from user
    log("Cleaning up old Jupyter SLURM jobs on the server...")
    try:
        res = subprocess.run(
            ["ssh", "tanuh", "squeue -u mohan.manepalli -h -o %i,%j"],
            capture_output=True, text=True, check=True
        )
        job_lines = res.stdout.strip().split('\n')
        jobs_to_cancel = []
        for line in job_lines:
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                job_id = parts[0]
                job_name = parts[1]
                if any(prefix in job_name for prefix in ('loraFT_', 'jupyter_')):
                    jobs_to_cancel.append(job_id)
        
        if jobs_to_cancel:
            log(f"Cancelling existing jobs: {jobs_to_cancel}")
            subprocess.run(["ssh", "tanuh", f"scancel {','.join(jobs_to_cancel)}"], check=True)
        else:
            log("No existing Jupyter jobs found to cancel.")
    except Exception as e:
        log(f"Error during job cleanup: {e}")

    # Clean up local port forwards
    log("Killing existing port forwards...")
    subprocess.run("kill -9 $(ss -tlnp | grep -E '8881|8882|8883|8884' | grep -o -E 'pid=[0-9]+' | cut -d= -f2) 2>/dev/null || true", shell=True)


    # 5. Launch new Jupyter jobs
    ports = [8881, 8882, 8883, 8884]
    port_idx = 0
    submitted_jobs = []
    
    sorted_nodes = sorted(reservations.keys())
    
    for node in sorted_nodes:
        gres_count = reservations[node]
        if gres_count <= 0:
            continue
        
        port = ports[port_idx]
        port_idx += 1
        
        job_name = f"jupyter_{node}"
        log(f"Launching Jupyter on {node} (GPU count: {gres_count}) on port {port}...")
        
        try:
            cmd = [
                "ssh", "tanuh",
                f"cd /data/vds/mmk/Codes/oct_data_synthesis && sbatch -w {node} -J {job_name} --gres=gpu:{gres_count} tools/launch_jupyter.sh {port}"
            ]
            subprocess.run(cmd, check=True)
            submitted_jobs.append((node, job_name, port))
        except Exception as e:
            log(f"Failed to submit job for node {node}: {e}")

    if not submitted_jobs:
        log("No jobs were successfully submitted. Exiting.")
        return

    log("Waiting 30 seconds for SLURM node allocation...")
    time.sleep(30)

    # 6. Retrieve allocated node names and build port forwarding
    forward_mappings = []
    for node, job_name, port in submitted_jobs:
        try:
            res = subprocess.run(
                ["ssh", "tanuh", f"squeue -u mohan.manepalli -n {job_name} -h -o %N"],
                capture_output=True, text=True, check=True
            )
            allocated_node = res.stdout.strip()
            if allocated_node:
                log(f"Job '{job_name}' allocated to node: {allocated_node}")
                forward_mappings.append((port, allocated_node))
            else:
                log(f"Job '{job_name}' is still pending or was not found.")
        except Exception as e:
            log(f"Error querying allocated node for {job_name}: {e}")

    if not forward_mappings:
        log("No allocated nodes found for port forwarding. Exiting.")
        return

    # 7. Setup SSH Port Forwarding
    local_fw_cmd = ["ssh", "-N", "-f"]
    for port, allocated_node in forward_mappings:
        local_fw_cmd.extend(["-L", f"{port}:{allocated_node}:{port}"])
    local_fw_cmd.append("tanuh")
    
    log(f"Running local port forwarding command: {' '.join(local_fw_cmd)}")
    try:
        subprocess.run(local_fw_cmd, check=True)
    except Exception as e:
        log(f"Error setting up local port forwarding: {e}")

    remote_fw_cmd = ["ssh", "-N", "-f"]
    for port, _ in forward_mappings:
        remote_fw_cmd.extend(["-R", f"{port}:localhost:{port}"])
    remote_fw_cmd.append("mohankumar@10.72.38.239")
    
    log(f"Running remote port forwarding command: {' '.join(remote_fw_cmd)}")
    try:
        subprocess.run(remote_fw_cmd, check=True)
    except Exception as e:
        log(f"Error setting up remote port forwarding: {e}")

    log("Dynamic Jupyter GPU reservation and forwarding deployment completed successfully.")

if __name__ == "__main__":
    main()
