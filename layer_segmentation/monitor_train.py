import subprocess
import time
import os

LOG_FILE = "/data/vds/mmk/Codes/oct_data_synthesis/layer_segmentation/logs/auto_eval_256.log"

def is_running():
    try:
        # Check active processes on the system
        res = subprocess.run(["pgrep", "-f", "train_nr206.py"], capture_output=True, text=True)
        res2 = subprocess.run(["pgrep", "-f", "train_oct5k.py"], capture_output=True, text=True)
        
        # Filter out our own monitor_train.py PID or pgrep processes from the lines
        my_pid = os.getpid()
        
        pids = []
        for out in [res.stdout, res2.stdout]:
            for line in out.strip().split('\n'):
                if line:
                    try:
                        pid = int(line.split()[0])
                        if pid != my_pid:
                            pids.append(pid)
                    except ValueError:
                        continue
        return len(pids) > 0
    except Exception as e:
        print(f"Error checking processes: {e}")
        return False

def main():
    print(f"Starting python monitor process (PID: {os.getpid()})...")
    with open(LOG_FILE, "w") as f:
        f.write("Waiting for 256-input training processes to finish...\n")
        f.flush()
        
    while is_running():
        time.sleep(30)
        
    with open(LOG_FILE, "a") as f:
        f.write("Training complete! Launching 256-input cross-evaluations...\n")
        f.flush()
        
    # Execute the cross evaluation
    try:
        cmd = ["/data/vds/env_pt/bin/python", "/data/vds/mmk/Codes/oct_data_synthesis/layer_segmentation/run_cross_eval_256.py"]
        with open(LOG_FILE, "a") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
            f.write("All evaluations finished successfully!\n")
    except Exception as e:
        with open(LOG_FILE, "a") as f:
            f.write(f"Error during cross-evaluation: {e}\n")

if __name__ == '__main__':
    main()
