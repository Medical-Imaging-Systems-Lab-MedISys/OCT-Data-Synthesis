import mlflow
import subprocess
import os
import sys

# Define tracking URI
mlflow.set_tracking_uri('http://10.24.38.15:5000')

def get_run_id(run_name):
    # Query MLflow for the run ID matching run_name
    for exp_name in ['NR206_Segmentation', 'OCT5k_Segmentation']:
        try:
            df = mlflow.search_runs(experiment_names=[exp_name])
            if 'tags.mlflow.runName' in df.columns:
                match = df[df['tags.mlflow.runName'] == run_name]
                if not match.empty:
                    # Filter for active or completed runs
                    return match.iloc[0]['run_id']
        except Exception as e:
            continue
    return None

def main():
    print("Starting master cross-evaluation launcher for 256-input size models...")
    
    NR206_DIR = "/data/vds/mmk/Codes/oct_data_synthesis/DATA/NR206"
    OCT5K_DIR = "/data/vds/mmk/Codes/oct_data_synthesis/DATA/OCT5k/OCT5k_split"
    
    # 8 evaluation configurations:
    # (source_run_name, model_type, source_dataset, target_dataset, data_dir, output_dir_suffix, run_name, job_id, gpu_id)
    evals = [
        # OCT5k models evaluated on NR206 (4 runs, Job 2316 + Job 2319)
        ("RETFound_OCT5k_Finetune_Aug_256", "finetune", "oct5k", "nr206", NR206_DIR, "predictions_256/OCT5k_models_on_NR206/finetune_aug", "OCT5k_Finetune_Aug_256_on_NR206", "2316", "0"),
        ("RETFound_OCT5k_Frozen_Aug_256", "frozen", "oct5k", "nr206", NR206_DIR, "predictions_256/OCT5k_models_on_NR206/frozen_aug", "OCT5k_Frozen_Aug_256_on_NR206", "2316", "1"),
        ("RETFound_OCT5k_Finetune_256", "finetune", "oct5k", "nr206", NR206_DIR, "predictions_256/OCT5k_models_on_NR206/finetune_no_aug", "OCT5k_Finetune_256_on_NR206", "2319", "0"),
        ("RETFound_OCT5k_Frozen_256", "frozen", "oct5k", "nr206", NR206_DIR, "predictions_256/OCT5k_models_on_NR206/frozen_no_aug", "OCT5k_Frozen_256_on_NR206", "2319", "1"),
        
        # NR206 models evaluated on OCT5k (4 runs, Job 2320 + Job 2318)
        ("RETFound_NR206_Finetune_WM_Removed_Aug_256", "finetune", "nr206", "oct5k", OCT5K_DIR, "predictions_256/NR206_models_on_OCT5k/finetune_aug", "NR206_Finetune_WM_Removed_Aug_256_on_OCT5k", "2320", "0"),
        ("RETFound_NR206_Frozen_WM_Removed_Aug_256", "frozen", "nr206", "oct5k", OCT5K_DIR, "predictions_256/NR206_models_on_OCT5k/frozen_aug", "NR206_Frozen_WM_Removed_Aug_256_on_OCT5k", "2320", "1"),
        ("RETFound_NR206_Finetune_WM_Removed_256", "finetune", "nr206", "oct5k", OCT5K_DIR, "predictions_256/NR206_models_on_OCT5k/finetune_no_aug", "NR206_Finetune_WM_Removed_256_on_OCT5k", "2318", "0"),
        ("RETFound_NR206_Frozen_WM_Removed_256", "frozen", "nr206", "oct5k", OCT5K_DIR, "predictions_256/NR206_models_on_OCT5k/frozen_no_aug", "NR206_Frozen_WM_Removed_256_on_OCT5k", "2318", "1")
    ]
    
    processes = []
    
    for src_run, model_type, src_ds, tgt_ds, data_dir, out_suffix, run_name, job_id, gpu_id in evals:
        # Retrieve Run ID
        run_id = get_run_id(src_run)
        if not run_id:
            print(f"Error: Could not find Run ID for completed run: '{src_run}'. Skipping evaluation.")
            continue
            
        print(f"Found Run ID for {src_run}: {run_id}. Preparing cross-evaluation...")
        
        output_dir = f"/data/vds/mmk/Codes/oct_data_synthesis/layer_segmentation/{out_suffix}"
        
        cmd = [
            "srun", f"--jobid={job_id}", "--overlap",
            "/data/vds/env_pt/bin/python", "/data/vds/mmk/Codes/oct_data_synthesis/layer_segmentation/cross_eval.py",
            "--run_id", run_id,
            "--model_type", model_type,
            "--source_dataset", src_ds,
            "--target_dataset", tgt_ds,
            "--data_dir", data_dir,
            "--output_dir", output_dir,
            "--run_name", run_name,
            "--img_size", "256",
            "--gpu", gpu_id
        ]
        
        log_file = f"/data/vds/mmk/Codes/oct_data_synthesis/layer_segmentation/logs/cross_eval_256_{run_name}.log"
        print(f"Launching command on Job {job_id} GPU {gpu_id}: {' '.join(cmd)}")
        
        # Spawn processes concurrently
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        
        f_log = open(log_file, "w")
        p = subprocess.Popen(cmd, env=env, stdout=f_log, stderr=subprocess.STDOUT)
        processes.append((p, run_name, f_log))
        
    # Wait for all processes to finish
    for p, run_name, f_log in processes:
        p.wait()
        f_log.close()
        print(f"Finished evaluation: {run_name} (Exit code: {p.returncode})")
        
    print("All 256-input cross-evaluations complete.")

if __name__ == '__main__':
    main()
