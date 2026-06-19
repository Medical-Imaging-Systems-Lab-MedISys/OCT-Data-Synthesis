# System Instructions: Tanuh AI Cluster Operational Manual
**Target Audience:** Autonomous Local Agents, Coding Assistants, and Deployment Scripts operating on behalf of the user.
**Objective:** Provide strict operational boundaries, execution protocols, and known environmental quirks for interacting with the Tanuh AI Cluster.

## 1. System Architecture & Hardware Definitions
You are operating on a high-performance computing (HPC) cluster managed by Slurm. The system consists of a master node (login node) and dedicated compute nodes.
* **Master Node:** Strictly for file management, code editing, SSH/SCP transfers, and job submission. **Zero compute workloads are permitted here.**
* **Node `n1`:** Equipped with Nvidia RTX Pro 6000 GPUs.
* **Node `n2`:** Equipped with Nvidia H200 GPUs (Optimal for Large Language Model Full Parameter Fine-Tuning and large-scale MLOps).

## 2. Hard Constraints (CRITICAL)
If you are generating bash commands, deployment scripts, or executing terminal tasks, you MUST obey the following rules:
* **NEVER use `conda init`.** Conda environments are managed dynamically via LMod.
* **NEVER run execution binaries (e.g., `python train.py`, `make`, `./hashcat`) directly on the master node.** This will crash the session. Always use `srun` or `sbatch`.
* **NEVER assume internet access on compute nodes.** Hugging Face models and large datasets must be pre-downloaded to network storage and referenced via local paths.
* **NEVER rely on the base OS Python or system libraries.** The host OS uses an outdated `glibc` (GLIBC_2.29/2.17 era). All dependencies must be isolated in Conda environments or pre-compiled binaries.

## 3. Environment Initialization (LMod + Conda)
To load environments in any shell or Slurm script, you must explicitly purge previous modules, load the Miniforge module, and activate the specific shared team environment.

**Standard Initialization Block:**

```

```text
Generated tanuh_ai_agent_manual.md

```bash
module purge
module load Miniforge3/26.1.1-3
source activate /data/team-xxx/<target_environment>

```

**Known Workspaces:**

* `/data/team-xxx/env_pt`: For Medical Image Segmentation (MONAI, PyTorch, SciML, Phys-MFFDA).
* `/data/team-xxx/env_llm`: For bleeding-edge LLM tasks (Hugging Face `transformers`, `peft`, `trl`, Gemma 3 architectures).
* `/data/team-xxx/env_tools`: For system utilities bypassing OS limitations (e.g., `rsync`).

## 4. Slurm Job Submission Protocols

All training, cracking, or processing tasks must be wrapped in a Slurm batch script (`sbatch`) or an interactive run (`srun`).

**Standard `sbatch` Template (e.g., for H200 LLM Training):**

```bash
#!/bin/bash
#SBATCH --job-name=agent_task
#SBATCH --nodes=1
#SBATCH --nodelist=n2              # Target specific hardware if needed
#SBATCH --gres=gpu:1               # Request GPU resources
#SBATCH --partition=normal
#SBATCH --time=24:00:00

module purge
module load Miniforge3/26.1.1-3
source activate /data/team-xxx/env_llm

# Task execution goes here
srun python script.py

```

## 5. Storage Staging & I/O Optimization

Do not train models directly off the `/data` or `/home` network drives (NFS). You must stage data to the local NVMe `/tmp` drive of the allocated compute node and clean it up afterward.

**Required Staging Workflow within Scripts:**

```bash
PERSISTENT_DATA="/data/team-xxx/Dataset"
LOCAL_SCRATCH="/tmp/$USER_job_$SLURM_JOB_ID"

# 1. Stage Data
mkdir -p $LOCAL_SCRATCH
cp -r $PERSISTENT_DATA/* $LOCAL_SCRATCH/

# 2. Execute (pointing to LOCAL_SCRATCH)
srun python train.py --data_dir $LOCAL_SCRATCH

# 3. Cleanup (Mandatory)
rm -rf $LOCAL_SCRATCH

```

## 6. Known Edge Cases & System Quirks

When debugging errors, check these known cluster-specific issues first:

* **Hugging Face `float8` AttributeError:** If `transformers` crashes looking for `torch.float8_e8m0fnu`, inject a monkey patch at the top of the Python script: `if not hasattr(torch, "float8_e8m0fnu"): setattr(torch, "float8_e8m0fnu", torch.float32)`.
* **C/C++ Linker Crashes (`ld returned 1 exit status` / `.ltrans.o` errors):** The cluster's system linker does not support Link Time Optimization (LTO) via GCC 12. If compiling software from source (like hashcat), you MUST append `ENABLE_LTO=0` to the `make` command.
* **GLIBC Version Mismatches:** If a pre-compiled Linux binary fails with a GLIBC error, do not attempt to upgrade the OS. Instead, pull the package through `conda-forge` inside a Miniforge environment to sandbox modern C libraries.
* **Missing Rsync:** Execute `rsync` using the explicit path to the Conda-installed binary on the cluster: `--rsync-path="/data/team-xxx/env_tools/bin/rsync"`.
