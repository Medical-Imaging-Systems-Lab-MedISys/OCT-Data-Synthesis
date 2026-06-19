# Retinal OCT Synthetic Data Generation & Translation Pipeline

This repository contains procedural models and Deep Learning pipelines (cGAN & Pix2Pix) to generate and translate synthetic Retinal Optical Coherence Tomography (OCT) images matching the anatomy of the `NR206` dataset.

---

## 1. Project Directory Structure

The project has been organized into a clean, modular structure:

```
oct_data_synthesis/
├── docs/                             # Project manuals and guidelines
│   └── server_instruction_manual.md  # SSH cluster and SLURM instructions
├── models/                           # Machine Learning translation models
│   ├── cgan_linear/                  # Baseline Linear Conditional GAN
│   │   ├── conditional-GAN-generating-NR206.py
│   │   ├── conditional-GAN-generating-fashion-mnist.py
│   │   ├── config.json
│   │   └── run_sbatch.sh
│   └── pix2pix/                      # Pix2Pix cGAN (U-Net & PatchGAN)
│       ├── train_pix2pix.py
│       ├── config_pix2pix.json
│       └── run_pix2pix_sbatch.sh
├── synthesis/                        # Procedural synthesis engines
│   ├── generate_synthetic_oct.py     # Noisy procedural generator (Rayleigh speckle)
│   └── generate_synthetic_oct_clean.py # Clean procedural generator (smooth gradients)
└── tools/                            # Utility and profiling web tools
    └── intensity_profiler/           # Flask app for multi-line intensity profiles
        ├── app.py
        └── templates/index.html
```

---

## 2. Procedural Data Synthesis

The synthesis models procedurally construct retinal structures with 8 distinct layers (e.g. ILM, NFL, IPL/INL, OPL, ONL, ELM/IS, OS/RPE, RPE/Chor, and Sclera) following mathematical baseline configurations fitted from `NR206`.

### Generate Noisy (Speckled) Dataset
```bash
python synthesis/generate_synthetic_oct.py --count 100 --min-gamma 0.5 --max-gamma 1.5 --output-dir synthetic_dataset
```

### Generate Clean (Smooth Gradient) Dataset
```bash
python synthesis/generate_synthetic_oct_clean.py --count 100 --min-gamma 0.5 --max-gamma 1.5 --output-dir synthetic_dataset_clean
```

---

## 3. Pix2Pix GAN Translation Model

The Pix2Pix conditional GAN maps synthetic speckled inputs (procedurally generated from real masks) to real-looking OCT outputs.

### Architecture
- **Generator**: A deep **U-Net** model with direct skip connections between mirroring layers in the encoder and decoder to retain high-frequency layer boundary structures.
- **Discriminator**: A **PatchGAN** classifier that operates on local image patches (70x70) to identify real/fake pairs, focusing model learning on speckle texture realism.

### Configuration (`models/pix2pix/config_pix2pix.json`)
Allows custom adjustments for image sizes (`128`/`256`), training parameters (`batch_size`, `epochs`, `learning_rate`), and L1 reconstruction loss weighting (`lambda_L1: 100.0`).

### Verification check
To verify that network architectures, shapes, and GPU forward/backward loops compile successfully, run:
```bash
python models/pix2pix/train_pix2pix.py --verify
```

---

## 4. Running on the SLURM Cluster (node `n1`)

Both translation models include custom SLURM batch execution scripts that handle module initialization, environment loading, and **Local NVMe SSD Staging** to maximize GPU training speed.

### SSD Staging Strategy (Inside the sbatch scripts)
1. Creates a temporary scratch path on the node's local high-speed SSD (`/tmp/`).
2. Copies the dataset files to the SSD.
3. Automatically updates the configuration files using `sed` to point the python dataloaders to the SSD copy.
4. Executes training.
5. Cleans up the scratch SSD and restores configuration files upon completion/exit.

### Launching Training Jobs

To submit jobs targeting the **NVIDIA RTX PRO 6000 Blackwell GPU** on node `n1`:

#### Train Pix2Pix GAN (1 GPU)
```bash
sbatch models/pix2pix/run_pix2pix_sbatch.sh
```

#### Train Linear cGAN (4 GPUs)
```bash
sbatch models/cgan_linear/run_sbatch.sh
```

To monitor your running jobs:
```bash
squeue -u $USER
```
