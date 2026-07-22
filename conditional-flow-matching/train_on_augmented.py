#!/usr/bin/env python3
import os
import sys
import argparse
import datetime
import random
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt

# Seed everything for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Ensure conditional-flow-matching directory is in python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from diffusers import UNet2DModel
from train_val import synthesize_from_mask

# 4-Column Validation Grid Logger
def log_validation_grids_4col(x1_orig, x0, x1_gen, x1_gt, epoch, batch_idx):
    """
    Logs a 4-column figure:
    1. Original (Preprocessed & Resized)
    2. Prior (Synthetic Mask Scan)
    3. Generated Synthesis
    4. Ground Truth (Target Shifted Scan)
    """
    x1_orig_disp = (x1_orig.squeeze().cpu().numpy() + 1.0) / 2.0
    x0_disp = (x0.squeeze().cpu().numpy() + 1.0) / 2.0
    x1_gen_disp = (x1_gen.squeeze().cpu().numpy() + 1.0) / 2.0
    x1_gt_disp = (x1_gt.squeeze().cpu().numpy() + 1.0) / 2.0

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = [
        "Original (Preprocessed)",
        "Prior (Synthetic Mask)",
        "Generated Synthesis",
        "Ground Truth (Target)"
    ]
    images = [x1_orig_disp, x0_disp, x1_gen_disp, x1_gt_disp]

    for ax, title, img in zip(axes, titles, images):
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.imshow(img, cmap='gray', vmin=0.0, vmax=1.0)
        ax.axis('off')

    plt.tight_layout()
    mlflow.log_figure(fig, f"validation_grids/epoch_{epoch}_sample_{batch_idx}.png")
    plt.close(fig)

class AugmentedOCTDataset(Dataset):
    def __init__(self, images_dir, labels_dir, min_gamma=0.5, max_gamma=1.2):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        
        self.filenames = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
    def __len__(self):
        return len(self.filenames)
        
    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = os.path.join(self.images_dir, fname)
        lbl_path = os.path.join(self.labels_dir, fname)
        
        # Load Ground Truth (target shifted/bent image)
        x1_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if x1_img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        # Determine the name of the original preprocessed image
        parts = fname.split("_aug_")
        if len(parts) > 1:
            orig_fname = parts[0] + "_orig.png"
        else:
            orig_fname = fname
            
        orig_path = os.path.join(self.images_dir, orig_fname)
        x1_orig_img = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
        if x1_orig_img is None:
            # Fallback to current image if original not found
            x1_orig_img = x1_img
            
        # Load mask
        mask_bgra = cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED)
        if mask_bgra is None:
            raise FileNotFoundError(f"Label mask not found: {lbl_path}")
            
        if len(mask_bgra.shape) == 3 and mask_bgra.shape[2] == 3:
            alpha = np.full((mask_bgra.shape[0], mask_bgra.shape[1], 1), 255, dtype=np.uint8)
            mask_bgra = np.concatenate([mask_bgra, alpha], axis=2)
            
        # Synthesize prior x0 dynamically from mask
        x0_img_raw = synthesize_from_mask(mask_bgra, self.min_gamma, self.max_gamma)
        
        # Ensure correct target size (256x256)
        target_size = (256, 256)
        if x0_img_raw.shape[:2] != target_size:
            x0_img_raw = cv2.resize(x0_img_raw, target_size, interpolation=cv2.INTER_LINEAR)
        if x1_img.shape[:2] != target_size:
            x1_img = cv2.resize(x1_img, target_size, interpolation=cv2.INTER_LINEAR)
        if x1_orig_img.shape[:2] != target_size:
            x1_orig_img = cv2.resize(x1_orig_img, target_size, interpolation=cv2.INTER_LINEAR)
            
        # Binary layer mask for spatial weighting
        mask_squashed = cv2.resize(mask_bgra, target_size, interpolation=cv2.INTER_NEAREST)
        is_bg = (mask_squashed[:, :, 0] == 0) & (mask_squashed[:, :, 1] == 0) & (mask_squashed[:, :, 2] == 0)
        layer_mask = (~is_bg).astype(np.float32)
        
        # Normalize to [-1.0, 1.0]
        x0 = (x0_img_raw.astype(np.float32) / 127.5) - 1.0
        x1 = (x1_img.astype(np.float32) / 127.5) - 1.0
        x1_orig = (x1_orig_img.astype(np.float32) / 127.5) - 1.0
        
        x0_tensor = torch.from_numpy(x0).unsqueeze(0)
        x1_tensor = torch.from_numpy(x1).unsqueeze(0)
        x1_orig_tensor = torch.from_numpy(x1_orig).unsqueeze(0)
        mask_tensor = torch.from_numpy(layer_mask).unsqueeze(0)
        
        return x0_tensor, x1_tensor, x1_orig_tensor, mask_tensor

@torch.no_grad()
def generate_samples(model, x0, num_steps=50):
    # Support both wrapped (DataParallel) and raw models
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    raw_model.eval()
    
    x_t = x0.clone()
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t_val = i / num_steps
        t_batch = torch.full((x_t.shape[0],), t_val, device=x0.device)
        v_pred = raw_model(x_t, t_batch).sample
        x_t = x_t + v_pred * dt
    return x_t

def main():
    parser = argparse.ArgumentParser(description="Train CFM model on preprocessed augmented dataset")
    parser.add_argument("--loss_type", type=str, default="l2", choices=["l1", "l2"], help="Loss type: l1 or l2")
    parser.add_argument("--no_spatial_weighting", action="store_true", help="Disable spatial loss weighting")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "experiment_name": "OCT_CFM_8BitNorm_Augmented_Splits",
        "run_name": f"cfm_8bitnorm_aug_{args.loss_type.lower()}{'_nospatial' if args.no_spatial_weighting else '_spatial'}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        "loss_type": args.loss_type.lower(),
        "no_spatial_weighting": args.no_spatial_weighting,
        "mlflow_tracking_uri": "http://10.24.38.15:5000",
        "batch_size": 32, # Increased batch size because we train on 4 GPUs
        "epochs": 300,
        "learning_rate": 0.0002,
        "image_size": 256,
        "channels": 1,
        "val_check_interval": 5,
        "num_val_images": 5,
        "inference_steps": 50,
        "sigma": 0.0,
        "w_bg": 1.0 if args.no_spatial_weighting else 0.4,
        "w_layers": 1.0
    }

    # 1. Load Datasets
    repo_root = os.path.abspath(os.path.join(BASE_DIR, ".."))
    train_images = os.path.join(repo_root, "DATA/NR206_augmented/train/images")
    train_labels = os.path.join(repo_root, "DATA/NR206_augmented/train/labels")
    val_images = os.path.join(repo_root, "DATA/NR206_augmented/val/images")
    val_labels = os.path.join(repo_root, "DATA/NR206_augmented/val/labels")

    train_dataset = AugmentedOCTDataset(train_images, train_labels)
    val_dataset = AugmentedOCTDataset(val_images, val_labels)

    # Use more loaders/workers for speed
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    # 2. Model Configuration
    model = UNet2DModel(
        sample_size=config["image_size"],
        in_channels=config["channels"],
        out_channels=config["channels"],
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    ).to(device)

    # Multi-GPU support: wrap in nn.DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for parallel training!")
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    FM = ConditionalFlowMatcher(sigma=config["sigma"])

    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment(config["experiment_name"])

    print(f"Starting Training: {config['run_name']}...")

    with mlflow.start_run(run_name=config["run_name"]):
        mlflow.log_params(config)
        mlflow.set_tag("loss_type", args.loss_type.upper())
        mlflow.set_tag("spatial_weighting", "Disabled" if args.no_spatial_weighting else "Enabled")
        mlflow.set_tag("num_gpus", str(torch.cuda.device_count() if torch.cuda.is_available() else 0))

        for epoch in range(1, config["epochs"] + 1):
            model.train()
            total_train_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']}")
            for x0, x1, x1_orig, layer_mask in pbar:
                x0, x1, layer_mask = x0.to(device), x1.to(device), layer_mask.to(device)
                optimizer.zero_grad()

                t, x_t, u_t = FM.sample_location_and_conditional_flow(x0, x1)
                v_pred = model(x_t, t.squeeze()).sample

                # Calculate loss map
                if config["loss_type"] == "l1":
                    loss_map = torch.abs(v_pred - u_t)
                else:
                    loss_map = (v_pred - u_t) ** 2

                # Apply spatial loss weights
                spatial_weight = layer_mask * config["w_layers"] + (1.0 - layer_mask) * config["w_bg"]
                loss = (loss_map * spatial_weight).mean()

                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})

            avg_train_loss = total_train_loss / len(train_loader)
            mlflow.log_metric("train_loss_epoch", avg_train_loss, step=epoch)

            # Validation Loop
            model.eval()
            total_val_loss = 0.0
            logged_count = 0

            with torch.no_grad():
                for batch_idx, (val_x0, val_x1, val_x1_orig, val_mask) in enumerate(val_loader):
                    val_x0, val_x1, val_mask = val_x0.to(device), val_x1.to(device), val_mask.to(device)
                    t, x_t, u_t = FM.sample_location_and_conditional_flow(val_x0, val_x1)
                    v_pred = model(x_t, t.squeeze()).sample

                    if config["loss_type"] == "l1":
                        loss_map = torch.abs(v_pred - u_t)
                    else:
                        loss_map = (v_pred - u_t) ** 2

                    spatial_weight = val_mask * config["w_layers"] + (1.0 - val_mask) * config["w_bg"]
                    val_loss = (loss_map * spatial_weight).mean()
                    total_val_loss += val_loss.item()

                    # Log validation image grid (4 columns) every few epochs
                    if epoch % config["val_check_interval"] == 0 and logged_count < config["num_val_images"]:
                        x1_gen = generate_samples(model, val_x0, num_steps=config["inference_steps"])
                        # Select first sample of the batch to log
                        log_validation_grids_4col(
                            val_x1_orig[0],
                            val_x0[0],
                            x1_gen[0],
                            val_x1[0],
                            epoch,
                            batch_idx
                        )
                        logged_count += 1

                avg_val_loss = total_val_loss / len(val_loader)
                mlflow.log_metric("val_loss_epoch", avg_val_loss, step=epoch)
                print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save and log checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        checkpoint_path = f"checkpoints/cfm_model_{config['run_name']}.pt"
        
        # Save unwrapped model to ensure loadability without DataParallel wrapper
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        torch.save(raw_model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
        
        try:
            mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
            mlflow.pytorch.log_model(raw_model, "model", registered_model_name=f"CFM_8BitNorm_AugSplit_{args.loss_type.upper()}_Model")
            print("Successfully registered model in MLflow.")
        except Exception as e:
            print(f"Warning: Failed to log to MLflow: {e}")

if __name__ == "__main__":
    main()
