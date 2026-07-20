#!/usr/bin/env python3
"""
Reinforcement Learning Fine-Tuning (RLFT) for Conditional Flow Matching (CFM) Retinal OCT Synthesis
Author: Mohan Kumar Manepalli
Description:
    Fine-tunes the baseline CFM 8-bit model using Policy Gradient / Reward-Weighted Optimization.
    Incorporates a multi-objective reward model:
      1. Layer Alignment Reward (R_layer)
      2. Contrast & Structural Fidelity Reward (R_contrast)
      3. Speckle Noise Realism Reward (R_speckle)
"""

import os
import sys
import argparse
import datetime
import random
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import numpy as np
from tqdm import tqdm
import mlflow
import mlflow.pytorch

# Global Seeding
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from diffusers import UNet2DModel

# Import synthetic prior helper functions
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from train_val import synthesize_from_mask, log_validation_grids

# =====================================================================
# 1. Image Preprocessing: Watermark Removal & Curved Cropping
# =====================================================================
def crop_and_pad_curved(image, mask_bgra):
    H, W = image.shape[:2]
    is_bg = (mask_bgra[:, :, 0] == 0) & (mask_bgra[:, :, 1] == 0) & (mask_bgra[:, :, 2] == 0)
    is_retina = ~is_bg
    has_retina = np.any(is_retina, axis=0)
    b8 = np.full(W, H - 1, dtype=np.int32)
    if np.any(has_retina):
        b8[has_retina] = H - 1 - np.argmax(is_retina[::-1, :][:, has_retina], axis=0)
    
    b8 = np.clip(b8 + 3, 0, H - 1)
    max_y = np.max(b8[has_retina]) if np.any(has_retina) else H
    max_y = min(H, max_y + 5)
    
    cropped_h = max_y
    max_dim = max(cropped_h, W)
    pad_h = max_dim - cropped_h
    pad_w = max_dim - W
    
    safe_bottom = H - 20
    safe_top = max(0, safe_bottom - 50)
    bottom_patch = image[safe_top:safe_bottom]
    patch_height = bottom_patch.shape[0]
    
    tiles_needed = int(np.ceil(max_dim / patch_height)) if patch_height > 0 else 1
    tiles = []
    for i in range(tiles_needed):
        shift = np.random.randint(0, W) if W > 0 else 0
        shifted = np.roll(bottom_patch, shift, axis=1)
        if i % 2 == 1:
            shifted = np.flip(shifted, axis=0)
        tiles.append(shifted)
        
    tiled_bg = np.concatenate(tiles, axis=0)[:max_dim, :W]
    if pad_w > 0:
        if len(image.shape) == 3:
            tiled_bg = np.pad(tiled_bg, ((0, 0), (0, pad_w), (0, 0)), mode='symmetric')
        else:
            tiled_bg = np.pad(tiled_bg, ((0, 0), (0, pad_w)), mode='symmetric')
            
    y_coords = np.arange(max_dim)[:, None]
    keep_mask = y_coords <= b8[None, :]
    if pad_w > 0:
        keep_mask = np.pad(keep_mask, ((0, 0), (0, pad_w)), mode='constant', constant_values=False)
        
    cropped_img = image[:cropped_h]
    if len(image.shape) == 3:
        padded_img = np.pad(cropped_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        keep_mask_3d = np.expand_dims(keep_mask, axis=-1)
        final_image = np.where(keep_mask_3d, padded_img, tiled_bg)
    else:
        padded_img = np.pad(cropped_img, ((0, pad_h), (0, pad_w)), mode='constant')
        final_image = np.where(keep_mask, padded_img, tiled_bg)
        
    return final_image

# =====================================================================
# 2. Dataset Loader with Watermark Removal
# =====================================================================
class WeightedOCTDataset(Dataset):
    def __init__(self, labels_dir, real_dir, sample_weight=1.0, min_gamma=0.5, max_gamma=1.5):
        self.labels_dir = labels_dir
        self.real_dir = real_dir
        self.sample_weight = sample_weight
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        
        self.filenames = sorted([
            f for f in os.listdir(real_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        lbl_path = os.path.join(self.labels_dir, fname)
        real_path = os.path.join(self.real_dir, fname)
        
        x1_img = cv2.imread(real_path, cv2.IMREAD_GRAYSCALE)
        if x1_img is None:
            raise FileNotFoundError(f"Real image not found: {real_path}")
            
        # Dynamically remove watermark from bottom-left before curved cropping
        if x1_img.shape[0] >= 350 and x1_img.shape[1] >= 600:
            clean_patch = x1_img[350:, 600:]
            if clean_patch.size > 0:
                h_target = x1_img[350:, :150].shape[0]
                w_target = x1_img[350:, :150].shape[1]
                x1_img[350:, :150] = np.flip(clean_patch, axis=1)[:h_target, :w_target]

        mask_bgra = cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED)
        if mask_bgra is None:
            raise FileNotFoundError(f"Label mask not found: {lbl_path}")
            
        if len(mask_bgra.shape) == 3 and mask_bgra.shape[2] == 3:
            alpha = np.full((mask_bgra.shape[0], mask_bgra.shape[1], 1), 255, dtype=np.uint8)
            mask_bgra = np.concatenate([mask_bgra, alpha], axis=2)

        target_size = (256, 256)
        x0_img_raw = synthesize_from_mask(mask_bgra, self.min_gamma, self.max_gamma)
        x0_img_squashed = cv2.resize(x0_img_raw, target_size, interpolation=cv2.INTER_LINEAR)
        mask_squashed = cv2.resize(mask_bgra, target_size, interpolation=cv2.INTER_NEAREST)
        
        x0_cropped = crop_and_pad_curved(x0_img_squashed, mask_squashed)
        x1_img_squashed = cv2.resize(x1_img, target_size, interpolation=cv2.INTER_LINEAR)
        x1_cropped = crop_and_pad_curved(x1_img_squashed, mask_squashed)
        
        is_bg = (mask_squashed[:, :, 0] == 0) & (mask_squashed[:, :, 1] == 0) & (mask_squashed[:, :, 2] == 0)
        layer_mask = (~is_bg).astype(np.float32)
        
        x0 = (x0_cropped.astype(np.float32) / 127.5) - 1.0
        x1 = (x1_cropped.astype(np.float32) / 127.5) - 1.0
        
        x0_tensor = torch.from_numpy(x0).unsqueeze(0)
        x1_tensor = torch.from_numpy(x1).unsqueeze(0)
        mask_tensor = torch.from_numpy(layer_mask).unsqueeze(0)
        weight_tensor = torch.tensor([self.sample_weight], dtype=torch.float32)
        
        return x0_tensor, x1_tensor, mask_tensor, weight_tensor

# =====================================================================
# 3. Differentiable Multi-Objective Reward Function for OCT Scans
# =====================================================================
class OCTRewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Sobel kernel for edge & boundary detection
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, x_gen, x_gt, layer_mask):
        """
        Computes composite reward R for generated images [B, 1, H, W] in [-1, 1].
        """
        # 1. Layer Boundary Alignment Reward (R_layer)
        grad_x = F.conv2d(x_gen, self.sobel_x, padding=1)
        grad_y = F.conv2d(x_gen, self.sobel_y, padding=1)
        edge_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        r_layer = (edge_mag * layer_mask).mean(dim=[1, 2, 3])

        # 2. Contrast & Structural Fidelity Reward (R_contrast)
        retina_intensity = (x_gen * layer_mask).sum(dim=[1, 2, 3]) / (layer_mask.sum(dim=[1, 2, 3]) + 1e-6)
        bg_mask = 1.0 - layer_mask
        bg_intensity = (x_gen * bg_mask).sum(dim=[1, 2, 3]) / (bg_mask.sum(dim=[1, 2, 3]) + 1e-6)
        r_contrast = retina_intensity - bg_intensity

        # 3. Speckle Noise Realism Reward (R_speckle)
        # Correctly compute variance of ONLY the background region pixels (ignore zeroed-out mask areas)
        bg_mean = (x_gen * bg_mask).sum(dim=[1, 2, 3], keepdim=True) / (bg_mask.sum(dim=[1, 2, 3], keepdim=True) + 1e-6)
        bg_var = ((x_gen - bg_mean) ** 2 * bg_mask).sum(dim=[1, 2, 3]) / (bg_mask.sum(dim=[1, 2, 3]) + 1e-6)
        r_speckle = -torch.abs(bg_var - 0.05) # Penalize background noise variance deviation

        # Total Composite Reward Score
        r_total = 0.5 * r_layer + 0.3 * r_contrast + 0.2 * r_speckle
        return r_total, r_layer, r_contrast, r_speckle

# =====================================================================
# 4. ODE Rollout & Generator Functions
# =====================================================================
def generate_samples_ODE(model, x0, num_steps=50):
    model.eval()
    x_t = x0.clone()
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t_val = i / num_steps
        t_batch = torch.full((x_t.shape[0],), t_val, device=x0.device)
        v_pred = model(x_t, t_batch).sample
        x_t = x_t + v_pred * dt
    return x_t

# =====================================================================
# 5. Main RLFT Training Loop
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="RLFT Fine-Tuning for Baseline and Cropped CFM Models")
    parser.add_argument("--epochs", type=int, default=100, help="Number of RL fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for RL fine-tuning")
    parser.add_argument("--loss_type", type=str, default="l2", choices=["l1", "l2"], help="Base model loss type: l1 or l2")
    parser.add_argument("--baseline_checkpoint", type=str, default=None, help="Explicit path to baseline model checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve Checkpoint Path dynamically
    checkpoint_to_load = None
    if args.baseline_checkpoint and os.path.exists(args.baseline_checkpoint):
        checkpoint_to_load = args.baseline_checkpoint
    else:
        ckpt_dir = "checkpoints"
        if os.path.exists(ckpt_dir):
            matched = [
                os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir)
                if args.loss_type.lower() in f.lower() and f.endswith(".pt")
            ]
            if matched:
                matched.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                checkpoint_to_load = matched[0]
        
        if not checkpoint_to_load:
            default_fallback = "checkpoints/cfm_model_cfm_8bitnorm_2026-07-14_14-51-24.pt"
            if os.path.exists(default_fallback):
                checkpoint_to_load = default_fallback

    config = {
        "experiment_name": f"OCT_CFM_8BitNorm_RLFT_{args.loss_type.upper()}",
        "run_name": f"cfm_8bitnorm_rlft_{args.loss_type.lower()}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        "loss_type": args.loss_type.lower(),
        "baseline_checkpoint_used": checkpoint_to_load if checkpoint_to_load else "initialized_from_scratch",
        "mlflow_tracking_uri": "http://10.24.38.15:5000",
        "batch_size": 16,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "image_size": 256,
        "channels": 1,
        "val_check_interval": 5,
        "num_val_images": 3,
        "inference_steps": 50,
        "sigma": 0.0,
        "w_bg": 0.4,
        "w_layers": 1.0
    }

    # 1. Datasets & Loaders
    nr206_dataset = WeightedOCTDataset(
        labels_dir="DATA/NR206/train_labels",
        real_dir="DATA/NR206/train",
        sample_weight=1.0
    )
    train_loader = DataLoader(nr206_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2, pin_memory=True)

    val_dataset = WeightedOCTDataset(
        labels_dir="DATA/NR206/test_labels",
        real_dir="DATA/NR206/test",
        sample_weight=1.0
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    # 2. Model & Reward Function Initialization
    model = UNet2DModel(
        sample_size=config["image_size"],
        in_channels=config["channels"],
        out_channels=config["channels"],
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    ).to(device)

    # Load Base Model Checkpoint
    if checkpoint_to_load and os.path.exists(checkpoint_to_load):
        print(f"Loading base CFM checkpoint ({args.loss_type.upper()}) from {checkpoint_to_load}...")
        model.load_state_dict(torch.load(checkpoint_to_load, map_location=device))
    else:
        print(f"Notice: Target checkpoint {checkpoint_to_load} not found. Starting from baseline weights...")

    reward_model = OCTRewardModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    FM = ConditionalFlowMatcher(sigma=config["sigma"])

    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment(config["experiment_name"])

    print(f"Starting RLFT Fine-Tuning on {device}...")

    with mlflow.start_run(run_name=config["run_name"]):
        mlflow.log_params(config)
        mlflow.set_tag("training_mode", "RLFT (Reward-Weighted Policy Optimization)")

        for epoch in range(1, config["epochs"] + 1):
            model.train()
            total_loss = 0.0
            total_reward = 0.0

            pbar = tqdm(train_loader, desc=f"RLFT Epoch {epoch}/{config['epochs']}")
            for x0, x1, layer_mask, s_weight in pbar:
                x0, x1, layer_mask, s_weight = x0.to(device), x1.to(device), layer_mask.to(device), s_weight.to(device)
                optimizer.zero_grad()

                # Sample trajectory location
                t, x_t, u_t = FM.sample_location_and_conditional_flow(x0, x1)
                v_pred = model(x_t, t.squeeze()).sample

                # Compute Flow Trajectory Error
                spatial_weight = layer_mask * config["w_layers"] + (1.0 - layer_mask) * config["w_bg"]
                base_flow_loss = ((v_pred - u_t) ** 2 * spatial_weight).mean(dim=[1, 2, 3])

                # Compute Reward Weighting Score
                with torch.no_grad():
                    x_gen = generate_samples_ODE(model, x0, num_steps=20)
                    r_total, r_layer, r_contrast, r_speckle = reward_model(x_gen, x1, layer_mask)
                    
                    # Standardize rewards for stable softmax scaling (Advantage)
                    adv = (r_total - r_total.mean()) / (r_total.std() + 1e-4)
                    reward_weights = torch.softmax(adv / 0.1, dim=0) # temperature = 0.1

                # Reward-Weighted Policy Gradient Loss (incorporating sample_weight)
                rlft_loss = (base_flow_loss * reward_weights * s_weight.squeeze()).sum()

                rlft_loss.backward()
                optimizer.step()

                total_loss += rlft_loss.item()
                total_reward += r_total.mean().item()
                pbar.set_postfix({"loss": rlft_loss.item(), "reward": r_total.mean().item()})

            avg_loss = total_loss / len(train_loader)
            avg_reward = total_reward / len(train_loader)
            mlflow.log_metric("rlft_loss_epoch", avg_loss, step=epoch)
            mlflow.log_metric("reward_epoch", avg_reward, step=epoch)

            # Validation Loop
            model.eval()
            if epoch % config["val_check_interval"] == 0:
                logged_count = 0
                with torch.no_grad():
                    for batch_idx, (val_x0, val_x1, val_mask, val_sw) in enumerate(val_loader):
                        val_x0, val_x1, val_gt = val_x0.to(device), val_x1.to(device), val_x1.to(device)
                        if logged_count < config["num_val_images"]:
                            x1_gen = generate_samples_ODE(model, val_x0, num_steps=config["inference_steps"])
                            log_validation_grids(val_x0[0], x1_gen[0], val_gt[0], epoch, batch_idx)
                            logged_count += 1

                print(f"RLFT Epoch {epoch} | Policy Loss: {avg_loss:.4f} | Reward Score: {avg_reward:.4f}")

        # Save Final RLFT Model
        os.makedirs("checkpoints", exist_ok=True)
        checkpoint_path = f"checkpoints/cfm_model_{config['run_name']}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"RLFT Model saved to {checkpoint_path}")

        try:
            mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
            mlflow.pytorch.log_model(model, "model", registered_model_name=f"CFM_8BitNorm_RLFT_{config['loss_type'].upper()}_Model")
            print(f"Successfully registered model in MLflow Model Registry under CFM_8BitNorm_RLFT_{config['loss_type'].upper()}_Model")
        except Exception as e:
            print(f"Warning: MLflow model registration failed: {e}")

if __name__ == "__main__":
    main()
