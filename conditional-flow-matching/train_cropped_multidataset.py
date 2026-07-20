#!/usr/bin/env python3
import os
import sys
import argparse
import datetime
import random
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import mlflow
import mlflow.pytorch

# Seed everything
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from diffusers import UNet2DModel

# Import synthesize_from_mask & crop_and_pad_curved
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from train_val import synthesize_from_mask, log_validation_grids

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

class WeightedOCTDataset(Dataset):
    def __init__(self, labels_dir, real_dir, sample_weight=1.0, min_gamma=0.5, max_gamma=1.5):
        self.labels_dir = labels_dir
        self.real_dir = real_dir
        self.sample_weight = sample_weight
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        
        self.filenames = sorted([
            f for f in os.listdir(real_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
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
            
        # Dynamically remove watermark from bottom-left before curved cropping and background tiling
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
        
        # Binary layer mask for spatial loss weighting
        is_bg = (mask_squashed[:, :, 0] == 0) & (mask_squashed[:, :, 1] == 0) & (mask_squashed[:, :, 2] == 0)
        layer_mask = (~is_bg).astype(np.float32)
        
        # Normalize to [-1.0, 1.0]
        x0 = (x0_cropped.astype(np.float32) / 127.5) - 1.0
        x1 = (x1_cropped.astype(np.float32) / 127.5) - 1.0
        
        x0_tensor = torch.from_numpy(x0).unsqueeze(0)
        x1_tensor = torch.from_numpy(x1).unsqueeze(0)
        mask_tensor = torch.from_numpy(layer_mask).unsqueeze(0)
        weight_tensor = torch.tensor([self.sample_weight], dtype=torch.float32)
        
        return x0_tensor, x1_tensor, mask_tensor, weight_tensor

@torch.no_grad()
def generate_samples(model, x0, num_steps=50):
    model.eval()
    x_t = x0.clone()
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t_val = i / num_steps
        t_batch = torch.full((x_t.shape[0],), t_val, device=x0.device)
        v_pred = model(x_t, t_batch).sample
        x_t = x_t + v_pred * dt
    return x_t

def main():
    parser = argparse.ArgumentParser(description="Train Cropped CFM with Multi-Dataset Pseudo Weighting")
    parser.add_argument("--loss_type", type=str, default="l2", choices=["l1", "l2"], help="Loss type: l1 or l2")
    parser.add_argument("--pseudo_weight", type=float, default=0.5, help="Loss weight for pseudo-labeled dataset samples")
    parser.add_argument("--no_spatial_weighting", action="store_true", help="Disable spatial loss weighting (w_bg=1.0, w_layers=1.0)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "experiment_name": "OCT_CFM_8BitNorm_Cropped_MultiDataset",
        "run_name": f"cfm_8bitnorm_cropped_{args.loss_type.lower()}_weight{args.pseudo_weight}{'_nospatial' if args.no_spatial_weighting else ''}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        "loss_type": args.loss_type.lower(),
        "pseudo_weight": args.pseudo_weight,
        "no_spatial_weighting": args.no_spatial_weighting,
        "mlflow_tracking_uri": "http://10.24.38.15:5000",
        "batch_size": 16,
        "epochs": 300,
        "learning_rate": 0.0002,
        "image_size": 256,
        "channels": 1,
        "val_check_interval": 5,
        "num_val_images": 3,
        "inference_steps": 50,
        "sigma": 0.0,
        "w_bg": 1.0 if args.no_spatial_weighting else 0.4,
        "w_layers": 1.0
    }

    # 1. Load Datasets
    # NR206 Manual Ground Truth Dataset -> Weight 1.0
    nr206_dataset = WeightedOCTDataset(
        labels_dir="DATA/NR206/train_labels",
        real_dir="DATA/NR206/train",
        sample_weight=1.0
    )

    datasets = [nr206_dataset]

    # OCT5k Pseudo-Labeled Dataset (if available) -> Weight args.pseudo_weight
    oct5k_labels = "DATA/pseudo_labels/predictions_oct5k"
    oct5k_real = "DATA/OCT5k/Images"
    if os.path.exists(oct5k_labels) and os.path.exists(oct5k_real):
        oct5k_dataset = WeightedOCTDataset(
            labels_dir=oct5k_labels,
            real_dir=oct5k_real,
            sample_weight=args.pseudo_weight
        )
        datasets.append(oct5k_dataset)
        print(f"Added OCT5k pseudo-labeled dataset ({len(oct5k_dataset)} samples) with weight {args.pseudo_weight}")

    combined_dataset = ConcatDataset(datasets)
    train_loader = DataLoader(combined_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2, pin_memory=True)

    val_dataset = WeightedOCTDataset(
        labels_dir="DATA/NR206/test_labels",
        real_dir="DATA/NR206/test",
        sample_weight=1.0
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    # 2. Model & Optimizer
    model = UNet2DModel(
        sample_size=config["image_size"],
        in_channels=config["channels"],
        out_channels=config["channels"],
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    FM = ConditionalFlowMatcher(sigma=config["sigma"])

    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment(config["experiment_name"])

    print(f"Starting Cropped CFM ({args.loss_type.upper()}) Training on {device} across {len(combined_dataset)} total samples...")

    with mlflow.start_run(run_name=config["run_name"]):
        mlflow.log_params(config)
        mlflow.set_tag("cropping_strategy", "Drop & Replace Curved Cropping")
        mlflow.set_tag("pseudo_weight", str(args.pseudo_weight))
        mlflow.set_tag("loss_type", args.loss_type.upper())

        for epoch in range(1, config["epochs"] + 1):
            model.train()
            total_train_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']}")
            for x0, x1, layer_mask, s_weight in pbar:
                x0, x1, layer_mask, s_weight = x0.to(device), x1.to(device), layer_mask.to(device), s_weight.to(device)
                optimizer.zero_grad()

                t, x_t, u_t = FM.sample_location_and_conditional_flow(x0, x1)
                v_pred = model(x_t, t.squeeze()).sample

                # Spatial Loss Map (L1 or L2)
                if config["loss_type"] == "l1":
                    loss_map = torch.abs(v_pred - u_t)
                else:
                    loss_map = (v_pred - u_t) ** 2

                spatial_weight = layer_mask * config["w_layers"] + (1.0 - layer_mask) * config["w_bg"]
                # Apply sample_weight (1.0 for NR206, 0.5 for pseudo-labeled samples)
                sample_weight_4d = s_weight.view(-1, 1, 1, 1)
                loss = (loss_map * spatial_weight * sample_weight_4d).mean()

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
                for batch_idx, (val_x0, val_x1, val_mask, val_sw) in enumerate(val_loader):
                    val_x0, val_x1, val_mask, val_sw = val_x0.to(device), val_x1.to(device), val_mask.to(device), val_sw.to(device)
                    t, x_t, u_t = FM.sample_location_and_conditional_flow(val_x0, val_x1)
                    v_pred = model(x_t, t.squeeze()).sample

                    if config["loss_type"] == "l1":
                        loss_map = torch.abs(v_pred - u_t)
                    else:
                        loss_map = (v_pred - u_t) ** 2

                    spatial_weight = val_mask * config["w_layers"] + (1.0 - val_mask) * config["w_bg"]
                    val_loss = (loss_map * spatial_weight * val_sw.view(-1, 1, 1, 1)).mean()
                    total_val_loss += val_loss.item()

                    if epoch % config["val_check_interval"] == 0 and logged_count < config["num_val_images"]:
                        x1_gen = generate_samples(model, val_x0, num_steps=config["inference_steps"])
                        log_validation_grids(val_x0[0], x1_gen[0], val_x1[0], epoch, batch_idx)
                        logged_count += 1

                avg_val_loss = total_val_loss / len(val_loader)
                mlflow.log_metric("val_loss_epoch", avg_val_loss, step=epoch)
                print(f"Epoch {epoch} | Train {args.loss_type.upper()} Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        os.makedirs("checkpoints", exist_ok=True)
        checkpoint_path = f"checkpoints/cfm_model_{config['run_name']}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
        try:
            mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
            mlflow.pytorch.log_model(model, "model", registered_model_name=f"CFM_8BitNorm_Cropped_{args.loss_type.upper()}_Model")
            print(f"Successfully registered model in MLflow Model Registry under CFM_8BitNorm_Cropped_{args.loss_type.upper()}_Model")
        except Exception as e:
            print(f"Warning: Failed to log/register model to MLflow: {e}")

if __name__ == "__main__":
    main()
