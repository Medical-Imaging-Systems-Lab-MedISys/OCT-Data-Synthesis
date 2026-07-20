#!/usr/bin/env python3
import os
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import mlflow
import mlflow.pytorch

# Import torchcfm components
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

# Import diffusers UNet2DModel architecture
from diffusers import UNet2DModel

# Import synthetic generation script for online prior generation
from train_val import NR206DynamicDataset, synthesize_from_mask, log_validation_grids

# ==========================================
# 1. Configuration & Hyperparameters (L1 Spatial Loss)
# ==========================================
CONFIG = {
    "experiment_name": "OCT_CFM_8BitNorm_L1_Spatial",
    "run_name": f"cfm_8bitnorm_l1_spatial_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    "loss_type": "l1",
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
    "w_bg": 0.4,
    "w_layers": 1.0,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def train():
    mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
    mlflow.set_experiment(CONFIG["experiment_name"])

    # Image Transforms (8-bit norm: [0, 255] -> [-1, 1])
    transform = transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = NR206DynamicDataset(
        labels_dir="DATA/NR206/train_labels",
        real_dir="DATA/NR206/train"
    )
    val_dataset = NR206DynamicDataset(
        labels_dir="DATA/NR206/test_labels",
        real_dir="DATA/NR206/test"
    )

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    # Initialize 2D U-Net Model (in_channels=1, out_channels=1)
    model = UNet2DModel(
        sample_size=CONFIG["image_size"],
        in_channels=CONFIG["channels"],
        out_channels=CONFIG["channels"],
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    FM = ConditionalFlowMatcher(sigma=CONFIG["sigma"])

    print(f"Starting L1 Spatial CFM training on {device}...")

    with mlflow.start_run(run_name=CONFIG.get("run_name")):
        mlflow.log_params(CONFIG)
        mlflow.set_tag("normalization", "8-bit")
        mlflow.set_tag("loss_type", "L1 Spatial Weighted")
        mlflow.set_tag("loss_weighting", f"w_bg = {CONFIG['w_bg']}, w_layers = {CONFIG['w_layers']}")
        mlflow.set_tag("mlflow.note.content", f"CFM training run with L1 loss, 8-bit normalization, spatial loss weighting (w_bg = {CONFIG['w_bg']}, w_layers = {CONFIG['w_layers']}).")

        for epoch in range(1, CONFIG["epochs"] + 1):
            model.train()
            total_train_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']}")
            for x0, x1, layer_mask in pbar:
                x0, x1, layer_mask = x0.to(device), x1.to(device), layer_mask.to(device)
                optimizer.zero_grad()

                t, x_t, u_t = FM.sample_location_and_conditional_flow(x0, x1)
                v_pred = model(x_t, t.squeeze()).sample

                # L1 Loss with spatial weighting: |v_pred - u_t| * weight_map
                loss_map = torch.abs(v_pred - u_t)
                weight_map = layer_mask * CONFIG["w_layers"] + (1.0 - layer_mask) * CONFIG["w_bg"]
                loss = (loss_map * weight_map).mean()

                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})

            avg_train_loss = total_train_loss / len(train_loader)
            mlflow.log_metric("train_loss_epoch", avg_train_loss, step=epoch)

            # Validation Loop
            model.eval()
            total_val_loss = 0.0
            logged_images_count = 0

            with torch.no_grad():
                for batch_idx, (val_x0, val_x1, val_mask) in enumerate(val_loader):
                    val_x0, val_x1, val_mask = val_x0.to(device), val_x1.to(device), val_mask.to(device)
                    t, x_t, u_t = FM.sample_location_and_conditional_flow(val_x0, val_x1)
                    v_pred = model(x_t, t.squeeze()).sample

                    loss_map = torch.abs(v_pred - u_t)
                    weight_map = val_mask * CONFIG["w_layers"] + (1.0 - val_mask) * CONFIG["w_bg"]
                    val_loss = (loss_map * weight_map).mean()
                    total_val_loss += val_loss.item()

                    if epoch % CONFIG["val_check_interval"] == 0 and logged_images_count < CONFIG["num_val_images"]:
                        x1_gen = generate_samples(model, val_x0, num_steps=CONFIG["inference_steps"])
                        log_validation_grids(val_x0[0], x1_gen[0], val_x1[0], epoch, batch_idx)
                        logged_images_count += 1

                avg_val_loss = total_val_loss / len(val_loader)
                mlflow.log_metric("val_loss_epoch", avg_val_loss, step=epoch)
                print(f"Epoch {epoch} | Train L1 Loss: {avg_train_loss:.4f} | Val L1 Loss: {avg_val_loss:.4f}")

        os.makedirs("checkpoints", exist_ok=True)
        checkpoint_path = f"checkpoints/cfm_model_{CONFIG['run_name']}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
        try:
            mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
            mlflow.pytorch.log_model(model, "model", registered_model_name="CFM_8BitNorm_L1_Model")
            print("Successfully registered model in MLflow Model Registry under CFM_8BitNorm_L1_Model")
        except Exception as e:
            print(f"Warning: Failed to log artifact / register model to MLflow: {e}")

if __name__ == "__main__":
    train()
