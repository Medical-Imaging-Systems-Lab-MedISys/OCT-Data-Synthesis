#!/usr/bin/env python
# coding: utf-8
import os
import cv2
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from datetime import datetime
from tqdm import tqdm

# ==========================================
# Configuration & Hyperparameters
# ==========================================
CONFIG = {
    "experiment_name": "Exp13_cGAN_Cropped",
    "run_name": "",
    "mlflow_tracking_uri": "https://dagshub.com/IISc-MedISys/OCT-Data-Synthesis.mlflow",
    "batch_size": 16,
    "epochs": 100,
    "lr_G": 0.0002,
    "lr_D": 0.0002,
    "beta1": 0.5,
    "lambda_L1": 100.0,
    "image_size": 256,
    "noise_dim": 1, # Number of spatial noise channels
    "val_check_interval": 5,
    "num_val_images": 3
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# 1. Image Synthesis Helpers (Dynamic Prior Generation)
# =====================================================================
def sample_gamma_from_bell_curve(min_g, max_g):
    mean = (min_g + max_g) / 2.0
    std = (max_g - min_g) / 6.0
    return np.clip(np.random.normal(mean, std), min_g, max_g)

def apply_gamma(val, g):
    return 255.0 * np.power(val / 255.0, g)

def synthesize_from_mask(mask_bgra, min_gamma=0.5, max_gamma=1.2, custom_intensities=None):
    height, width, _ = mask_bgra.shape
    raw_img = np.zeros((height, width), dtype=np.float32)
    
    LAYERS_CFG = [
        { 'name': 'Red',         'meanInt': 220.0, 'min_g': 0.85, 'max_g': 1.15, 'color': [0, 0, 255] },
        { 'name': 'Olive',       'meanInt': 138.4, 'min_g': 0.90, 'max_g': 1.10, 'color': [0, 128, 128] },
        { 'name': 'Yellow',      'meanInt': 108.6, 'min_g': 0.90, 'max_g': 1.10, 'color': [0, 255, 255] },
        { 'name': 'DarkGreen',   'meanInt': 133.8, 'min_g': 0.90, 'max_g': 1.10, 'color': [0, 128, 0] },
        { 'name': 'BrightGreen', 'meanInt': 75.0,  'min_g': 0.95, 'max_g': 1.05, 'color': [0, 255, 0] },
        { 'name': 'Cyan',        'meanInt': 210.0, 'min_g': 0.90, 'max_g': 1.10, 'color': [255, 255, 0] },
        { 'name': 'Blue',        'meanInt': 137.5, 'min_g': 0.85, 'max_g': 1.15, 'color': [255, 0, 0] },
        { 'name': 'Magenta',     'meanInt': 210.0, 'min_g': 0.85, 'max_g': 1.15, 'color': [255, 0, 255] }
    ]
    
    layer_gammas = [sample_gamma_from_bell_curve(cfg['min_g'], cfg['max_g']) for cfg in LAYERS_CFG]
    bg_gamma = sample_gamma_from_bell_curve(min_gamma, max_gamma)
    
    x_indices = np.arange(width)
    layer_texture = (np.sin(x_indices * 0.05) * 3 + np.cos(x_indices * 0.02) * 2)[None, :]
    layer_texture = np.broadcast_to(layer_texture, (height, width))
    
    is_bg = (mask_bgra[:, :, 0] == 0) & (mask_bgra[:, :, 1] == 0) & (mask_bgra[:, :, 2] == 0)
    is_retina = ~is_bg
    has_retina = np.any(is_retina, axis=0)
    b8 = np.zeros(width, dtype=np.int32)
    if np.any(has_retina):
        b8 = height - 1 - np.argmax(is_retina[::-1, :], axis=0)
        b8[~has_retina] = height - 1
    else:
        b8 = np.full(width, height - 1, dtype=np.int32)

    y_coords = np.arange(height)[:, None]
    
    for i, cfg in enumerate(LAYERS_CFG):
        color = cfg['color']
        layer_mask = (mask_bgra[:, :, 0] == color[0]) & (mask_bgra[:, :, 1] == color[1]) & (mask_bgra[:, :, 2] == color[2])
        base_int = cfg['meanInt'] + layer_texture
        raw_img[layer_mask] = apply_gamma(base_int, layer_gammas[i])[layer_mask]

    sclera_mask = is_bg & (y_coords >= b8[None, :])
    dist_from_b8 = y_coords - b8[None, :]
    sclera_intensity = 59.0 + 20.0 * np.exp(-dist_from_b8 / 25.0)
    raw_img[sclera_mask] = apply_gamma(sclera_intensity, bg_gamma)[sclera_mask]
    
    vitreous_mask = is_bg & (y_coords < b8[None, :])
    vitreous_intensity = np.full((height, width), 59.0, dtype=np.float32)
    raw_img[vitreous_mask] = apply_gamma(vitreous_intensity, bg_gamma)[vitreous_mask]
    
    speckle = np.random.uniform(0.3, 1.2, size=(height, width))
    additive = np.random.uniform(-12.0, 12.0, size=(height, width))
    
    final_img = raw_img * speckle + additive
    final_img[is_bg] = np.clip(final_img[is_bg], 0, 90.0)
    return np.clip(final_img, 0, 255).astype(np.uint8)

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
        tiled_bg = np.pad(tiled_bg, ((0, 0), (0, pad_w)), mode='symmetric')
            
    y_coords = np.arange(max_dim)[:, None]
    keep_mask = y_coords <= b8[None, :]
    if pad_w > 0:
        keep_mask = np.pad(keep_mask, ((0, 0), (0, pad_w)), mode='constant', constant_values=False)
        
    cropped_img = image[:cropped_h]
    padded_img = np.pad(cropped_img, ((0, pad_h), (0, pad_w)), mode='constant')
    return np.where(keep_mask, padded_img, tiled_bg)

# =====================================================================
# 2. Dataset
# =====================================================================
class NR206DynamicDataset(Dataset):
    def __init__(self, labels_dir, real_dir, img_size=256):
        self.labels_dir = labels_dir
        self.real_dir = real_dir
        self.img_size = img_size
        self.filenames = sorted([f for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg'))])

    def __len__(self): return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        lbl_path = os.path.join(self.labels_dir, fname)
        real_path = os.path.join(self.real_dir, fname)
        
        x1_img = cv2.imread(real_path, cv2.IMREAD_GRAYSCALE)
        clean_patch = x1_img[350:, 600:]
        x1_img[350:, :150] = np.flip(clean_patch, axis=1)
            
        mask_bgra = cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED)
        if len(mask_bgra.shape) == 3 and mask_bgra.shape[2] == 3:
            alpha = np.full((mask_bgra.shape[0], mask_bgra.shape[1], 1), 255, dtype=np.uint8)
            mask_bgra = np.concatenate([mask_bgra, alpha], axis=2)
            
        x0_img = synthesize_from_mask(mask_bgra)
        
        target_size = (self.img_size, self.img_size)
        x0_img_squashed = cv2.resize(x0_img, target_size, interpolation=cv2.INTER_LINEAR)
        x1_img_squashed = cv2.resize(x1_img, target_size, interpolation=cv2.INTER_LINEAR)
        mask_squashed = cv2.resize(mask_bgra, target_size, interpolation=cv2.INTER_LINEAR)
        
        x0_img = crop_and_pad_curved(x0_img_squashed, mask_squashed)
        x1_img = crop_and_pad_curved(x1_img_squashed, mask_squashed)
        
        x0 = (x0_img.astype(np.float32) / 127.5) - 1.0
        x1 = (x1_img.astype(np.float32) / 127.5) - 1.0
        
        return torch.from_numpy(x0).unsqueeze(0), torch.from_numpy(x1).unsqueeze(0)

# =====================================================================
# 3. Networks (cGAN Architecture)
# =====================================================================
class ConditionalGenerator(nn.Module):
    def __init__(self, condition_nc=1, noise_nc=1, out_nc=1, ngf=64):
        super().__init__()
        # Input: (Condition + Spatially expanded noise)
        in_nc = condition_nc + noise_nc
        
        self.main = nn.Sequential(
            nn.Conv2d(in_nc, ngf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf, out_nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, condition, noise):
        # condition: [B, 1, H, W], noise: [B, 1, H, W]
        x = torch.cat([condition, noise], dim=1)
        return self.main(x)

class ConditionalDiscriminator(nn.Module):
    def __init__(self, in_nc=2, ndf=64): # in_nc = condition + target
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 1, bias=False)
        )

    def forward(self, condition, target):
        x = torch.cat([condition, target], dim=1)
        return self.main(x)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# ==========================================
# 4. Main Training Loop
# ==========================================
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss_type', type=str, default='l1', choices=['l1', 'l2'], help="Choose 'l1' or 'l2' for pixel loss")
    args = parser.parse_args()
    
    # Update config for MLflow tracking
    CONFIG["loss_type"] = args.loss_type.upper()
    CONFIG["run_name"] = f"cGAN_Cropped_{args.loss_type.upper()}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
    mlflow.set_experiment(CONFIG["experiment_name"])
    
    local_data_dir = os.environ.get("LOCAL_DATA_DIR", "NR206")
    train_loader = DataLoader(
        NR206DynamicDataset(os.path.join(local_data_dir, "train_labels"), os.path.join(local_data_dir, "train"), CONFIG["image_size"]),
        batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        NR206DynamicDataset(os.path.join(local_data_dir, "test_labels"), os.path.join(local_data_dir, "test"), CONFIG["image_size"]),
        batch_size=1, shuffle=False, num_workers=2
    )

    netG = ConditionalGenerator(noise_nc=CONFIG["noise_dim"]).to(device)
    netD = ConditionalDiscriminator().to(device)
    netG.apply(init_weights)
    netD.apply(init_weights)

    criterion_GAN = nn.BCEWithLogitsLoss()

    if args.loss_type == 'l2':
        criterion_Pixel = nn.MSELoss()
    else:
        criterion_Pixel = nn.L1Loss()
    
    optimizerD = optim.Adam(netD.parameters(), lr=CONFIG["lr_D"], betas=(CONFIG["beta1"], 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=CONFIG["lr_G"], betas=(CONFIG["beta1"], 0.999))

    with mlflow.start_run(run_name=CONFIG["run_name"]):
        mlflow.log_params(CONFIG)
        mlflow.set_tag("Model", "cGAN_Cropped_Linear_Latent")
        
        global_step = 0
        for epoch in range(1, CONFIG["epochs"] + 1):
            netG.train()
            netD.train()
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']}")
            for x0_prior, x1_real in pbar:
                x0_prior, x1_real = x0_prior.to(device), x1_real.to(device)
                b_size = x0_prior.size(0)
                
                # Spatial noise tensor
                noise = torch.randn(b_size, CONFIG["noise_dim"], CONFIG["image_size"], CONFIG["image_size"], device=device)
                
                # --- Train Discriminator ---
                netD.zero_grad()
                pred_real = netD(x0_prior, x1_real)
                loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
                
                x1_fake = netG(x0_prior, noise)
                pred_fake = netD(x0_prior, x1_fake.detach())
                loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
                
                loss_D = (loss_D_real + loss_D_fake) * 0.5
                loss_D.backward()
                optimizerD.step()

                # --- Train Generator ---
                netG.zero_grad()
                pred_fake_g = netD(x0_prior, x1_fake)
                loss_G_GAN = criterion_GAN(pred_fake_g, torch.ones_like(pred_fake_g))
                loss_G_Pixel = criterion_Pixel(x1_fake, x1_real) * CONFIG["lambda_L1"] # Using same lambda weight
                loss_G = loss_G_GAN + loss_G_Pixel

                loss_G.backward()
                optimizerG.step()
                
                global_step += 1
                mlflow.log_metrics({"D_loss": loss_D.item(), "G_loss": loss_G.item()}, step=global_step)
                pbar.set_postfix({"D": f"{loss_D.item():.4f}", "G": f"{loss_G.item():.4f}"})

            # --- Validation & Plotting ---
            if epoch % CONFIG["val_check_interval"] == 0 or epoch == 1:
                netG.eval()
                logged_images = 0
                with torch.no_grad():
                    for batch_idx, (val_x0, val_x1) in enumerate(val_loader):
                        if logged_images >= CONFIG["num_val_images"]: break
                        
                        val_x0, val_x1 = val_x0.to(device), val_x1.to(device)
                        val_noise = torch.randn(1, CONFIG["noise_dim"], CONFIG["image_size"], CONFIG["image_size"], device=device)
                        val_fake = netG(val_x0, val_noise)
                        
                        # Un-normalize to [0, 1]
                        x0_disp = (val_x0[0].squeeze().cpu().numpy() + 1.0) / 2.0
                        x1_gen_disp = (val_fake[0].squeeze().cpu().numpy() + 1.0) / 2.0
                        x1_gt_disp = (val_x1[0].squeeze().cpu().numpy() + 1.0) / 2.0

                        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                        titles = ["Prior (Synthetic)", "Generated (cGAN)", "Ground Truth"]
                        for ax, title, img in zip(axes, titles, [x0_disp, x1_gen_disp, x1_gt_disp]):
                            ax.set_title(title, fontweight='bold', pad=10)
                            ax.imshow(img, cmap='gray')
                            ax.axis('off')
                        
                        plt.tight_layout()
                        mlflow.log_figure(fig, f"validation_grids/epoch_{epoch}_sample_{batch_idx}.png")
                        plt.close(fig)
                        logged_images += 1

        torch.save(netG.state_dict(), "cgan_generator.pth")
        mlflow.log_artifact("cgan_generator.pth")
        os.remove("cgan_generator.pth")

if __name__ == "__main__":
    train()
