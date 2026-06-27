#!/usr/bin/env python
# coding: utf-8

import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from datetime import datetime
import argparse

# =====================================================================
# 1. Image Synthesis Helper (from masks to synthetic speckled images)
# =====================================================================

def sample_gamma_from_bell_curve(min_g, max_g):
    """
    Samples a gamma value from a normal (bell-curve) distribution
    centered between min_g and max_g, and truncated to those bounds.
    """
    mean = (min_g + max_g) / 2.0
    std = (max_g - min_g) / 6.0
    return np.clip(np.random.normal(mean, std), min_g, max_g)

def apply_gamma(val, g):
    return 255.0 * np.power(val / 255.0, g)

def synthesize_from_mask(mask_bgra, min_gamma=0.5, max_gamma=1.2, custom_intensities=None):
    """
    Synthesizes a realistic-looking synthetic OCT image (with speckle noise)
    directly from a BGRA layer segmentation mask.
    """
    height, width, _ = mask_bgra.shape
    raw_img = np.zeros((height, width), dtype=np.float32)
    
    # Baseline layer parameters (fitted from NR206)
    LAYERS_CFG = [
        { 'name': 'Red',         'meanInt': 220.0, 'min_g': 0.85, 'max_g': 1.15, 'color': [0, 0, 255] },     # BGR Red
        { 'name': 'Olive',       'meanInt': 138.4, 'min_g': 0.90, 'max_g': 1.10, 'color': [0, 128, 128] },   # BGR Olive
        { 'name': 'Yellow',      'meanInt': 108.6, 'min_g': 0.90, 'max_g': 1.10, 'color': [0, 255, 255] },   # BGR Yellow
        { 'name': 'DarkGreen',   'meanInt': 133.8, 'min_g': 0.90, 'max_g': 1.10, 'color': [0, 128, 0] },     # BGR Dark Green
        { 'name': 'BrightGreen', 'meanInt': 75.0,  'min_g': 0.95, 'max_g': 1.05, 'color': [0, 255, 0] },     # BGR Bright Green
        { 'name': 'Cyan',        'meanInt': 210.0, 'min_g': 0.90, 'max_g': 1.10, 'color': [255, 255, 0] },   # BGR Cyan
        { 'name': 'Blue',        'meanInt': 137.5, 'min_g': 0.85, 'max_g': 1.15, 'color': [255, 0, 0] },     # BGR Blue
        { 'name': 'Magenta',     'meanInt': 210.0, 'min_g': 0.85, 'max_g': 1.15, 'color': [255, 0, 255] }    # BGR Magenta
    ]
    
    if custom_intensities is not None:
        for cfg in LAYERS_CFG:
            name = cfg['name']
            if name in custom_intensities:
                cfg['meanInt'] = custom_intensities[name]
    
    layer_gammas = [sample_gamma_from_bell_curve(cfg['min_g'], cfg['max_g']) for cfg in LAYERS_CFG]
    bg_gamma = sample_gamma_from_bell_curve(min_gamma, max_gamma)
    
    # Organic micro-texture along columns
    x_indices = np.arange(width)
    layer_texture = (np.sin(x_indices * 0.05) * 3 + np.cos(x_indices * 0.02) * 2)[None, :] # (1, width)
    layer_texture = np.broadcast_to(layer_texture, (height, width))
    
    # Detect background pixels
    is_bg = (mask_bgra[:, :, 0] == 0) & (mask_bgra[:, :, 1] == 0) & (mask_bgra[:, :, 2] == 0)
    
    # Identify retina pixels to find boundaries column-wise
    is_retina = ~is_bg
    has_retina = np.any(is_retina, axis=0)
    b8 = np.zeros(width, dtype=np.int32)
    if np.any(has_retina):
        b8 = height - 1 - np.argmax(is_retina[::-1, :], axis=0)
        b8[~has_retina] = height - 1
    else:
        b8 = np.full(width, height - 1, dtype=np.int32)
        
    y_coords = np.arange(height)[:, None] # (H, 1)
    
    # Render layers
    for i, cfg in enumerate(LAYERS_CFG):
        color = cfg['color']
        layer_mask = (mask_bgra[:, :, 0] == color[0]) & (mask_bgra[:, :, 1] == color[1]) & (mask_bgra[:, :, 2] == color[2])
        base_int = cfg['meanInt'] + layer_texture
        raw_img[layer_mask] = apply_gamma(base_int, layer_gammas[i])[layer_mask]
        
    # Sclera / deep background (decays quickly back to dark background)
    sclera_mask = is_bg & (y_coords >= b8[None, :])
    dist_from_b8 = y_coords - b8[None, :]
    sclera_intensity = 59.0 + 20.0 * np.exp(-dist_from_b8 / 25.0)
    raw_img[sclera_mask] = apply_gamma(sclera_intensity, bg_gamma)[sclera_mask]
    
    # Vitreous humor background (above retina)
    vitreous_mask = is_bg & (y_coords < b8[None, :])
    vitreous_intensity = np.full((height, width), 59.0, dtype=np.float32)
    raw_img[vitreous_mask] = apply_gamma(vitreous_intensity, bg_gamma)[vitreous_mask]
    
    # Apply Speckle Noise (Rayleigh/Gaussian simulation) and Clamping
    speckle = np.random.uniform(0.3, 1.2, size=(height, width))
    additive = np.random.uniform(-12.0, 12.0, size=(height, width))
    
    final_img = raw_img * speckle + additive
    final_img[is_bg] = np.clip(final_img[is_bg], 0, 90.0)
    
    final_img = np.clip(final_img, 0, 255).astype(np.uint8)
    
    return final_img

def prepare_synthetic_dataset(labels_path, synthetic_path, min_gamma=0.5, max_gamma=1.5):
    """
    Pre-generates paired synthetic speckled OCT images matching the real masks.
    """
    if os.path.exists(synthetic_path) and len(os.listdir(synthetic_path)) > 0:
        print(f"Synthetic dataset already cached at: {synthetic_path}")
        return

    os.makedirs(synthetic_path, exist_ok=True)
    print(f"Generating synthetic images from masks {labels_path} -> {synthetic_path}...")
    
    filenames = sorted([
        f for f in os.listdir(labels_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ])
    
    for idx, fname in enumerate(filenames):
        lbl_path = os.path.join(labels_path, fname)
        mask_bgra = cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED)
        
        # If read with only 3 channels, pad with alpha
        if mask_bgra is not None and len(mask_bgra.shape) == 3 and mask_bgra.shape[2] == 3:
            alpha = np.full((mask_bgra.shape[0], mask_bgra.shape[1], 1), 255, dtype=np.uint8)
            mask_bgra = np.concatenate([mask_bgra, alpha], axis=2)
            
        if mask_bgra is None:
            continue
            
        synth_img = synthesize_from_mask(mask_bgra, min_gamma, max_gamma)
        out_path = os.path.join(synthetic_path, fname)
        cv2.imwrite(out_path, synth_img)
        
        if (idx + 1) % 50 == 0 or (idx + 1) == len(filenames):
            print(f"Synthesized {idx + 1}/{len(filenames)} images...")

# =====================================================================
# 2. PyTorch Paired Dataset
# =====================================================================

def profile_single_image_intensities(real_img, mask_bgra, global_defaults):
    """
    Profiles the mean intensity of each layer present in the mask_bgra
    for a specific real_img. Returns a dict mapping layer name to mean intensity.
    """
    real_np = np.array(real_img)
    intensities = {}
    
    LAYERS_CFG = [
        { 'name': 'Red',         'color': [0, 0, 255] },
        { 'name': 'Olive',       'color': [0, 128, 128] },
        { 'name': 'Yellow',      'color': [0, 255, 255] },
        { 'name': 'DarkGreen',   'color': [0, 128, 0] },
        { 'name': 'BrightGreen', 'color': [0, 255, 0] },
        { 'name': 'Cyan',        'color': [255, 255, 0] },
        { 'name': 'Blue',        'color': [255, 0, 0] },
        { 'name': 'Magenta',     'color': [255, 0, 255] }
    ]
    
    for cfg in LAYERS_CFG:
        name = cfg['name']
        color = cfg['color']
        layer_mask = (mask_bgra[:, :, 0] == color[0]) & (mask_bgra[:, :, 1] == color[1]) & (mask_bgra[:, :, 2] == color[2])
        matching_pixels = real_np[layer_mask]
        if len(matching_pixels) > 0:
            intensities[name] = float(np.mean(matching_pixels))
        else:
            intensities[name] = global_defaults[name]
            
    return intensities

class PairedOCTDataset(Dataset):
    def __init__(self, labels_dir, real_dir, img_size, transform=None, normalization='minmax', is_train=True):
        self.labels_dir = labels_dir
        self.real_dir = real_dir
        self.img_size = img_size
        self.transform = transform
        self.normalization = normalization
        self.is_train = is_train
        
        self.filenames = sorted([
            f for f in os.listdir(real_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        print(f"Dataset pairing... Pre-loading {len(self.filenames)} files directly into RAM and profiling layer intensities...")
        
        # Initialize default intensities for fallback (adjusted baseline values)
        global_defaults = {
            'Red': 220.0,
            'Olive': 138.4,
            'Yellow': 108.6,
            'DarkGreen': 133.8,
            'BrightGreen': 75.0,
            'Cyan': 210.0,
            'Blue': 137.5,
            'Magenta': 210.0
        }
        
        # Scaling ratios based on NORMAL1.png tuning relative to real profiled NORMAL1.png values:
        # Real: Red=180.43, Olive=144.61, Yellow=120.62, DarkGreen=137.30, BrightGreen=98.55, Cyan=115.35, Blue=222.17, Magenta=206.72
        ratios = {
            'Red': 240.0 / 180.43,
            'Olive': 155.0 / 144.61,
            'Yellow': 122.0 / 120.62,
            'DarkGreen': 149.0 / 137.30,
            'BrightGreen': 86.0 / 98.55,
            'Cyan': 230.0 / 115.35,
            'Blue': 130.0 / 222.17,
            'Magenta': 230.0 / 206.72
        }
        
        self.real_images = []
        self.synth_images = []
        self.masks = []
        
        for fname in self.filenames:
            # 1. Load Real Image into RAM using cv2
            real_path = os.path.join(self.real_dir, fname)
            real_np = cv2.imread(real_path, cv2.IMREAD_GRAYSCALE)
            
            # Remove watermark dynamically in RAM
            clean_patch = real_np[350:, 600:]
            real_np[350:, :150] = np.flip(clean_patch, axis=1)
            
            # 2. Load Real Anatomical Mask into RAM
            lbl_path = os.path.join(self.labels_dir, fname)
            mask_bgra = cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED)
            if mask_bgra is not None and len(mask_bgra.shape) == 3 and mask_bgra.shape[2] == 3:
                alpha = np.full((mask_bgra.shape[0], mask_bgra.shape[1], 1), 255, dtype=np.uint8)
                mask_bgra = np.concatenate([mask_bgra, alpha], axis=2)
            self.masks.append(mask_bgra)
            
            # 4. Resize real image using cv2
            real_np_resized = cv2.resize(real_np, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            real_img_resized = Image.fromarray(real_np_resized, mode='L')
            
            # 5. Synthesize upfront to preload to VRAM without custom intensities
            synth_np = synthesize_from_mask(mask_bgra, min_gamma=0.5, max_gamma=1.5, custom_intensities=None)
            synth_np_resized = cv2.resize(synth_np, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            synth_img = Image.fromarray(synth_np_resized, mode='L')
            
            if self.transform:
                synth_img = self.transform(synth_img)
                real_img_resized = self.transform(real_img_resized)
                
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.real_images.append(real_img_resized.to(device))
            self.synth_images.append(synth_img.to(device))
            
        if self.normalization == 'zscore':
            print("Applying Z-Score Normalization...")
            real_stack = torch.stack(self.real_images)
            synth_stack = torch.stack(self.synth_images)
            self.real_mean = real_stack.mean()
            self.real_std = real_stack.std()
            self.synth_mean = synth_stack.mean()
            self.synth_std = synth_stack.std()
            
            self.real_images = [(img - self.real_mean) / (self.real_std + 1e-8) for img in self.real_images]
            self.synth_images = [(img - self.synth_mean) / (self.synth_std + 1e-8) for img in self.synth_images]
            
            self.mean = (self.real_mean, self.synth_mean)
            self.std = (self.real_std, self.synth_std)
            
        print(f"VRAM Pre-loading complete!")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if self.is_train:
            mask_bgra = self.masks[idx]
            synth_np = synthesize_from_mask(mask_bgra, min_gamma=0.5, max_gamma=1.5, custom_intensities=None)
            synth_np_resized = cv2.resize(synth_np, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            synth_img = Image.fromarray(synth_np_resized, mode='L')
            
            if self.transform:
                synth_img = self.transform(synth_img)
                
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            synth_img = synth_img.to(device)
            
            if self.normalization == 'zscore':
                synth_img = (synth_img - self.synth_mean) / (self.synth_std + 1e-8)
                
            return synth_img, self.real_images[idx]
        else:
            return self.synth_images[idx], self.real_images[idx]

# =====================================================================
# 3. Model Architectures (U-Net & PatchGAN)
# =====================================================================

class UNetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                up += [nn.Dropout(0.5)]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class UNetGenerator(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, img_size=128, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=True):
        """
        Constructs a U-Net Generator with skip-connections.
        Automatically scales based on the image size.
        """
        super().__init__()
        num_downs = int(np.log2(img_size))
        num_downs = max(5, num_downs) # Minimum depth layer guard
        
        # Build U-Net recursively from innermost to outermost
        unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=None, innermost=True, norm_layer=norm_layer)
        for _ in range(num_downs - 5):
            unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UNetSkipConnectionBlock(ngf * 4, ngf * 8, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UNetSkipConnectionBlock(ngf * 2, ngf * 4, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UNetSkipConnectionBlock(ngf, ngf * 2, submodule=unet_block, norm_layer=norm_layer)
        self.model = UNetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

    def forward(self, x):
        return self.model(x)


class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_nc=2, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """
        PatchGAN (70x70) Classifier with Spectral Normalization.
        Takes concatenated input/target images and classifies local patches.
        """
        super().__init__()
        import torch.nn.utils.spectral_norm as spectral_norm
        
        model = [
            spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            model += [
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=False)),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
            
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        model += [
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=False)),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        model += [spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1))]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# Weights initialization helper
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

# =====================================================================
# 4. Helper functions for training step
# =====================================================================

def tensor_to_numpy(grid_tensor):
    np_grid = grid_tensor.cpu().numpy()
    np_grid = np.transpose(np_grid, (1, 2, 0))
    # Unnormalize [-1, 1] -> [0, 255]
    np_grid = ((np_grid * 0.5 + 0.5) * 255).astype(np.uint8)
    return np_grid

def prob_tensor_to_numpy(grid_tensor):
    np_grid = grid_tensor.cpu().numpy()
    np_grid = np.transpose(np_grid, (1, 2, 0))
    # Scale [0, 1] -> [0, 255]
    np_grid = (np_grid * 255).astype(np.uint8)
    return np_grid

def verify_setup(config):
    """
    Verifies that the entire network forward/backward pipeline is working
    correctly using dummy tensors on the chosen device.
    """
    print("\n--- Starting Dry-Run Verification ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Initialize networks
    generator = UNetGenerator(input_nc=1, output_nc=1, img_size=config['img_size'], norm_layer=nn.InstanceNorm2d).to(device)
    discriminator = PatchGANDiscriminator(input_nc=2, n_layers=config.get('n_layers_D', 3), norm_layer=nn.InstanceNorm2d).to(device)
    generator.apply(init_weights)
    discriminator.apply(init_weights)
    
    # Create dummy tensors representing a batch of data
    batch_size = 2
    dummy_synthetic = torch.randn(batch_size, 1, config['img_size'], config['img_size']).to(device)
    dummy_real = torch.randn(batch_size, 1, config['img_size'], config['img_size']).to(device)
    
    print(f"Input synthetic shape: {dummy_synthetic.shape}")
    print(f"Input real shape:      {dummy_real.shape}")
    
    # Test generator forward pass
    fake_real = generator(dummy_synthetic)
    print(f"Generated fake shape:  {fake_real.shape}")
    assert fake_real.shape == dummy_real.shape, "Generator output shape mismatch!"
    
    # Test discriminator forward pass (paired)
    real_pair = torch.cat([dummy_synthetic, dummy_real], dim=1)
    fake_pair = torch.cat([dummy_synthetic, fake_real.detach()], dim=1)
    
    pred_real = discriminator(real_pair)
    pred_fake = discriminator(fake_pair)
    print(f"Discriminator pred real shape: {pred_real.shape}")
    print(f"Discriminator pred fake shape: {pred_fake.shape}")
    
    # Loss functions
    criterion_GAN = nn.BCEWithLogitsLoss()
    use_L2 = 'lambda_L2' in config
    criterion_Pixel = nn.MSELoss() if use_L2 else nn.L1Loss()
    pixel_lambda = config.get('lambda_L2', config.get('lambda_L1', 100.0))
    
    # Calculate losses & backward pass test
    loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
    loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
    loss_D = (loss_D_real + loss_D_fake) * 0.5
    
    loss_D.backward(retain_graph=True)
    print("Discriminator backward pass successful.")
    
    # Generator backward test
    pred_fake_g = discriminator(torch.cat([dummy_synthetic, fake_real], dim=1))
    loss_G_GAN = criterion_GAN(pred_fake_g, torch.ones_like(pred_fake_g))
    loss_G_Pixel = criterion_Pixel(fake_real, dummy_real) * pixel_lambda
    loss_G = loss_G_GAN + loss_G_Pixel
    
    loss_G.backward()
    print("Generator backward pass successful.")
    print("--- Dry-Run Verification Completed Successfully! ---\n")

# =====================================================================
# 5. Main Training Loop
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Pix2Pix Training script for OCT Image Translation")
    parser.add_argument('--config', type=str, default='models/pix2pix/config_pix2pix.json', help='Path to configuration file')
    parser.add_argument('--verify', action='store_true', help='Only verify models and forward/backward logic without training')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    if args.verify:
        verify_setup(config)
        return
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {device}")
    
    normalization_type = config.get('normalization', 'minmax')
    
    # Normalization and Dataset setup
    if normalization_type == 'zscore':
        print("Using Z-score normalization")
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
    
    # Initialize Datasets using Online Augmentation RAM Cache
    train_dataset = PairedOCTDataset(config['train_labels_path'], config['train_data_path'], config['img_size'], transform=transform, normalization=normalization_type, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=0)
    
    test_dataset = PairedOCTDataset(config['test_labels_path'], config['test_data_path'], config['img_size'], transform=transform, normalization=normalization_type, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # test_batch will now be dynamically sampled during the epoch loop
    
    # Network Initialization
    generator = UNetGenerator(input_nc=1, output_nc=1, img_size=config['img_size'], norm_layer=nn.InstanceNorm2d)
    discriminator = PatchGANDiscriminator(input_nc=2, n_layers=config.get('n_layers_D', 3), norm_layer=nn.InstanceNorm2d)
    
    # Weights initialization
    generator.apply(init_weights)
    discriminator.apply(init_weights)
    
    # Multi-GPU setups
    use_multi_gpu = config.get("use_multi_gpu", True)
    small_dataset_threshold = config.get("small_dataset_threshold", 50)
    dataset_size = len(train_dataset)
    
    if use_multi_gpu and dataset_size < small_dataset_threshold:
        print(f"Dataset size ({dataset_size}) is smaller than threshold ({small_dataset_threshold}). Using single device.")
        use_multi_gpu = False
        
    num_gpus = config.get("num_gpus", 4)
    available_gpus = torch.cuda.device_count()
    gpus_to_use = min(num_gpus, available_gpus)
    
    if use_multi_gpu and gpus_to_use > 1:
        print(f"Using {gpus_to_use} GPUs for training (out of {available_gpus} available)!")
        device_ids = list(range(gpus_to_use))
        generator = nn.DataParallel(generator, device_ids=device_ids)
        discriminator = nn.DataParallel(discriminator, device_ids=device_ids)
    else:
        print(f"Using a single GPU/device for training (requested {num_gpus}, available {available_gpus}).")
        
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    # Loss functions & Optimizers
    criterion_GAN = nn.BCEWithLogitsLoss()
    use_L2 = 'lambda_L2' in config
    criterion_Pixel = nn.MSELoss() if use_L2 else nn.L1Loss()
    pixel_lambda = config.get('lambda_L2', config.get('lambda_L1', 100.0))
    pixel_loss_name = "L2Loss (MSE)" if use_L2 else "L1Loss (MAE)"
    
    optimizer_G = optim.Adam(generator.parameters(), lr=config['learning_rate'], betas=(config['beta1'], 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config['learning_rate'], betas=(config['beta1'], 0.999))
    
    # MLflow Setup
    tracking_uri = config.get("mlflow_tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"Using remote MLflow tracking server: {tracking_uri}")

    mlflow.set_experiment(config['experiment_name'])
    
    # Set detailed markdown description for the experiment
    try:
        experiment = mlflow.get_experiment_by_name(config['experiment_name'])
        if experiment:
            client = mlflow.tracking.MlflowClient()
            experiment_description = (
                "# Pix2Pix Retinal OCT Image Translation Experiment\n\n"
                "This experiment trains a conditional GAN (Pix2Pix) mapping procedurally synthesized "
                "speckled OCT images to real-looking OCT scans.\n\n"
                "## Dataset Summary:\n"
                f"- **Training Set Size:** {len(train_dataset)} paired samples\n"
                f"- **Testing Set Size:** {len(test_dataset)} paired samples\n"
                f"- **Validation Visuals Batch Size:** 16 samples\n"
                f"- **Target Resolution:** {config['img_size']}x{config['img_size']} (Square)\n\n"
                "## Model Components:\n"
                "- **Generator:** U-Net architecture with skip connections (Instance Normalization).\n"
                "- **Discriminator:** PatchGAN classifying concatenated (synthetic, target) pairs (Instance Normalization + Spectral Normalization).\n"
                f"- **Loss Functions:** BCEWithLogitsLoss (Adversarial) + {pixel_loss_name} (Pixel-wise reconstruction).\n"
            )
            client.set_experiment_tag(experiment.experiment_id, "mlflow.note.content", experiment_description)
    except Exception as e:
        print(f"Warning: Could not set experiment description: {e}")

    run_name = f"Pix2Pix_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    
    with mlflow.start_run(run_name=run_name) as run:
        n_layers = config.get('n_layers_D', 3)
        penalty_type = "L2" if "lambda_L2" in config else "L1"
        penalty_value = config.get("lambda_L2", config.get("lambda_L1", "N/A"))
        
        disc_desc = f"PatchGAN ({n_layers}-Layers, InstanceNorm + SpectralNorm)"
        
        run_description = (
            f"**Run:** {run_name}\n\n"
            f"**Configuration Summary:**\n"
            f"- **Resolution:** {config['img_size']}x{config['img_size']}\n"
            f"- **Batch Size:** {config['batch_size']}\n"
            f"- **Epochs:** {config['epochs']}\n"
            f"- **{penalty_type} Lambda:** {penalty_value}\n"
            f"- **Generator:** U-Net (InstanceNorm2d)\n"
            f"- **Discriminator:** {disc_desc}"
        )
        mlflow.set_tag("mlflow.note.content", run_description)
        mlflow.set_tag("Resolution", f"{config['img_size']}x{config['img_size']}")
        mlflow.set_tag("Generator", "U-Net-InstanceNorm")
        mlflow.set_tag("Discriminator", disc_desc)
        mlflow.set_tag(f"{penalty_type}_Lambda", str(penalty_value))
        
        mlflow.log_params(config)
        mlflow.log_artifact(args.config)
        
        # Log to local markdown file
        os.makedirs("docs", exist_ok=True)
        local_log_path = os.path.join("docs", "experiments_log.md")
        with open(local_log_path, "a") as log_file:
            log_file.write(f"\n---\n\n## {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {run_name}\n")
            log_file.write(run_description + "\n")
        
        # (Initial static validation log removed; validation is now logged dynamically every 5 epochs)

        epochs = config['epochs']
        for epoch in range(epochs):
            print(f"Starting epoch {epoch + 1}/{epochs}...")
            
            g_losses = []
            g_losses_gan = []
            g_losses_l1 = []
            d_losses = []
            
            generator.train()
            discriminator.train()
            
            for idx, (synth_imgs, real_imgs) in enumerate(train_loader):
                synth_imgs = synth_imgs.to(device)
                real_imgs = real_imgs.to(device)
                
                # ------------------
                # Train Discriminator
                # ------------------
                optimizer_D.zero_grad()
                
                # Real pair loss
                real_pair = torch.cat([synth_imgs, real_imgs], dim=1)
                pred_real = discriminator(real_pair)
                loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
                
                # Fake pair loss
                fake_imgs = generator(synth_imgs)
                fake_pair = torch.cat([synth_imgs, fake_imgs.detach()], dim=1)
                pred_fake = discriminator(fake_pair)
                loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
                
                loss_D = (loss_D_real + loss_D_fake) * 0.5
                loss_D.backward()
                optimizer_D.step()
                
                # ------------------
                # Train Generator
                # ------------------
                optimizer_G.zero_grad()
                
                # Generator wants the discriminator to believe the fake is real
                fake_pair_g = torch.cat([synth_imgs, fake_imgs], dim=1)
                pred_fake_g = discriminator(fake_pair_g)
                loss_G_GAN = criterion_GAN(pred_fake_g, torch.ones_like(pred_fake_g))
                
                # Pixel-wise loss
                loss_G_Pixel = criterion_Pixel(fake_imgs, real_imgs) * pixel_lambda
                
                # Total Generator loss
                loss_G = loss_G_GAN + loss_G_Pixel
                loss_G.backward()
                optimizer_G.step()
                
                g_losses.append(loss_G.item())
                g_losses_gan.append(loss_G_GAN.item())
                g_losses_l1.append(loss_G_Pixel.item())
                d_losses.append(loss_D.item())
                
            mean_g_loss = np.mean(g_losses)
            mean_g_loss_gan = np.mean(g_losses_gan)
            mean_g_loss_l1 = np.mean(g_losses_l1)
            mean_d_loss = np.mean(d_losses)
            
            print(f"Epoch {epoch+1} - Train G loss: {mean_g_loss:.4f} (GAN: {mean_g_loss_gan:.4f}, L1: {mean_g_loss_l1:.4f}), Train D loss: {mean_d_loss:.4f}")
            mlflow.log_metric("g_loss", mean_g_loss, step=epoch)
            mlflow.log_metric("g_loss_gan", mean_g_loss_gan, step=epoch)
            mlflow.log_metric("g_loss_l1", mean_g_loss_l1, step=epoch)
            mlflow.log_metric("d_loss", mean_d_loss, step=epoch)
            
            # Validation loop
            generator.eval()
            discriminator.eval()
            val_g_losses = []
            val_g_losses_gan = []
            val_g_losses_l1 = []
            val_d_losses = []
            
            with torch.no_grad():
                for val_batch in test_loader:
                    val_synth_imgs, val_real_imgs = val_batch
                    val_synth_imgs = val_synth_imgs.to(device)
                    val_real_imgs = val_real_imgs.to(device)
                    
                    # Generate fake images
                    val_fakes = generator(val_synth_imgs)
                    
                    # D validation loss
                    val_real_pair = torch.cat([val_synth_imgs, val_real_imgs], dim=1)
                    pred_real = discriminator(val_real_pair)
                    loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
                    
                    val_fake_pair = torch.cat([val_synth_imgs, val_fakes.detach()], dim=1)
                    pred_fake = discriminator(val_fake_pair)
                    loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
                    val_loss_D = (loss_D_real + loss_D_fake) * 0.5
                    
                    # G validation loss
                    val_fake_pair_g = torch.cat([val_synth_imgs, val_fakes], dim=1)
                    pred_fake_g = discriminator(val_fake_pair_g)
                    val_loss_G_GAN = criterion_GAN(pred_fake_g, torch.ones_like(pred_fake_g))
                    val_loss_G_Pixel = criterion_Pixel(val_fakes, val_real_imgs) * config.get('lambda_L2', config.get('lambda_L1', 100))
                    val_loss_G = val_loss_G_GAN + val_loss_G_Pixel
                    
                    val_g_losses.append(val_loss_G.item())
                    val_g_losses_gan.append(val_loss_G_GAN.item())
                    val_g_losses_l1.append(val_loss_G_Pixel.item())
                    val_d_losses.append(val_loss_D.item())
            
            mean_val_g_loss = np.mean(val_g_losses)
            mean_val_g_loss_gan = np.mean(val_g_losses_gan)
            mean_val_g_loss_l1 = np.mean(val_g_losses_l1)
            mean_val_d_loss = np.mean(val_d_losses)
            
            print(f"Epoch {epoch+1} - Val G loss: {mean_val_g_loss:.4f} (GAN: {mean_val_g_loss_gan:.4f}, L1: {mean_val_g_loss_l1:.4f}), Val D loss: {mean_val_d_loss:.4f}")
            mlflow.log_metric("val_g_loss", mean_val_g_loss, step=epoch)
            mlflow.log_metric("val_g_loss_gan", mean_val_g_loss_gan, step=epoch)
            mlflow.log_metric("val_g_loss_l1", mean_val_g_loss_l1, step=epoch)
            mlflow.log_metric("val_d_loss", mean_val_d_loss, step=epoch)
            
            generator.train()
            discriminator.train()
            
            # Periodically save test visuals to MLflow (every 5 epochs)
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                # Dynamically sample and augment validation priors on-the-fly!
                test_batch = next(iter(test_loader))
                test_synth_imgs, test_real_imgs = test_batch
                test_synth_imgs = test_synth_imgs.to(device)
                test_real_imgs = test_real_imgs.to(device)
                
                generator.eval()
                with torch.no_grad():
                    test_fakes = generator(test_synth_imgs)
                
                # Log exactly 3 comparison image grid files (each with prior, synthetic fake, real ground truth side by side)
                for i in range(min(3, len(test_synth_imgs))):
                    # Move to CPU and handle normalization differences to get approx [0, 1] range
                    if normalization_type == 'zscore':
                        mean_s, std_s = test_dataset.synth_mean.item(), test_dataset.synth_std.item()
                        mean_r, std_r = test_dataset.real_mean.item(), test_dataset.real_std.item()
                        
                        x0_disp = test_synth_imgs[i].squeeze().cpu().numpy() * std_s + mean_s
                        x1_gen_disp = test_fakes[i].squeeze().cpu().numpy() * std_r + mean_r
                        x1_gt_disp = test_real_imgs[i].squeeze().cpu().numpy() * std_r + mean_r
                    else:
                        # Assuming [-1, 1] minmax from composed transform
                        x0_disp = (test_synth_imgs[i].squeeze().cpu().numpy() + 1.0) / 2.0
                        x1_gen_disp = (test_fakes[i].squeeze().cpu().numpy() + 1.0) / 2.0
                        x1_gt_disp = (test_real_imgs[i].squeeze().cpu().numpy() + 1.0) / 2.0

                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    titles = ["Prior (Synthetic Mask)", "Generated Synthesis", "Ground Truth (Real)"]
                    images = [x0_disp, x1_gen_disp, x1_gt_disp]

                    for ax, title, img in zip(axes, titles, images):
                        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
                        ax.imshow(img, cmap='gray')
                        ax.axis('off')

                    plt.tight_layout()
                    mlflow.log_figure(fig, f"validation_grids/epoch_{epoch+1}_sample_{i+1}.png")
                    plt.close(fig)
                    
        # Log models at the end of the run
        gen_to_log = generator.module if isinstance(generator, nn.DataParallel) else generator
        disc_to_log = discriminator.module if isinstance(discriminator, nn.DataParallel) else discriminator
        mlflow.pytorch.log_model(gen_to_log, "generator_model", serialization_format="pickle")
        mlflow.pytorch.log_model(disc_to_log, "discriminator_model", serialization_format="pickle")
        print("Training completed and logged successfully.")

if __name__ == '__main__':
    main()
