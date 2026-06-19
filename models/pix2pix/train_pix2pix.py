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

def synthesize_from_mask(mask_bgra, min_gamma=0.5, max_gamma=1.5):
    """
    Synthesizes a realistic-looking synthetic OCT image (with speckle noise)
    directly from a BGRA layer segmentation mask.
    """
    height, width, _ = mask_bgra.shape
    raw_img = np.zeros((height, width), dtype=np.float32)
    
    # Baseline layer parameters (fitted from NR206)
    LAYERS_CFG = [
        { 'name': 'Red',         'meanInt': 200.0, 'color': [0, 0, 255] },     # BGR Red
        { 'name': 'Olive',       'meanInt': 130.0, 'color': [0, 128, 128] },   # BGR Olive
        { 'name': 'Yellow',      'meanInt': 180.0, 'color': [0, 255, 255] },   # BGR Yellow
        { 'name': 'DarkGreen',   'meanInt': 110.0, 'color': [0, 128, 0] },     # BGR Dark Green
        { 'name': 'BrightGreen', 'meanInt': 60.0,  'color': [0, 255, 0] },     # BGR Bright Green
        { 'name': 'Cyan',        'meanInt': 140.0, 'color': [255, 255, 0] },   # BGR Cyan
        { 'name': 'Blue',        'meanInt': 245.0, 'color': [255, 0, 0] },     # BGR Blue
        { 'name': 'Magenta',     'meanInt': 80.0,  'color': [255, 0, 255] }    # BGR Magenta
    ]
    
    layer_gammas = [sample_gamma_from_bell_curve(min_gamma, max_gamma) for _ in range(8)]
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
    sclera_intensity = 10.0 + 20.0 * np.exp(-dist_from_b8 / 25.0)
    raw_img[sclera_mask] = apply_gamma(sclera_intensity, bg_gamma)[sclera_mask]
    
    # Vitreous humor background (above retina)
    vitreous_mask = is_bg & (y_coords < b8[None, :])
    vitreous_intensity = np.full((height, width), 10.0, dtype=np.float32)
    raw_img[vitreous_mask] = apply_gamma(vitreous_intensity, bg_gamma)[vitreous_mask]
    
    # Apply Speckle Noise (Rayleigh/Gaussian simulation) and Clamping
    speckle = np.random.uniform(0.4, 1.2, size=(height, width))
    additive = np.random.uniform(-8.0, 8.0, size=(height, width))
    
    final_img = raw_img * speckle + additive
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

class PairedOCTDataset(Dataset):
    def __init__(self, synthetic_dir, real_dir, img_size, transform=None):
        self.synthetic_dir = synthetic_dir
        self.real_dir = real_dir
        self.img_size = img_size
        self.transform = transform
        
        self.filenames = sorted([
            f for f in os.listdir(real_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        print(f"Dataset successfully paired: {len(self.filenames)} files found.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        synth_path = os.path.join(self.synthetic_dir, filename)
        real_path = os.path.join(self.real_dir, filename)
        
        # Open in grayscale
        synth_img = Image.open(synth_path).convert('L')
        real_img = Image.open(real_path).convert('L')
        
        # Resize to specified training resolution
        synth_img = synth_img.resize((self.img_size, self.img_size), Image.BILINEAR)
        real_img = real_img.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        if self.transform:
            synth_img = self.transform(synth_img)
            real_img = self.transform(real_img)
            
        return synth_img, real_img

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
        PatchGAN (70x70) Classifier.
        Takes concatenated input/target images and classifies local patches.
        """
        super().__init__()
        model = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            model += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
            
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        model += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=False),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        model += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
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

def verify_setup(config):
    """
    Verifies that the entire network forward/backward pipeline is working
    correctly using dummy tensors on the chosen device.
    """
    print("\n--- Starting Dry-Run Verification ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Initialize networks
    generator = UNetGenerator(input_nc=1, output_nc=1, img_size=config['img_size']).to(device)
    discriminator = PatchGANDiscriminator(input_nc=2).to(device)
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
    criterion_L1 = nn.L1Loss()
    
    # Calculate losses & backward pass test
    loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
    loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
    loss_D = (loss_D_real + loss_D_fake) * 0.5
    
    loss_D.backward()
    print("Discriminator backward pass successful.")
    
    # Generator backward test
    pred_fake_g = discriminator(torch.cat([dummy_synthetic, fake_real], dim=1))
    loss_G_GAN = criterion_GAN(pred_fake_g, torch.ones_like(pred_fake_g))
    loss_G_L1 = criterion_L1(fake_real, dummy_real) * config['lambda_L1']
    loss_G = loss_G_GAN + loss_G_L1
    
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
    
    # Establish dataset directories
    train_synth_dir = os.path.join(os.path.dirname(config['train_data_path']), 'train_synthetic')
    test_synth_dir = os.path.join(os.path.dirname(config['test_data_path']), 'test_synthetic')
    
    # Prepare datasets (if not cached)
    prepare_synthetic_dataset(config['train_labels_path'], train_synth_dir)
    prepare_synthetic_dataset(config['test_labels_path'], test_synth_dir)
    
    # Normalization and Dataset setup
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    
    train_dataset = PairedOCTDataset(train_synth_dir, config['train_data_path'], config['img_size'], transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    
    test_dataset = PairedOCTDataset(test_synth_dir, config['test_data_path'], config['img_size'], transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Fixed validation inputs
    test_batch = next(iter(test_loader))
    test_synth_imgs, test_real_imgs = test_batch
    test_synth_imgs = test_synth_imgs.to(device)
    test_real_imgs = test_real_imgs.to(device)
    
    # Network Initialization
    generator = UNetGenerator(input_nc=1, output_nc=1, img_size=config['img_size'])
    discriminator = PatchGANDiscriminator(input_nc=2)
    
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
    criterion_L1 = nn.L1Loss()
    
    optimizer_G = optim.Adam(generator.parameters(), lr=config['learning_rate'], betas=(config['beta1'], 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config['learning_rate'], betas=(config['beta1'], 0.999))
    
    # MLflow Setup
    tracking_uri = config.get("mlflow_tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"Using remote MLflow tracking server: {tracking_uri}")

    mlflow.set_experiment(config['experiment_name'])
    run_name = f"Pix2Pix_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(config)
        mlflow.log_artifact(args.config)
        
        # Save fixed visual setup
        synth_grid = vutils.make_grid(test_synth_imgs, nrow=4, normalize=True)
        real_grid = vutils.make_grid(test_real_imgs, nrow=4, normalize=True)
        mlflow.log_image(tensor_to_numpy(synth_grid), "validation_synthetic_inputs.png")
        mlflow.log_image(tensor_to_numpy(real_grid), "validation_real_targets.png")
        
        epochs = config['epochs']
        for epoch in range(epochs):
            print(f"Starting epoch {epoch + 1}/{epochs}...")
            
            g_losses = []
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
                
                # Pixel-wise similarity (L1 Loss)
                loss_G_L1 = criterion_L1(fake_imgs, real_imgs) * config['lambda_L1']
                
                loss_G = loss_G_GAN + loss_G_L1
                loss_G.backward()
                optimizer_G.step()
                
                g_losses.append(loss_G.item())
                d_losses.append(loss_D.item())
                
            mean_g_loss = np.mean(g_losses)
            mean_d_loss = np.mean(d_losses)
            
            print(f"Epoch {epoch+1} - G loss: {mean_g_loss:.4f}, D loss: {mean_d_loss:.4f}")
            mlflow.log_metric("g_loss", mean_g_loss, step=epoch)
            mlflow.log_metric("d_loss", mean_d_loss, step=epoch)
            
            # Periodically save test visuals
            generator.eval()
            with torch.no_grad():
                test_fakes = generator(test_synth_imgs)
            
            test_fakes_grid = vutils.make_grid(test_fakes, nrow=4, normalize=True)
            mlflow.log_image(tensor_to_numpy(test_fakes_grid), f"validation_fakes_epoch_{epoch+1}.png")
            
            # Local visualization save & logging
            if epoch % 10 == 0 or epoch == epochs - 1:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(tensor_to_numpy(synth_grid[:16]))
                axes[0].set_title("Synthetic Input (Speckled)")
                axes[0].axis('off')
                
                axes[1].imshow(tensor_to_numpy(real_grid[:16]))
                axes[1].set_title("Real Target")
                axes[1].axis('off')
                
                axes[2].imshow(tensor_to_numpy(test_fakes_grid))
                axes[2].set_title("Generated Fake")
                axes[2].axis('off')
                
                plt.tight_layout()
                fig_path = f"sample_epoch_{epoch+1}.png"
                plt.savefig(fig_path)
                plt.close(fig)
                if os.path.exists(fig_path):
                    os.remove(fig_path)
                    
        # Log models at the end of the run
        gen_to_log = generator.module if isinstance(generator, nn.DataParallel) else generator
        disc_to_log = discriminator.module if isinstance(discriminator, nn.DataParallel) else discriminator
        mlflow.pytorch.log_model(gen_to_log, "generator_model", serialization_format="pickle")
        mlflow.pytorch.log_model(disc_to_log, "discriminator_model", serialization_format="pickle")
        print("Training completed and logged successfully.")

if __name__ == '__main__':
    main()
