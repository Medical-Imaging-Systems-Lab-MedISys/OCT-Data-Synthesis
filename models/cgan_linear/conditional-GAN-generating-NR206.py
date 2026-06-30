#!/usr/bin/env python
# coding: utf-8

import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch import autograd
from torch.autograd import Variable
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from datetime import datetime

# Load configurations
# Parse command line overrides
import argparse

# -------------------------------------------------------------------
# Global Seeding for Reproducibility
# -------------------------------------------------------------------
import random
import numpy as np
import torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
try:
    import pytorch_lightning as pl
    pl.seed_everything(42, workers=True)
except ImportError:
    pass
# -------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'), help='Path to config file')
parser.add_argument('--num_gpus', type=int, default=None, help='Override number of GPUs to use')
args, unknown = parser.parse_known_args()

with open(args.config, 'r') as f:
    config = json.load(f)

if args.num_gpus is not None:
    config['num_gpus'] = args.num_gpus
    print(f"Overriding num_gpus from command line: {args.num_gpus}")


train_data_path = config['train_data_path']
train_labels_path = config['train_labels_path']
test_data_path = config['test_data_path']
test_labels_path = config['test_labels_path']
img_size = config['img_size']
batch_size = config['batch_size']
z_size = config.get('z_size', 100)
generator_layer_size = config.get('generator_layer_size', [256, 512, 1024])
discriminator_layer_size = config.get('discriminator_layer_size', [1024, 512, 256])
epochs = config['epochs']
learning_rate = config['learning_rate']
experiment_name = config.get('experiment_name', 'cGAN_NR206')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('torch version:', torch.__version__)
print('device:', device)

# Define MLflow experiment and run name
tracking_uri = config.get("mlflow_tracking_uri")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
    print(f"Using remote MLflow tracking server: {tracking_uri}")

mlflow.set_experiment(experiment_name)

# Set detailed markdown description for the experiment
try:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        client = mlflow.tracking.MlflowClient()
        experiment_description = (
            "# Linear Conditional GAN Retinal OCT Generation Experiment\n\n"
            "This experiment trains a baseline linear Conditional GAN (cGAN) to generate real-looking "
            "OCT scans conditioned on flattened layer segmentation masks.\n\n"
            "## Model Components:\n"
            "- **Generator:** Linear/Fully-Connected network mapping (noise z + flattened mask) to flattened images.\n"
            "- **Discriminator:** Linear/Fully-Connected network classifying (image + mask) pairs.\n"
            "- **Loss Functions:** BCELoss (Adversarial)."
        )
        client.set_experiment_tag(experiment.experiment_id, "mlflow.note.content", experiment_description)
except Exception as e:
    print(f"Warning: Could not set experiment description: {e}")

run_name = f"{experiment_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


# ## - Pytorch Dataset: NR206

def sample_gamma_from_bell_curve(min_g, max_g, rng=None):
    if rng is None: rng = np.random.RandomState()
    mean = (min_g + max_g) / 2.0
    std = (max_g - min_g) / 6.0
    return np.clip(rng.normal(mean, std), min_g, max_g)

def apply_gamma(val, g):
    return 255.0 * np.power(val / 255.0, g)

def synthesize_from_mask(mask_bgra, min_gamma=0.5, max_gamma=1.2, custom_intensities=None):
    import cv2
    height, width, _ = mask_bgra.shape
    raw_img = np.zeros((height, width), dtype=np.float32)
    
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
    
    layer_gammas = [sample_gamma_from_bell_curve(cfg['min_g'], cfg['max_g'], rng) for cfg in LAYERS_CFG]
    bg_gamma = sample_gamma_from_bell_curve(min_gamma, max_gamma, rng)
    
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
    
    speckle = rng.uniform(0.3, 1.2, size=(height, width))
    additive = rng.uniform(-12.0, 12.0, size=(height, width))
    
    final_img = raw_img * speckle + additive
    final_img[is_bg] = np.clip(final_img[is_bg], 0, 90.0)
    final_img = np.clip(final_img, 0, 255).astype(np.uint8)
    return final_img

def profile_single_image_intensities(real_img, mask_bgra, global_defaults):
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

class NR206Dataset(Dataset):
    def __init__(self, data_path, labels_path, img_size, transform=None, is_train=True):
        import cv2
        self.data_path = data_path
        self.labels_path = labels_path
        self.img_size = img_size
        self.transform = transform
        self.is_train = is_train
        
        self.filenames = sorted([
            f for f in os.listdir(data_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        print(f"Loaded {len(self.filenames)} samples from {data_path}. Preloading to VRAM...")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
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
        self.image_intensities = []
        
        for filename in self.filenames:
            img_path = os.path.join(self.data_path, filename)
            mask_path = os.path.join(self.labels_path, filename)
            
            # Load real image as grayscale ('L')
            img = Image.open(img_path).convert('L')
            real_np = np.array(img)
            
            # Remove watermark on the bottom-left corner
            clean_patch = real_np[350:, 600:]
            real_np[350:, :150] = np.flip(clean_patch, axis=1)
            
            # Load BGRA anatomical mask
            mask_bgra = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask_bgra is not None and len(mask_bgra.shape) == 3 and mask_bgra.shape[2] == 3:
                alpha = np.full((mask_bgra.shape[0], mask_bgra.shape[1], 1), 255, dtype=np.uint8)
                mask_bgra = np.concatenate([mask_bgra, alpha], axis=2)
            self.masks.append(mask_bgra)
            
            # Profile intensities
            img_intensities = profile_single_image_intensities(real_np, mask_bgra, global_defaults)
            for layer in img_intensities:
                img_intensities[layer] *= ratios[layer]
            self.image_intensities.append(img_intensities)
            
            # Resize and transform real image
            real_img_resized = Image.fromarray(real_np).resize((self.img_size, self.img_size), Image.BILINEAR)
            
            # Pre-synthesize static version for validation / fallback
            synth_np = synthesize_from_mask(mask_bgra, min_gamma=0.5, max_gamma=1.5, custom_intensities=img_intensities)
            synth_img = Image.fromarray(synth_np, mode='L').resize((self.img_size, self.img_size), Image.BILINEAR)
            
            if self.transform:
                real_img_resized = self.transform(real_img_resized)
                synth_img = self.transform(synth_img)
                
            self.real_images.append(real_img_resized.to(device))
            self.synth_images.append(synth_img.to(device))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if self.is_train:
            mask_bgra = self.masks[idx]
            custom_intensities = self.image_intensities[idx]
            synth_np = synthesize_from_mask(mask_bgra, min_gamma=0.5, max_gamma=1.5, custom_intensities=custom_intensities)
            synth_img = Image.fromarray(synth_np, mode='L').resize((self.img_size, self.img_size), Image.BILINEAR)
            if self.transform:
                synth_img = self.transform(synth_img)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            return self.real_images[idx], synth_img.to(device)
        else:
            return self.real_images[idx], self.synth_images[idx]


# ## - Generator

class Generator(nn.Module):
    def __init__(self, generator_layer_size, z_size, img_size):
        super().__init__()
        
        self.z_size = z_size
        self.img_size = img_size
        
        self.model = nn.Sequential(
            nn.Linear(self.z_size + self.img_size * self.img_size, generator_layer_size[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[0], generator_layer_size[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[1], generator_layer_size[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[2], self.img_size * self.img_size),
            nn.Tanh()
        )
    
    def forward(self, z, priors):
        # Reshape z
        z = z.view(-1, self.z_size)
        
        # Flatten priors
        priors_flat = priors.view(-1, self.img_size * self.img_size)
        
        # Concat noise & priors
        x = torch.cat([z, priors_flat], 1)
        
        # Generator out
        out = self.model(x)
        
        return out.view(-1, 1, self.img_size, self.img_size)


# ## - Discriminator

class Discriminator(nn.Module):
    def __init__(self, discriminator_layer_size, img_size):
        super().__init__()
        
        self.img_size = img_size
        
        self.model = nn.Sequential(
            nn.Linear(self.img_size * self.img_size * 2, discriminator_layer_size[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(discriminator_layer_size[0], discriminator_layer_size[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(discriminator_layer_size[1], discriminator_layer_size[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(discriminator_layer_size[2], 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, priors):
        # Flatten image & priors
        x_flat = x.view(-1, self.img_size * self.img_size)
        priors_flat = priors.view(-1, self.img_size * self.img_size)
        
        # Concat image & priors
        x_concat = torch.cat([x_flat, priors_flat], 1)
        
        # Discriminator out
        out = self.model(x_concat)
        
        return out.squeeze()


# ## - Adversarial Learning steps

def generator_train_step(curr_batch_size, discriminator, generator, g_optimizer, criterion, real_images, priors, lambda_L1):
    g_optimizer.zero_grad()
    
    # Building z
    z = Variable(torch.randn(curr_batch_size, z_size)).to(device)
    
    # Generating fake images
    fake_images = generator(z, priors)
    
    # Disciminating fake images
    validity = discriminator(fake_images, priors)
    
    # Calculating discrimination loss (fake images)
    g_loss_GAN = criterion(validity, Variable(torch.ones(curr_batch_size)).to(device))
    
    # L1 reconstruction loss between generated fake and ground truth target
    g_loss_L1 = torch.mean(torch.abs(fake_images - real_images)) * lambda_L1
    
    g_loss = g_loss_GAN + g_loss_L1
    
    g_loss.backward()
    g_optimizer.step()
    
    return g_loss.item(), g_loss_GAN.item(), g_loss_L1.item()


def discriminator_train_step(curr_batch_size, discriminator, generator, d_optimizer, criterion, real_images, priors):
    d_optimizer.zero_grad()

    # Disciminating real images
    real_validity = discriminator(real_images, priors)
    
    # Calculating discrimination loss (real images)
    real_loss = criterion(real_validity, Variable(torch.ones(curr_batch_size)).to(device))
    
    # Building z
    z = Variable(torch.randn(curr_batch_size, z_size)).to(device)
    
    # Generating fake images
    fake_images = generator(z, priors)
    
    # Disciminating fake images
    fake_validity = discriminator(fake_images, priors)
    
    # Calculating discrimination loss (fake images)
    fake_loss = criterion(fake_validity, Variable(torch.zeros(curr_batch_size)).to(device))
    
    # Sum two losses
    d_loss = real_loss + fake_loss
    
    d_loss.backward()
    d_optimizer.step()
    
    return d_loss.item()


# Helper to convert PyTorch grid to numpy [0, 255] uint8
def tensor_to_numpy(grid_tensor):
    np_grid = grid_tensor.cpu().numpy()
    np_grid = np.transpose(np_grid, (1, 2, 0))
    np_grid = (np_grid * 255).astype(np.uint8)
    return np_grid


# ## - Main Training Loop

def main():
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    
    train_dataset = NR206Dataset(train_data_path, train_labels_path, img_size, transform=transform, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = NR206Dataset(test_data_path, test_labels_path, img_size, transform=transform, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Fixed batch of test inputs for evaluation
    test_batch = next(iter(test_loader))
    test_real_images, test_priors = test_batch
    test_real_images = test_real_images.to(device)
    test_priors = test_priors.to(device)
    
    # Initialize networks
    generator = Generator(generator_layer_size, z_size, img_size)
    discriminator = Discriminator(discriminator_layer_size, img_size)
    
    # Check dataset size and toggles for multi-GPU
    use_multi_gpu = config.get("use_multi_gpu", True)
    small_dataset_threshold = config.get("small_dataset_threshold", 500)
    dataset_size = len(train_dataset)
    
    if use_multi_gpu and dataset_size < small_dataset_threshold:
        print(f"Dataset size ({dataset_size}) is smaller than threshold ({small_dataset_threshold}). Restricting to single GPU.")
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
    
    criterion = nn.BCELoss()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
    
    with mlflow.start_run(run_name=run_name) as run:
        # Set run description note
        mlflow.set_tag("mlflow.note.content", f"Linear cGAN training run with batch size {batch_size}, {epochs} epochs, and {img_size}x{img_size} resolution.")
        # Log parameters
        mlflow.log_params(config)
        # Log configuration file as artifact
        mlflow.log_artifact(args.config)
        
        # Log fixed inputs
        test_real_grid = vutils.make_grid(test_real_images, nrow=4, normalize=False)
        test_real_grid = (test_real_grid + 1.0) / 2.0
        test_real_grid = torch.clamp(test_real_grid, 0.0, 1.0)
        
        test_prior_grid = vutils.make_grid(test_priors, nrow=4, normalize=False)
        test_prior_grid = (test_prior_grid + 1.0) / 2.0
        test_prior_grid = torch.clamp(test_prior_grid, 0.0, 1.0)
        
        mlflow.log_image(tensor_to_numpy(test_real_grid), "test_real_images.png")
        mlflow.log_image(tensor_to_numpy(test_prior_grid), "test_priors.png")
        
        global_step = 0
        
        for epoch in range(epochs):
            print(f"Starting epoch {epoch+1}/{epochs}...")
            
            epoch_g_losses = []
            epoch_g_losses_gan = []
            epoch_g_losses_l1 = []
            epoch_d_losses = []
            for batch_idx, data in enumerate(train_loader):
                real_images = data['real'].float()
                priors = data['prior'].float()
                
                real_images = Variable(real_images).to(device)
                priors = Variable(priors).to(device)
                curr_batch_size = real_images.size(0)
                
                # Train networks
                d_loss = discriminator_train_step(
                    curr_batch_size, discriminator, generator, d_optimizer, criterion, real_images, priors
                )
                g_loss, g_loss_gan, g_loss_l1 = generator_train_step(
                    curr_batch_size, discriminator, generator, g_optimizer, criterion, real_images, priors, config.get('lambda_L1', 100.0)
                )
                
                epoch_g_losses.append(g_loss)
                epoch_g_losses_gan.append(g_loss_gan)
                epoch_g_losses_l1.append(g_loss_l1)
                epoch_d_losses.append(d_loss)
                
                # Log batch-level metrics
                mlflow.log_metric("batch_g_loss", g_loss, step=global_step)
                mlflow.log_metric("batch_g_loss_gan", g_loss_gan, step=global_step)
                mlflow.log_metric("batch_g_loss_l1", g_loss_l1, step=global_step)
                mlflow.log_metric("batch_d_loss", d_loss, step=global_step)
                global_step += 1
                
            mean_g_loss = np.mean(epoch_g_losses)
            mean_g_loss_gan = np.mean(epoch_g_losses_gan)
            mean_g_loss_l1 = np.mean(epoch_g_losses_l1)
            mean_d_loss = np.mean(epoch_d_losses)
            print(f"Epoch {epoch+1} - Train g_loss: {mean_g_loss:.4f}, Train d_loss: {mean_d_loss:.4f}")
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
                for val_images, val_priors in test_loader:
                    curr_val_batch_size = val_images.size(0)
                    val_images = val_images.to(device)
                    val_priors = val_priors.to(device)
                    
                    val_z = Variable(torch.randn(curr_val_batch_size, z_size)).to(device)
                    val_fake_images = generator(val_z, val_priors)
                    
                    # D validation loss
                    real_validity = discriminator(val_images, val_priors)
                    fake_validity = discriminator(val_fake_images, val_priors)
                    real_loss = criterion(real_validity, Variable(torch.ones(curr_val_batch_size)).to(device))
                    fake_loss = criterion(fake_validity, Variable(torch.zeros(curr_val_batch_size)).to(device))
                    val_d_loss = (real_loss + fake_loss).item()
                    
                    # G validation loss
                    fake_validity_g = discriminator(val_fake_images, val_priors)
                    val_g_loss_gan = criterion(fake_validity_g, Variable(torch.ones(curr_val_batch_size)).to(device)).item()
                    val_g_loss_l1 = (torch.mean(torch.abs(val_fake_images - val_images)) * config.get('lambda_L1', 100.0)).item()
                    val_g_loss = val_g_loss_gan + val_g_loss_l1
                    
                    val_g_losses.append(val_g_loss)
                    val_g_losses_gan.append(val_g_loss_gan)
                    val_g_losses_l1.append(val_g_loss_l1)
                    val_d_losses.append(val_d_loss)
                    
            mean_val_g_loss = np.mean(val_g_losses)
            mean_val_g_loss_gan = np.mean(val_g_losses_gan)
            mean_val_g_loss_l1 = np.mean(val_g_losses_l1)
            mean_val_d_loss = np.mean(val_d_losses)
            print(f"Epoch {epoch+1} - Val g_loss: {mean_val_g_loss:.4f} (GAN: {mean_val_g_loss_gan:.4f}, L1: {mean_val_g_loss_l1:.4f}), Val d_loss: {mean_val_d_loss:.4f}")
            mlflow.log_metric("val_g_loss", mean_val_g_loss, step=epoch)
            mlflow.log_metric("val_g_loss_gan", mean_val_g_loss_gan, step=epoch)
            mlflow.log_metric("val_g_loss_l1", mean_val_g_loss_l1, step=epoch)
            mlflow.log_metric("val_d_loss", mean_val_d_loss, step=epoch)
            
            generator.train()
            discriminator.train()
            
            # Generate and log evaluation images on test priors (every 5 epochs)
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                generator.eval()
                with torch.no_grad():
                    test_z = torch.randn(test_priors.size(0), z_size).to(device)
                    test_fake_images = generator(test_z, test_priors)
                
                # Log exactly 3 comparison image grid files (each with prior, fake, real side by side)
                for i in range(min(3, test_priors.size(0))):
                    grid_tensors = [test_priors[i].expand(3,-1,-1), test_fake_images[i].expand(3,-1,-1), test_real_images[i].expand(3,-1,-1)]
                    grid = vutils.make_grid(grid_tensors, nrow=3, normalize=False)
                    grid = (grid + 1.0) / 2.0
                    grid = torch.clamp(grid, 0.0, 1.0)
                    mlflow.log_image(tensor_to_numpy(grid), f"validation_grid_{i+1}_epoch_{epoch+1}.png")
            
            # Log periodic training samples for progress tracking
            if epoch % 5 == 0 or epoch == epochs - 1:
                # Log first batch of train samples
                with torch.no_grad():
                    train_z = torch.randn(min(16, curr_batch_size), z_size).to(device)
                    train_fake_images = generator(train_z, priors[:16])
                
                train_fake_grid = vutils.make_grid(train_fake_images, nrow=4, normalize=False)
                train_fake_grid = (train_fake_grid + 1.0) / 2.0
                train_fake_grid = torch.clamp(train_fake_grid, 0.0, 1.0)
                
                train_real_grid = vutils.make_grid(real_images[:16], nrow=4, normalize=False)
                train_real_grid = (train_real_grid + 1.0) / 2.0
                train_real_grid = torch.clamp(train_real_grid, 0.0, 1.0)
                
                train_prior_grid = vutils.make_grid(priors[:16], nrow=4, normalize=False)
                train_prior_grid = (train_prior_grid + 1.0) / 2.0
                train_prior_grid = torch.clamp(train_prior_grid, 0.0, 1.0)
                
                mlflow.log_image(tensor_to_numpy(train_real_grid), f"train_real_images_epoch_{epoch+1}.png")
                mlflow.log_image(tensor_to_numpy(train_prior_grid), f"train_priors_epoch_{epoch+1}.png")
                mlflow.log_image(tensor_to_numpy(train_fake_grid), f"train_fake_images_epoch_{epoch+1}.png")
                
                # Show on plot (without blocking)
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(tensor_to_numpy(train_real_grid))
                axes[0].set_title("Real Image")
                axes[0].axis('off')
                axes[1].imshow(tensor_to_numpy(train_prior_grid))
                axes[1].set_title("Prior Image")
                axes[1].axis('off')
                axes[2].imshow(tensor_to_numpy(train_fake_grid))
                axes[2].set_title("Fake Image")
                axes[2].axis('off')
                plt.tight_layout()
                # Save and close plot to avoid UI blocking/memory issues
                fig_path = f"sample_epoch_{epoch+1}.png"
                plt.savefig(fig_path)
                plt.close(fig)
                if os.path.exists(fig_path):
                    os.remove(fig_path)
                    
            generator.train()

        # Log final pytorch models (unwrapped if DataParallel)
        gen_to_log = generator.module if isinstance(generator, nn.DataParallel) else generator
        disc_to_log = discriminator.module if isinstance(discriminator, nn.DataParallel) else discriminator
        mlflow.pytorch.log_model(gen_to_log, "generator_model", serialization_format="pickle")
        mlflow.pytorch.log_model(disc_to_log, "discriminator_model", serialization_format="pickle")
        print("Training completed and logged successfully.")


if __name__ == "__main__":
    main()
