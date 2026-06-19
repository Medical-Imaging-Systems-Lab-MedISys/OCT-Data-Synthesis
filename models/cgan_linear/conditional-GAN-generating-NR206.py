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
try:
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
except NameError:
    config_path = 'config.json'

with open(config_path, 'r') as f:
    config = json.load(f)

# Parse command line overrides
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_gpus', type=int, default=None, help='Override number of GPUs to use')
args, unknown = parser.parse_known_args()

if args.num_gpus is not None:
    config['num_gpus'] = args.num_gpus
    print(f"Overriding num_gpus from command line: {args.num_gpus}")


train_data_path = config['train_data_path']
train_labels_path = config['train_labels_path']
test_data_path = config['test_data_path']
test_labels_path = config['test_labels_path']
img_size = config['img_size']
batch_size = config['batch_size']
z_size = config['z_size']
generator_layer_size = config['generator_layer_size']
discriminator_layer_size = config['discriminator_layer_size']
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

class NR206Dataset(Dataset):
    def __init__(self, data_path, labels_path, img_size, transform=None):
        self.data_path = data_path
        self.labels_path = labels_path
        self.img_size = img_size
        self.transform = transform
        
        # List all image files
        self.filenames = sorted([
            f for f in os.listdir(data_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        print(f"Loaded {len(self.filenames)} samples from {data_path}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = os.path.join(self.data_path, filename)
        mask_path = os.path.join(self.labels_path, filename)
        
        # Load image and mask as grayscale ('L')
        img = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        
        # Remove watermark on the bottom-left corner of the real image (in original 500x750 coordinate space)
        img_np = np.array(img)
        img_np[350:, :150] = 0
        img = Image.fromarray(img_np)
        
        # Resize
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        
        return img, mask


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
    
    def forward(self, z, masks):
        # Reshape z
        z = z.view(-1, self.z_size)
        
        # Flatten masks
        masks_flat = masks.view(-1, self.img_size * self.img_size)
        
        # Concat noise & masks
        x = torch.cat([z, masks_flat], 1)
        
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
    
    def forward(self, x, masks):
        # Flatten image & masks
        x_flat = x.view(-1, self.img_size * self.img_size)
        masks_flat = masks.view(-1, self.img_size * self.img_size)
        
        # Concat image & masks
        x_concat = torch.cat([x_flat, masks_flat], 1)
        
        # Discriminator out
        out = self.model(x_concat)
        
        return out.squeeze()


# ## - Adversarial Learning steps

def generator_train_step(curr_batch_size, discriminator, generator, g_optimizer, criterion, masks):
    g_optimizer.zero_grad()
    
    # Building z
    z = Variable(torch.randn(curr_batch_size, z_size)).to(device)
    
    # Generating fake images
    fake_images = generator(z, masks)
    
    # Disciminating fake images
    validity = discriminator(fake_images, masks)
    
    # Calculating discrimination loss (fake images)
    g_loss = criterion(validity, Variable(torch.ones(curr_batch_size)).to(device))
    
    g_loss.backward()
    g_optimizer.step()
    
    return g_loss.item()


def discriminator_train_step(curr_batch_size, discriminator, generator, d_optimizer, criterion, real_images, masks):
    d_optimizer.zero_grad()

    # Disciminating real images
    real_validity = discriminator(real_images, masks)
    
    # Calculating discrimination loss (real images)
    real_loss = criterion(real_validity, Variable(torch.ones(curr_batch_size)).to(device))
    
    # Building z
    z = Variable(torch.randn(curr_batch_size, z_size)).to(device)
    
    # Generating fake images
    fake_images = generator(z, masks)
    
    # Disciminating fake images
    fake_validity = discriminator(fake_images, masks)
    
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
    
    train_dataset = NR206Dataset(train_data_path, train_labels_path, img_size, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = NR206Dataset(test_data_path, test_labels_path, img_size, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Fixed batch of test inputs for evaluation
    test_batch = next(iter(test_loader))
    test_real_images, test_masks = test_batch
    test_real_images = test_real_images.to(device)
    test_masks = test_masks.to(device)
    
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
        mlflow.log_artifact(config_path)
        
        # Log fixed inputs
        test_real_grid = vutils.make_grid(test_real_images, nrow=4, normalize=True)
        test_mask_grid = vutils.make_grid(test_masks, nrow=4, normalize=True)
        mlflow.log_image(tensor_to_numpy(test_real_grid), "test_real_images.png")
        mlflow.log_image(tensor_to_numpy(test_mask_grid), "test_masks.png")
        
        for epoch in range(epochs):
            print(f"Starting epoch {epoch+1}/{epochs}...")
            
            epoch_g_losses = []
            epoch_d_losses = []
            
            for i, (images, masks) in enumerate(train_loader):
                real_images = Variable(images).to(device)
                masks = Variable(masks).to(device)
                curr_batch_size = real_images.size(0)
                
                # Train networks
                d_loss = discriminator_train_step(
                    curr_batch_size, discriminator, generator, d_optimizer, criterion, real_images, masks
                )
                g_loss = generator_train_step(
                    curr_batch_size, discriminator, generator, g_optimizer, criterion, masks
                )
                
                epoch_g_losses.append(g_loss)
                epoch_d_losses.append(d_loss)
                
            mean_g_loss = np.mean(epoch_g_losses)
            mean_d_loss = np.mean(epoch_d_losses)
            print(f"Epoch {epoch+1} - g_loss: {mean_g_loss:.4f}, d_loss: {mean_d_loss:.4f}")
            
            # Log metrics to mlflow
            mlflow.log_metric("g_loss", mean_g_loss, step=epoch)
            mlflow.log_metric("d_loss", mean_d_loss, step=epoch)
            
            # Generate and log evaluation images on test masks
            generator.eval()
            with torch.no_grad():
                test_z = torch.randn(test_masks.size(0), z_size).to(device)
                test_fake_images = generator(test_z, test_masks)
            
            test_fake_grid = vutils.make_grid(test_fake_images, nrow=4, normalize=True)
            mlflow.log_image(tensor_to_numpy(test_fake_grid), f"test_fake_images_epoch_{epoch+1}.png")
            
            # Log periodic training samples for progress tracking
            if epoch % 5 == 0 or epoch == epochs - 1:
                # Log first batch of train samples
                train_real_grid = vutils.make_grid(real_images[:16], nrow=4, normalize=True)
                train_mask_grid = vutils.make_grid(masks[:16], nrow=4, normalize=True)
                with torch.no_grad():
                    train_z = torch.randn(min(16, curr_batch_size), z_size).to(device)
                    train_fake_images = generator(train_z, masks[:16])
                train_fake_grid = vutils.make_grid(train_fake_images, nrow=4, normalize=True)
                
                mlflow.log_image(tensor_to_numpy(train_real_grid), f"train_real_images_epoch_{epoch+1}.png")
                mlflow.log_image(tensor_to_numpy(train_mask_grid), f"train_masks_epoch_{epoch+1}.png")
                mlflow.log_image(tensor_to_numpy(train_fake_grid), f"train_fake_images_epoch_{epoch+1}.png")
                
                # Show on plot (without blocking)
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(tensor_to_numpy(train_real_grid))
                axes[0].set_title("Real Image")
                axes[0].axis('off')
                axes[1].imshow(tensor_to_numpy(train_mask_grid))
                axes[1].set_title("Mask")
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
