import os
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import mlflow
import numpy as np


# Import torchcfm (from the cloned repository)
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

# Assuming you are using diffusers for the U-Net architecture, 
# which is highly recommended for Flow Matching image generation.
# pip install diffusers
from diffusers import UNet2DModel

# Import your custom dataset here (e.g., the NR206PairedDataset we discussed)
# from dataset import NR206PairedDataset

import datetime
# ==========================================
# 1. Configuration & Hyperparameters
LOSS_TYPE = "l1"  # Set to "l1" or "l2"

CONFIG = {
    "experiment_name": "Exp12_CFM_Base",
    "run_name": f"CFM_{LOSS_TYPE.upper()}_Cropped_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    "loss_type": LOSS_TYPE,
    "mlflow_tracking_uri": "https://dagshub.com/IISc-MedISys/OCT-Data-Synthesis.mlflow",
    "batch_size": 16,
    "epochs": 100,
    "learning_rate": 0.0002,
    "image_size": 256,         # Adjust to your OCT dimensions (e.g., 256 or 512)
    "in_channels": 2,             # Grayscale OCT
    "out_channels": 1,
    "val_check_interval": 5,   # Log images every 5 epochs
    "num_val_images": 3,       # Exactly 3 image grids per logging
    "inference_steps": 50,     # ODE Solver steps
    "sigma": 0.0               # Deterministic path for paired data
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. ODE Solver (Euler Method) for Inference
# ==========================================
@torch.no_grad()
def generate_samples_with_guidance(model, x0, num_steps=50):
    """
    Solves the learned ODE translating the prior (x0) to the real image,
    while continuously conditioning the model on the static prior x0.
    
    Args:
        model: The trained vector field network (initialized with in_channels=2)
        x0: The synthetic prior image tensor of shape [B, 1, H, W]
        num_steps: Number of integration steps for the Euler solver
        
    Returns:
        x_t: The synthesized OCT image tensor of shape [B, 1, H, W]
    """
    model.eval()
    device = x0.device
    
    # Initialize the evolving image x_t at time t=0
    x_t = x0.clone()
    
    # Step size for Euler integration
    dt = 1.0 / num_steps
    
    # Iterate through time from t=0 to t=1
    for i in range(num_steps):
        # Calculate current continuous time t
        t_val = i / num_steps
        t_batch = torch.full((x_t.shape[0],), t_val, device=device)
        
        # ==============================================================
        # CRITICAL MODIFICATION: Continuous Prior Conditioning
        # Concatenate the evolving image (x_t) with the original prior (x0)
        # x_t shape: [B, 1, H, W]
        # x0 shape:  [B, 1, H, W]
        # model_input shape: [B, 2, H, W]
        # ==============================================================
        model_input = torch.cat([x_t, x0], dim=1)
        
        # Predict the velocity field using the concatenated 2-channel input
        output = model(model_input, t_batch)
        
        # Handle Diffusers UNet2DModel (.sample) vs standard PyTorch UNet
        v_pred = output.sample if hasattr(output, 'sample') else output
        
        # Euler update step: x_{t+dt} = x_t + v * dt
        # We apply the predicted 1-channel velocity only to the evolving image (x_t)
        x_t = x_t + v_pred * dt
        
    return x_t

# ==========================================
# 3. Validation Image Grid Logger
# ==========================================
def log_validation_grids(x0, x1_gen, x1_gt, epoch, batch_idx):
    """
    Creates a Matplotlib figure with Prior (Left), Generated (Middle), 
    and Ground Truth (Right), labels them, and logs to MLflow.
    """
    # Move to CPU and un-normalize from [-1, 1] to [0, 1] for plotting
    x0_disp = (x0.squeeze().cpu().numpy() + 1.0) / 2.0
    x1_gen_disp = (x1_gen.squeeze().cpu().numpy() + 1.0) / 2.0
    x1_gt_disp = (x1_gt.squeeze().cpu().numpy() + 1.0) / 2.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Prior (Synthetic Mask)", "Generated Synthesis", "Ground Truth (Real)"]
    images = [x0_disp, x1_gen_disp, x1_gt_disp]

    for ax, title, img in zip(axes, titles, images):
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.imshow(img, cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    
    # Save to MLflow
    mlflow.log_figure(fig, f"validation_grids/epoch_{epoch}_sample_{batch_idx}.png")
    plt.close(fig)


# =====================================================================
# 1. Image Synthesis Helpers (Dynamic Prior Generation)
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
# 2. PyTorch Dataset Class
# =====================================================================

class NR206DynamicDataset(Dataset):
    """
    Dynamically generates the synthetic prior (x0) from the segmentation mask
    at runtime to ensure unique noise/speckle profiles per epoch.
    Author: Mohan Kumar Manepalli
    """
    def __init__(self, labels_dir, real_dir, min_gamma=0.5, max_gamma=1.5):
        self.labels_dir = labels_dir
        self.real_dir = real_dir
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        
        # Assume 1-to-1 mapping based on filenames in the real directory
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
        
        # 1. Load the Ground Truth Real OCT
        x1_img = cv2.imread(real_path, cv2.IMREAD_GRAYSCALE)
        
        # Remove watermark dynamically
        clean_patch = x1_img[350:, 600:]
        x1_img[350:, :150] = np.flip(clean_patch, axis=1)
            
        # 2. Load the Segmentation Mask (BGRA)
        mask_bgra = cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED)
        if len(mask_bgra.shape) == 3 and mask_bgra.shape[2] == 3:
            alpha = np.full((mask_bgra.shape[0], mask_bgra.shape[1], 1), 255, dtype=np.uint8)
            mask_bgra = np.concatenate([mask_bgra, alpha], axis=2)
            
        # 3. Dynamically Generate Synthetic Prior
        x0_img = synthesize_from_mask(mask_bgra, self.min_gamma, self.max_gamma)
        
        # ==============================================================
        # FIX: Resize images to a strict power of 2 (e.g., 256x256)
        # ==============================================================
        target_size = (256, 256) # (width, height) for cv2. Change to (512, 512) if needed.
        x0_img_squashed = cv2.resize(x0_img, target_size, interpolation=cv2.INTER_LINEAR)
        x1_img_squashed = cv2.resize(x1_img, target_size, interpolation=cv2.INTER_LINEAR)
        mask_squashed = cv2.resize(mask_bgra, target_size, interpolation=cv2.INTER_LINEAR)
        
        # ==============================================================
        # Crop and pad images below the last curved layer AFTER resizing
        # ==============================================================
        x0_img = crop_and_pad_curved(x0_img_squashed, mask_squashed)
        x1_img = crop_and_pad_curved(x1_img_squashed, mask_squashed)
        
        # 4. Normalize to [-1.0, 1.0]
        x0 = (x0_img.astype(np.float32) / 127.5) - 1.0
        x1 = (x1_img.astype(np.float32) / 127.5) - 1.0
        
        # 5. Convert to PyTorch Tensors
        x0_tensor = torch.from_numpy(x0).unsqueeze(0)
        x1_tensor = torch.from_numpy(x1).unsqueeze(0)
        
        return x0_tensor, x1_tensor

# ==========================================
# 4. Main Training Routine
# ==========================================
def train():
    # Set up MLflow
    mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
    mlflow.set_experiment(CONFIG["experiment_name"])
    # ==========================================
    # Dynamic Data Loading from /tmp
    # ==========================================
    local_data_dir = os.environ.get("LOCAL_DATA_DIR", "./NR206")
    print(f"Loading real images and labels from: {local_data_dir}")

    # Define paths
    train_real = os.path.join(local_data_dir, "train")
    train_labels = os.path.join(local_data_dir, "train_labels")

    val_real = os.path.join(local_data_dir, "test")
    val_labels = os.path.join(local_data_dir, "test_labels")

    # Initialize Datasets
    train_dataset = NR206DynamicDataset(
        labels_dir=train_labels, 
        real_dir=train_real,
        min_gamma=0.5,
        max_gamma=1.5
    )
    
    val_dataset = NR206DynamicDataset(
        labels_dir=val_labels, 
        real_dir=val_real,
        min_gamma=0.5,
        max_gamma=1.5
    )

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=6,       # Parallel CPU generation for training
        pin_memory=True,     
        prefetch_factor=2    
    )
    
    # ADD THIS BACK IN: The Validation Loader
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1,        # Batch size 1 makes grid logging much easier
        shuffle=False, 
        num_workers=2,       # Keep a couple workers for val data generation too
        pin_memory=True
    )
    # Initialize Model (Vector Field Network)
    # A standard UNet adapted for continuous time embeddings
    model = UNet2DModel(
        sample_size=CONFIG["image_size"],
        in_channels=CONFIG["in_channels"],
        out_channels=CONFIG["out_channels"],
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])

    # Initialize Conditional Flow Matcher from torchcfm
    # sigma=0.0 maps a deterministic straight line between x0 and x1
    FM = ConditionalFlowMatcher(sigma=CONFIG["sigma"])

    print(f"Starting training on {device}...")

    with mlflow.start_run(run_name=CONFIG["run_name"]):
        mlflow.log_params(CONFIG)
        
        # We can also add detailed notes or tags to describe this specific experiment version.
        loss_upper = CONFIG.get('loss_type', 'l2').upper()
        mlflow.set_tag("mlflow.note.content", f"CFM training run with curved crop padding (squashed) and {loss_upper} Loss.")

        global_step = 0

        for epoch in range(1, CONFIG["epochs"] + 1):
            model.train()
            total_train_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']}")
            for x0, x1 in pbar:
                x0, x1 = x0.to(device), x1.to(device)
                optimizer.zero_grad()

                # torchcfm handles the time sampling and target velocity calculation
                # u_t is the target vector field (velocity)
                t, x_t, u_t = FM.sample_location_and_conditional_flow(x0, x1)
                
                # Model predicts the velocity field
                model_input = torch.cat([x_t, x0], dim=1)
                v_pred = model(model_input, t.squeeze()).sample
                
                # Flow Matching Objective: Match predicted velocity to target velocity
                if CONFIG.get("loss_type") == "l2":
                    loss = F.mse_loss(v_pred, u_t)
                else:
                    loss = F.l1_loss(v_pred, u_t)
                
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                global_step += 1
                
                # Log step-level metric
                mlflow.log_metric("train_loss_step", loss.item(), step=global_step)
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            avg_train_loss = total_train_loss / len(train_loader)
            mlflow.log_metric("train_loss_epoch", avg_train_loss, step=epoch)

            # ==========================================
            # Validation & Image Logging Phase
            # ==========================================
            if epoch % CONFIG["val_check_interval"] == 0 or epoch == 1:
                model.eval()
                total_val_loss = 0.0
                logged_images_count = 0

                with torch.no_grad():
                    for batch_idx, (val_x0, val_x1) in enumerate(val_loader):
                        val_x0, val_x1 = val_x0.to(device), val_x1.to(device)

                        # Compute Validation Loss (L1 or L2 of vector field)
                        t, x_t, u_t = FM.sample_location_and_conditional_flow(val_x0, val_x1)
                        
                        # --- MODIFICATION 1: Concatenate x_t and prior val_x0 ---
                        model_input = torch.cat([x_t, val_x0], dim=1)
                        
                        v_pred = model(model_input, t.squeeze()).sample
                        if CONFIG.get("loss_type") == "l2":
                            val_loss = F.mse_loss(v_pred, u_t)
                        else:
                            val_loss = F.l1_loss(v_pred, u_t)
                        total_val_loss += val_loss.item()

                        # Image Generation & Logging (Only log up to num_val_images)
                        if logged_images_count < CONFIG["num_val_images"]:
                            # --- MODIFICATION 2: Call the new guidance-based ODE solver ---
                            x1_gen = generate_samples_with_guidance(model, val_x0, num_steps=CONFIG["inference_steps"])

                            # Create and log the grid figure
                            # Assuming batch size of 1 for validation dataloader for clean grids
                            log_validation_grids(
                                val_x0[0], 
                                x1_gen[0], 
                                val_x1[0], 
                                epoch, 
                                batch_idx
                            )
                            logged_images_count += 1

                avg_val_loss = total_val_loss / len(val_loader)
                mlflow.log_metric("val_loss_epoch", avg_val_loss, step=epoch)
                print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save final model checkpoint to MLflow
        checkpoint_path = "final_model.pth"
        torch.save(model.state_dict(), checkpoint_path)
        mlflow.log_artifact(checkpoint_path)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

if __name__ == "__main__":
    train()
