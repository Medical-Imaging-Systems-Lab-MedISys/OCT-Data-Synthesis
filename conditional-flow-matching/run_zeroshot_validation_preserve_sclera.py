import os
import cv2
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import datetime
import argparse
from diffusers import UNet2DModel
from train_val import synthesize_from_mask, generate_samples

# Configuration
MLFLOW_URI = "http://10.24.38.15:5000"
EXPERIMENT_NAME = "OCT_CFM_ZeroShot_Validation_PreserveSclera"

# Paths to models & datasets
CHECKPOINTS = {
    "Baseline_8BitNorm": "checkpoints/cfm_model_cfm_8bitnorm_2026-07-14_14-51-24.pt",
    "L1_Spatial_Weight0.5": "checkpoints/cfm_model_cfm_8bitnorm_cropped_l1_weight0.5_2026-07-20_16-16-44.pt",
    "L2_Spatial_Weight0.5": "checkpoints/cfm_model_cfm_8bitnorm_cropped_l2_weight0.5_2026-07-20_16-16-48.pt"
}

DATA_ROOT = "DATA/NR206"

# Specific parameters requested by user
SAMPLE_PARAMS = {
    "NORMAL119": {"amplitude": 100.0, "center": 0.45, "width": 0.40},
    "NORMAL127": {"amplitude": 100.0, "center": 0.45, "width": 0.35},
    "NORMAL142": {"amplitude": 85.0,  "center": 0.45, "width": 0.40},
    "MORMAL142": {"amplitude": 85.0,  "center": 0.45, "width": 0.40},
}
DEFAULT_PARAMS = {"amplitude": 100.0, "center": 0.45, "width": 0.40}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def crop_and_pad_preserve_sclera(image, mask_bgra):
    """
    Preserves the real fibrous structure/sclera below the last retinal layer (b8)
    without replacing it with synthetic tiled background noise.
    """
    H, W = image.shape[:2]
    max_dim = max(H, W)
    pad_h = max_dim - H
    pad_w = max_dim - W
    
    if pad_h > 0 or pad_w > 0:
        if len(image.shape) == 3:
            final_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
        else:
            final_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='edge')
    else:
        final_image = image
        
    return final_image

def remove_watermark(img):
    if img.shape[0] >= 350 and img.shape[1] >= 600:
        clean_patch = img[350:, 600:]
        if clean_patch.size > 0:
            h_target = img[350:, :150].shape[0]
            w_target = img[350:, :150].shape[1]
            img[350:, :150] = np.flip(clean_patch, axis=1)[:h_target, :w_target]
    return img

def process_sample(img_path, mask_path, params):
    orig_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    orig_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if len(orig_mask.shape) == 2:
        orig_mask = cv2.cvtColor(orig_mask, cv2.COLOR_GRAY2BGRA)
    elif orig_mask.shape[2] == 3:
        orig_mask = cv2.cvtColor(orig_mask, cv2.COLOR_BGR2BGRA)

    # 1. Original Preprocessed (preserving sclera/fibrous structure below retina)
    img_clean = remove_watermark(orig_img.copy())
    orig_img_256 = cv2.resize(img_clean, (256, 256), interpolation=cv2.INTER_LINEAR)
    orig_mask_256 = cv2.resize(orig_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    orig_preprocessed = crop_and_pad_preserve_sclera(orig_img_256, orig_mask_256)

    # 2. Geometric modification
    H, W = orig_img.shape[:2]
    shifted_img = np.zeros_like(img_clean)
    shifted_mask = np.zeros_like(orig_mask)
    if len(orig_mask.shape) == 3 and orig_mask.shape[2] == 4:
        shifted_mask[:, :, 3] = 255
        
    amp = params["amplitude"]
    center = params["center"]
    width = params["width"]
    
    center_px = center * W
    width_px = width * W
    
    x = np.arange(W)
    dy = amp * np.exp(-((x - center_px)**2) / (2 * (width_px**2) + 1e-6))
    dy = np.round(dy).astype(int)
    
    for i in range(W):
        shift = dy[i]
        if shift > 0:
            shifted_img[shift:, i] = img_clean[:-shift, i]
            shifted_mask[shift:, i] = orig_mask[:-shift, i]
        else:
            shifted_img[:, i] = img_clean[:, i]
            shifted_mask[:, i] = orig_mask[:, i]
            
    # Crop ONLY pure black zero-padded band created by shift
    is_bg_shifted = (shifted_mask[:,:,0] == 0) & (shifted_mask[:,:,1] == 0) & (shifted_mask[:,:,2] == 0)
    is_ret_shifted = ~is_bg_shifted
    has_ret_shifted = np.any(is_ret_shifted, axis=0)
    
    max_dy = int(np.max(dy))
    if np.any(has_ret_shifted):
        top_y_per_col = np.argmax(is_ret_shifted, axis=0)
        min_top_y = np.min(top_y_per_col[has_ret_shifted])
        top_crop = min(max_dy, max(0, min_top_y - 5))
    else:
        top_crop = max_dy
        
    if top_crop > 0 and top_crop < shifted_img.shape[0]:
        shifted_img = shifted_img[top_crop:, :]
        shifted_mask = shifted_mask[top_crop:, :]
        
    geom_img_256 = cv2.resize(shifted_img, (256, 256), interpolation=cv2.INTER_LINEAR)
    geom_mask_256 = cv2.resize(shifted_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    geom_modified_img = crop_and_pad_preserve_sclera(geom_img_256, geom_mask_256)
    
    # 3. Prior (Synthetic Mask)
    prior_img = synthesize_from_mask(geom_mask_256)
    
    return orig_preprocessed, geom_modified_img, geom_mask_256, prior_img

def load_unet_model(checkpoint_path):
    model = UNet2DModel(
        sample_size=256,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    ).to(device)
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", type=str, default=None, help="Filter specific model key from CHECKPOINTS")
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    test_files = glob.glob(os.path.join(DATA_ROOT, "test", "*.png"))
    train_files = glob.glob(os.path.join(DATA_ROOT, "train", "*.png"))
    
    samples_map = {}
    for f in train_files:
        name = os.path.splitext(os.path.basename(f))[0]
        lbl = os.path.join(DATA_ROOT, "train_labels", os.path.basename(f))
        if os.path.exists(lbl):
            samples_map[name] = (f, lbl)
            
    for f in test_files:
        name = os.path.splitext(os.path.basename(f))[0]
        lbl = os.path.join(DATA_ROOT, "test_labels", os.path.basename(f))
        if os.path.exists(lbl):
            samples_map[name] = (f, lbl)
            
    target_sample_names = sorted(list(samples_map.keys()), key=natural_sort_key)
    print(f"Found {len(target_sample_names)} samples for zero-shot validation (Preserving Fibrous Sclera).")
    
    for model_name, ckpt_path in CHECKPOINTS.items():
        if args.model_key and model_name.lower() != args.model_key.lower():
            continue

        if not os.path.exists(ckpt_path):
            print(f"Skipping {model_name}: Checkpoint not found at {ckpt_path}")
            continue
            
        print(f"\n==========================================")
        print(f"Running Zero-Shot Validation (Preserve Sclera) for: {model_name}")
        print(f"Loading checkpoint: {ckpt_path}")
        print(f"==========================================")
        
        model = load_unet_model(ckpt_path)
        
        run_name = f"ZeroShot_Val_PreserveSclera_{model_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("checkpoint", ckpt_path)
            mlflow.log_param("model_name", model_name)
            mlflow.set_tag("mlflow.note.content", 
                           f"Zero-shot geometric validation on {model_name} preserving real fibrous sclera below retina. "
                           "Subplots per sample: Original (Preprocessed), Geometric Modified, Prior (Synthetic Mask), Generated Synthesis.")
            
            for name in target_sample_names:
                img_path, mask_path = samples_map[name]
                params = SAMPLE_PARAMS.get(name, DEFAULT_PARAMS)
                
                print(f"  -> Processing {name} with params (amp={params['amplitude']}, center={params['center']}, width={params['width']})...")
                
                orig_preprocessed, geom_modified_img, geom_mask_256, prior_img = process_sample(img_path, mask_path, params)
                
                prior_norm = (prior_img.astype(np.float32) / 127.5) - 1.0
                x0_tensor = torch.from_numpy(prior_norm).unsqueeze(0).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    x1_gen_tensor = generate_samples(model, x0_tensor, num_steps=50)
                    x1_gen_disp = (x1_gen_tensor.squeeze().cpu().numpy() + 1.0) / 2.0
                    x1_gen_disp = np.clip(x1_gen_disp, 0.0, 1.0)
                    
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                titles = [
                    "Original (Preprocessed)", 
                    "Geometric Modified", 
                    "Prior (Synthetic Mask)", 
                    "Generated Synthesis"
                ]
                images = [
                    orig_preprocessed / 255.0,
                    geom_modified_img / 255.0,
                    prior_img / 255.0,
                    x1_gen_disp
                ]
                
                for ax, title, img in zip(axes, titles, images):
                    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
                    ax.imshow(img, cmap='gray')
                    ax.axis('off')
                    
                plt.suptitle(f"Sample: {name} (Preserved Sclera) | Params: Amp={params['amplitude']}, Center={params['center']}, Width={params['width']}", fontsize=16, y=1.02)
                plt.tight_layout()
                
                mlflow.log_figure(fig, f"validation_grids/{name}_grid.png")
                plt.close(fig)
                
            print(f"Successfully completed zero-shot validation (Preserve Sclera) for {model_name}!")

if __name__ == "__main__":
    main()
