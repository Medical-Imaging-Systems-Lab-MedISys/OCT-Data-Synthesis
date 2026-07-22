#!/usr/bin/env python3
import os
import random
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

# Config paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_DIR = os.path.join(REPO_ROOT, "DATA/NR206/train")
MASK_DIR = os.path.join(REPO_ROOT, "DATA/NR206/train_labels")
OUTPUT_ROOT = os.path.join(REPO_ROOT, "DATA/NR206_augmented")

def crop_and_pad_curved(image, mask_bgra, orig_image=None):
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
    
    if orig_image is not None:
        if orig_image.shape[:2] != (H, W):
            orig_resized = cv2.resize(orig_image, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            orig_resized = orig_image
        bottom_patch = orig_resized[safe_top:safe_bottom]
    else:
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

def load_sample(filename):
    img_path = os.path.join(DATA_DIR, filename)
    mask_path = os.path.join(MASK_DIR, filename)
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Dynamically remove watermark from bottom-left
    if img is not None and img.shape[0] >= 350 and img.shape[1] >= 600:
        clean_patch = img[350:, 600:]
        if clean_patch.size > 0:
            h_target = img[350:, :150].shape[0]
            w_target = img[350:, :150].shape[1]
            img[350:, :150] = np.flip(clean_patch, axis=1)[:h_target, :w_target]
            
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    
    if mask is not None and (len(mask.shape) == 2 or mask.shape[2] == 3):
        if len(mask.shape) == 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
        elif mask.shape[2] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2BGRA)
            
    # Dynamically crop top pure black padding (no speckle)
    if img is not None:
        H, W = img.shape[:2]
        mid_img = img[:, int(W*0.15):int(W*0.85)]
        row_max = np.max(mid_img, axis=1)
        non_black_rows = np.where(row_max > 10)[0]
        if len(non_black_rows) > 0:
            first_non_black = non_black_rows[0]
            
            # Safeguard: check topmost retina layer coordinate to never cut into the retina
            if mask is not None:
                is_bg = (mask[:, :, 0] == 0) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 0)
                is_retina = ~is_bg
                if np.any(is_retina):
                    topmost_retina_y = np.min(np.where(is_retina)[0])
                    first_non_black = min(first_non_black, max(0, topmost_retina_y - 15))
                    
            if first_non_black > 0:
                img = img[first_non_black:, :]
                if mask is not None:
                    mask = mask[first_non_black:, :]
                    
    return img, mask

def find_default_center(mask):
    H, W = mask.shape[:2]
    red_mask = (mask[:, :, 0] == 0) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 255)
    thickness = np.sum(red_mask, axis=0)
    
    window_size = 21
    if W > window_size:
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(thickness, kernel, mode='same')
    else:
        smoothed = thickness
        
    fovea_idx = np.argmin(smoothed)
    return float(fovea_idx) / W

def generate_augmented_pair(img, mask, amp, center, width, tilt):
    H, W = img.shape[:2]
    
    # 1. Apply column shift (bending + tilt)
    shifted_img = np.zeros_like(img)
    shifted_mask = np.zeros_like(mask)
    if len(mask.shape) == 3 and mask.shape[2] == 4:
        shifted_mask[:,:,3] = 255
        
    x = np.arange(W)
    center_px = center * W
    width_px = width * W
    
    dy_bend = amp * np.exp(-((x - center_px)**2) / (2 * (width_px**2) + 1e-6))
    dy_tilt = tilt * (x - W/2) / (W/2)
    dy = np.round(dy_bend + dy_tilt).astype(int)
    
    for i in range(W):
        shift = dy[i]
        if shift > 0:
            shifted_img[shift:, i] = img[:-shift, i]
            shifted_mask[shift:, i] = mask[:-shift, i]
        elif shift < 0:
            shifted_img[:shift, i] = img[-shift:, i]
            shifted_mask[:shift, i] = mask[-shift:, i]
        else:
            shifted_img[:, i] = img[:, i]
            shifted_mask[:, i] = mask[:, i]
            
    # Crop the pure black zero-padded bands created by the shift
    is_bg_shifted = (shifted_mask[:,:,0] == 0) & (shifted_mask[:,:,1] == 0) & (shifted_mask[:,:,2] == 0)
    is_ret_shifted = ~is_bg_shifted
    has_ret_shifted = np.any(is_ret_shifted, axis=0)
    
    max_dy = int(np.max(dy))
    min_dy = int(np.min(dy))
    
    if np.any(has_ret_shifted):
        top_y_per_col = np.argmax(is_ret_shifted, axis=0)
        min_top_y = np.min(top_y_per_col[has_ret_shifted])
        top_crop = min(max_dy, max(0, min_top_y - 5))
    else:
        top_crop = max(0, max_dy)
        
    bottom_crop = max(0, -min_dy)
    
    if top_crop + bottom_crop < shifted_img.shape[0]:
        shifted_img = shifted_img[top_crop:shifted_img.shape[0] - bottom_crop, :]
        shifted_mask = shifted_mask[top_crop:shifted_mask.shape[0] - bottom_crop, :]
        
    target_size = (256, 256)
    img_squashed = cv2.resize(shifted_img, target_size, interpolation=cv2.INTER_LINEAR)
    mask_squashed = cv2.resize(shifted_mask, target_size, interpolation=cv2.INTER_NEAREST)
    
    orig_img_256 = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    
    final_img = crop_and_pad_curved(img_squashed, mask_squashed, orig_image=orig_img_256)
    final_mask = crop_and_pad_curved(mask_squashed, mask_squashed)
    
    return final_img, final_mask

def main():
    print("Reading source files...")
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(('.png', '.jpg'))])
    num_samples = len(files)
    print(f"Found {num_samples} samples.")
    
    # Deterministic shuffling for reproducibility
    random.seed(42)
    shuffled_files = files.copy()
    random.shuffle(shuffled_files)
    
    # Split proportions: 80% train, 15% val, 5% test
    n_train = int(0.80 * num_samples)
    n_val = int(0.15 * num_samples)
    
    train_files = shuffled_files[:n_train]
    val_files = shuffled_files[n_train:n_train + n_val]
    test_files = shuffled_files[n_train + n_val:]
    
    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }
    
    for split_name, split_list in splits.items():
        print(f"\nProcessing {split_name} split ({len(split_list)} base files)...")
        img_out_dir = os.path.join(OUTPUT_ROOT, split_name, "images")
        lbl_out_dir = os.path.join(OUTPUT_ROOT, split_name, "labels")
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(lbl_out_dir, exist_ok=True)
        
        for idx, filename in enumerate(split_list):
            base_name, ext = os.path.splitext(filename)
            img, mask = load_sample(filename)
            if img is None or mask is None:
                print(f"Warning: Failed to load {filename}, skipping.")
                continue
                
            # 1. Save preprocessed original image and mask
            target_size = (256, 256)
            img_squashed = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            mask_squashed = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
            
            orig_img = crop_and_pad_curved(img_squashed, mask_squashed, orig_image=img_squashed)
            orig_mask = crop_and_pad_curved(mask_squashed, mask_squashed)
            
            # Save original as {base_name}_orig.png
            orig_img_name = f"{base_name}_orig.png"
            cv2.imwrite(os.path.join(img_out_dir, orig_img_name), orig_img)
            cv2.imwrite(os.path.join(lbl_out_dir, orig_img_name), orig_mask)
            
            # Find default fovea center to guide augmentations
            c_fovea = find_default_center(mask)
            
            # 2. Generate 10 augmented variants
            is_bg = (mask[:, :, 0] == 0) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 0)
            is_retina = ~is_bg
            H_orig, W_orig = mask.shape[:2]
            y_indices, x_indices = np.where(is_retina)
            
            for i in range(1, 11):
                amp, center, width, tilt = 0.0, 0.0, 0.0, 0.0
                for attempt in range(100):
                    amp = random.uniform(40, 150)
                    center = float(np.clip(c_fovea + random.uniform(-0.05, 0.05), 0.10, 0.90))
                    width = random.uniform(0.40, 0.80)
                    tilt = random.uniform(-35, 35)
                    
                    # Verify no retina cutoff before executing full crop/resize pipeline
                    x_arr = np.arange(W_orig)
                    center_px = center * W_orig
                    width_px = width * W_orig
                    dy_bend = amp * np.exp(-((x_arr - center_px)**2) / (2 * (width_px**2) + 1e-6))
                    dy_tilt = tilt * (x_arr - W_orig/2) / (W_orig/2)
                    dy = np.round(dy_bend + dy_tilt).astype(int)
                    
                    if len(y_indices) > 0:
                        shifted_y = y_indices + dy[x_indices]
                        if not (np.any(shifted_y < 0) or np.any(shifted_y >= H_orig)):
                            break
                    else:
                        break
                        
                aug_img, aug_mask = generate_augmented_pair(img, mask, amp, center, width, tilt)
                
                aug_name = f"{base_name}_aug_{i}.png"
                cv2.imwrite(os.path.join(img_out_dir, aug_name), aug_img)
                cv2.imwrite(os.path.join(lbl_out_dir, aug_name), aug_mask)
                
            if (idx + 1) % 10 == 0 or (idx + 1) == len(split_list):
                print(f"  Processed {idx + 1}/{len(split_list)} base files.")
                
    print(f"\nAll splits generated successfully under: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()
