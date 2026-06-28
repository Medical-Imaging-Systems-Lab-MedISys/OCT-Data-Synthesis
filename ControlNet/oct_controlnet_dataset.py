import os
import json
import cv2
import numpy as np
from torch.utils.data import Dataset

# =====================================================================
# Helper Functions (Mirrored from Pix2Pix and CFM working code)
# =====================================================================

def sample_gamma_from_bell_curve(min_g, max_g):
    mean = (min_g + max_g) / 2.0
    std = (max_g - min_g) / 6.0
    return np.clip(np.random.normal(mean, std), min_g, max_g)

def apply_gamma(val, g):
    return 255.0 * np.power(val / 255.0, g)

def synthesize_from_mask(mask_bgra, min_gamma=0.5, max_gamma=1.2):
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
    final_image = np.where(keep_mask, padded_img, tiled_bg)
    return final_image

# =====================================================================
# ControlNet Specific OCT Dataset
# =====================================================================

class OCTControlNetDataset(Dataset):
    def __init__(self, labels_dir, real_dir, target_size=256, prompt="high-resolution retinal OCT scan, medical imaging"):
        self.labels_dir = labels_dir
        self.real_dir = real_dir
        self.target_size = target_size
        self.default_prompt = prompt
        
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
        
        # 1. Load Ground Truth Real OCT scan & remove watermark
        x1_img = cv2.imread(real_path, cv2.IMREAD_GRAYSCALE)
        clean_patch = x1_img[350:, 600:]
        x1_img[350:, :150] = np.flip(clean_patch, axis=1)
            
        # 2. Load Segmentation Mask (BGRA layer map)
        mask_bgra = cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED)
        if len(mask_bgra.shape) == 3 and mask_bgra.shape[2] == 3:
            alpha = np.full((mask_bgra.shape[0], mask_bgra.shape[1], 1), 255, dtype=np.uint8)
            mask_bgra = np.concatenate([mask_bgra, alpha], axis=2)
            
        # 3. Dynamically Generate Synthetic Speckle Prior
        x0_img = synthesize_from_mask(mask_bgra, min_gamma=0.5, max_gamma=1.5)
        
        # 4. Strict Squashing and Resizing
        resize_dim = (self.target_size, self.target_size)
        x0_squashed = cv2.resize(x0_img, resize_dim, interpolation=cv2.INTER_LINEAR)
        x1_squashed = cv2.resize(x1_img, resize_dim, interpolation=cv2.INTER_LINEAR)
        mask_squashed = cv2.resize(mask_bgra, resize_dim, interpolation=cv2.INTER_LINEAR)
        
        # 5. Crop and Pad Curved Boundary
        x0_final_gray = crop_and_pad_curved(x0_squashed, mask_squashed)
        x1_final_gray = crop_and_pad_curved(x1_squashed, mask_squashed)
        
        # 6. Convert Grayscale to 3-Channel RGB space for standard ControlNet UNet compatibility
        source_rgb = cv2.cvtColor(x0_final_gray, cv2.COLOR_GRAY2RGB)
        target_rgb = cv2.cvtColor(x1_final_gray, cv2.COLOR_GRAY2RGB)
        
        # 7. Standard Normalizations matching standard control net code structures
        source = source_rgb.astype(np.float32) / 255.0           # Normalize hint to [0, 1]
        target = (target_rgb.astype(np.float32) / 127.5) - 1.0  # Normalize target to [-1, 1]
        
        return dict(jpg=target, txt=self.default_prompt, hint=source)
