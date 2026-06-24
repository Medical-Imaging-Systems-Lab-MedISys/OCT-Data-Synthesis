#!/usr/bin/env python
import os
import argparse
import numpy as np
import cv2

# Baseline layer parameters (fitted from NR206)
LAYERS_CFG = [
    { 'name': 'Red',         'a': 0.000163,  'b': -0.1227, 'c': 137.8, 'd': 34.7, 'w': 43.5,  'meanInt': 165.5, 'min_g': 0.85, 'max_g': 1.15, 'color': [0, 0, 255, 255] },     # BGRA Red
    { 'name': 'Olive',       'a': 0.000130,  'b': -0.1091, 'c': 153.8, 'd': 19.2, 'w': 41.8,  'meanInt': 129.1, 'min_g': 0.90, 'max_g': 1.10, 'color': [0, 128, 128, 255] }, # BGRA Olive
    { 'name': 'Yellow',      'a': 0.000070,  'b': -0.0702, 'c': 166.3, 'd': 2.1,  'w': 10.9,  'meanInt': 107.4, 'min_g': 0.90, 'max_g': 1.10, 'color': [0, 255, 255, 255] }, # BGRA Yellow
    { 'name': 'DarkGreen',   'a': 0.000042,  'b': -0.0491, 'c': 171.6, 'd': -7.3, 'w': 40.5,  'meanInt': 123.3, 'min_g': 0.90, 'max_g': 1.10, 'color': [0, 128, 0, 255] },   # BGRA Dark Green
    { 'name': 'BrightGreen', 'a': -0.000000, 'b': -0.0200, 'c': 179.8, 'd': -7.5, 'w': 34.7,  'meanInt': 85.9, 'min_g': 0.95, 'max_g': 1.05,  'color': [0, 255, 0, 255] },   # BGRA Bright Green
    { 'name': 'Cyan',        'a': -0.000033, 'b': 0.0014,  'c': 189.1, 'd': -3.4, 'w': 26.6,  'meanInt': 99.7, 'min_g': 0.90, 'max_g': 1.10, 'color': [255, 255, 0, 255] }, # BGRA Cyan
    { 'name': 'Blue',        'a': -0.000039, 'b': 0.0054,  'c': 194.0, 'd': -3.2, 'w': 29.4,  'meanInt': 235.0, 'min_g': 0.85, 'max_g': 1.15, 'color': [255, 0, 0, 255] },   # BGRA Blue
    { 'name': 'Magenta',     'a': -0.000042, 'b': 0.0070,  'c': 201.4, 'd': -1.1, 'w': 18.2,  'meanInt': 193.0, 'min_g': 0.85, 'max_g': 1.15,  'color': [255, 0, 255, 255] }  # BGRA Magenta
]

def sample_gamma_from_bell_curve(min_g, max_g):
    """
    Samples a gamma value from a normal (bell-curve) distribution
    centered between min_g and max_g, and truncated to those bounds.
    """
    mean = (min_g + max_g) / 2.0
    # Standard deviation covering 99.7% of values within [min_g, max_g]
    std = (max_g - min_g) / 6.0
    val = np.random.normal(mean, std)
    return np.clip(val, min_g, max_g)

def apply_gamma(val, g):
    return 255.0 * np.power(val / 255.0, g)

def generate_oct_sample(width, height, min_gamma, max_gamma):
    # 1. Randomize spatial parameters for image-to-image variance
    vertical_offset = np.random.uniform(-13.0, 13.0)
    slope_scale = np.random.uniform(0.7, 1.3)
    fovea_x = np.random.uniform(345.0, 405.0)
    dip_scale = np.random.uniform(0.75, 1.25)
    dip_width_scale = np.random.uniform(0.8, 1.25)
    thickness_scale = np.random.uniform(0.95, 1.1)

    # 2. Randomize gamma parameters for layer-to-layer variance
    layer_gammas = [sample_gamma_from_bell_curve(cfg['min_g'], cfg['max_g']) for cfg in LAYERS_CFG]
    bg_gamma = sample_gamma_from_bell_curve(min_gamma, max_gamma)

    # Pre-calculate layer boundaries for each x coordinate
    x_indices = np.arange(width)
    
    # Calculate top of ILM / Red layer
    red_cfg = LAYERS_CFG[0]
    red_c = red_cfg['c'] - 7.5 + vertical_offset
    red_b = red_cfg['b'] * slope_scale
    red_d = red_cfg['d'] * dip_scale
    red_w = red_cfg['w'] * dip_width_scale
    dip_l1 = red_d * np.exp(-np.pow(x_indices - fovea_x, 2) / (2 * np.pow(red_w, 2)))
    y_top = red_cfg['a'] * np.pow(x_indices, 2) + red_b * x_indices + red_c + dip_l1

    # Thickness equations (replicated from Javascript model)
    t1 = (3 + 12 * np.exp(-x_indices / 100) - 1.5 * np.exp(-np.pow(x_indices - fovea_x, 2) / (2 * np.pow(40, 2)))) * thickness_scale
    
    t2_base = 16 + 14 * np.exp(-np.pow(x_indices - 220, 2) / (2 * np.pow(150, 2))) + 10 * np.exp(-np.pow(x_indices - 500, 2) / (2 * np.pow(150, 2)))
    t2_dip = 28 * np.exp(-np.pow(x_indices - fovea_x, 2) / (2 * np.pow(30, 2)))
    t2 = np.maximum(1.0, t2_base - t2_dip) * thickness_scale
    
    t3 = (7.0 - 5.0 * np.exp(-np.pow(x_indices - fovea_x, 2) / (2 * np.pow(35, 2)))) * thickness_scale
    
    t4 = (7.5 + 4.5 * np.exp(-np.pow(x_indices - 480, 2) / (2 * np.pow(60, 2))) - 5.5 * np.exp(-np.pow(x_indices - fovea_x, 2) / (2 * np.pow(30, 2)))) * thickness_scale
    
    t5 = (11.0 + 17.0 * np.exp(-np.pow(x_indices - fovea_x, 2) / (2 * np.pow(70, 2)))) * thickness_scale
    
    t6 = np.full(width, 8.0) * thickness_scale
    t7 = np.full(width, 5.0) * thickness_scale
    
    t8 = (12.0 + 2.0 * np.exp(-np.pow(x_indices - fovea_x, 2) / (2 * np.pow(150, 2)))) * thickness_scale

    # Compute stacked boundary lines
    b0 = y_top
    b1 = b0 + np.maximum(1.0, t1)
    b2 = b1 + np.maximum(1.0, t2)
    b3 = b2 + np.maximum(1.0, t3)
    b4 = b3 + np.maximum(1.0, t4)
    b5 = b4 + np.maximum(1.0, t5)
    b6 = b5 + np.maximum(1.0, t6)
    b7 = b6 + np.maximum(1.0, t7)
    b8 = b7 + np.maximum(1.0, t8)

    # Initialize empty canvas arrays
    raw_img = np.zeros((height, width), dtype=np.float32)
    label_mask = np.zeros((height, width, 4), dtype=np.uint8) # BGRA format

    # Grid of Y indexes
    y_coords = np.arange(height)[:, None] # (height, 1)

    # Organic micro-texture along columns
    layer_texture = (np.sin(x_indices * 0.05) * 3 + np.cos(x_indices * 0.02) * 2)[None, :] # (1, width)

    # 3. Vectorized rendering of layers
    # Vitreous humor (top background) - Set to pure black
    vitreous_mask = y_coords < b0
    raw_img[vitreous_mask] = 48.0
    label_mask[vitreous_mask] = [0, 0, 0, 255] # Black

    # Iterate through 8 layers
    bounds = [b0, b1, b2, b3, b4, b5, b6, b7, b8]
    for i in range(8):
        mask = (y_coords >= bounds[i]) & (y_coords < bounds[i+1])
        base_int = LAYERS_CFG[i]['meanInt'] + layer_texture
        base_int_full = np.broadcast_to(base_int, (height, width))
        raw_img[mask] = apply_gamma(base_int_full, layer_gammas[i])[mask]
        label_mask[mask] = LAYERS_CFG[i]['color']

    # Sclera / deep background - Set to pure black
    sclera_mask = y_coords >= b8
    raw_img[sclera_mask] = 48.0
    label_mask[sclera_mask] = [0, 0, 0, 255] # Black

    # 4. No Speckle Noise - simply clamp and convert to uint8
    final_img = np.clip(raw_img, 0, 255).astype(np.uint8)

    return final_img, label_mask

def main():
    parser = argparse.ArgumentParser(description="Batch Generate Clean Synthetic OCT Images and Labels")
    parser.add_argument("--count", type=int, default=10, help="Number of images to generate")
    parser.add_argument("--min-gamma", type=float, default=0.5, help="Minimum gamma value")
    parser.add_argument("--max-gamma", type=float, default=1.2, help="Maximum gamma value")
    parser.add_argument("--output-dir", type=str, default="synthetic_dataset_clean", help="Output root directory")
    
    args = parser.parse_args()

    # Create directories
    img_dir = os.path.join(args.output_dir, "images")
    lbl_dir = os.path.join(args.output_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    print(f"Generating {args.count} Clean OCT images (No Speckle Noise)...")
    print(f"Output directories:\n  - Raw Images:  {img_dir}\n  - Label Masks: {lbl_dir}")
    print(f"Gamma range (bell-curve): {args.min_gamma} to {args.max_gamma}\n")

    for idx in range(1, args.count + 1):
        filename = f"synthetic_{idx}.png"
        img, label = generate_oct_sample(width=750, height=500, min_gamma=args.min_gamma, max_gamma=args.max_gamma)
        
        cv2.imwrite(os.path.join(img_dir, filename), img)
        cv2.imwrite(os.path.join(lbl_dir, filename), label)
        
        if idx % 10 == 0 or idx == args.count:
            print(f"Generated {idx}/{args.count} images...")

    print("\nGeneration completed successfully!")

if __name__ == "__main__":
    main()
