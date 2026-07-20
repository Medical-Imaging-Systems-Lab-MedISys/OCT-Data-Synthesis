#!/usr/bin/env /data/vds/env_pt/bin/python
"""
convert_duke_masks.py
=====================
Convert Duke DME Chiu 2015 .mat files to per-slice PNG images and masks.

Reads ``Subject_NN.mat`` files (N=1..10).
Keys used:
  - ``images``       : shape (496, 768, 61) uint8 — OCT volume
  - ``manualLayers1``: shape (8, 768, 61) float64 — y-coords of 8 boundaries

Generates 9-class pixel masks:
    0 – vitreous (above boundary 0)
    1-7 – retinal layers between consecutive boundaries
    8 – choroid (below boundary 7)

NaN boundary values → column is marked as class 0 throughout.

Output:
    processed_images/Subject_NN/slice_NNN.png
    processed_masks/Subject_NN/slice_NNN.png

Usage:
    python convert_duke_masks.py \
        --data_dir "/data/vds/mmk/Codes/oct_data_synthesis/DATA/Duke DME Chiu 2015"
"""

import os
import argparse
import numpy as np
from PIL import Image

try:
    import scipy.io as sio
except ImportError:
    raise ImportError("scipy is required: pip install scipy")


NUM_BOUNDARIES = 8
NUM_CLASSES = 9   # 0=vitreous, 1-7=layers, 8=choroid


def layers_to_mask(img_h, img_w, boundaries):
    """
    Convert boundary y-coordinates to a pixel mask for a single B-scan column.

    Args:
        img_h (int): Image height.
        img_w (int): Image width.
        boundaries (np.ndarray): Shape (8, W) float64, y-coords per column.
                                 May contain NaN.

    Returns:
        mask (np.ndarray): uint8, shape (H, W), values 0-8.
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    rows = np.arange(img_h, dtype=np.float64)

    for col in range(img_w):
        bys = boundaries[:, col]   # (8,)
        if np.any(np.isnan(bys)):
            # NaN column — leave as class 0 (vitreous)
            continue
        for row in range(img_h):
            # Count boundaries strictly above this row (b <= row)
            n_above = int(np.sum(bys <= row))
            mask[row, col] = min(n_above, NUM_CLASSES - 1)

    return mask


def process_subject(mat_path, out_img_dir, out_msk_dir, subj_name):
    """Load one Subject mat file and write PNG slices."""
    mat = sio.loadmat(mat_path)

    images = mat['images']           # (H=496, W=768, Z=61) uint8
    layers = mat['manualLayers1']    # (8, W=768, Z=61) float64

    img_h, img_w, n_slices = images.shape

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_msk_dir, exist_ok=True)

    for sl in range(n_slices):
        img_slice = images[:, :, sl]         # (H, W) uint8
        bnd_slice = layers[:, :, sl]         # (8, W) float64

        mask = layers_to_mask(img_h, img_w, bnd_slice)

        img_out = os.path.join(out_img_dir, f"slice_{sl:03d}.png")
        msk_out = os.path.join(out_msk_dir, f"slice_{sl:03d}.png")

        Image.fromarray(img_slice).save(img_out)
        Image.fromarray(mask).save(msk_out)

    print(f"  {subj_name}: {n_slices} slices written.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Duke DME Chiu 2015 .mat files to PNG masks")
    parser.add_argument(
        '--data_dir', type=str,
        default='/data/vds/mmk/Codes/oct_data_synthesis/DATA/Duke DME Chiu 2015',
        help="Directory containing Subject_NN.mat files")
    args = parser.parse_args()

    out_img_root = os.path.join(args.data_dir, 'processed_images')
    out_msk_root = os.path.join(args.data_dir, 'processed_masks')

    mat_files = sorted([
        f for f in os.listdir(args.data_dir)
        if f.startswith('Subject_') and f.endswith('.mat')
    ])

    if not mat_files:
        print(f"[ERROR] No Subject_*.mat files found in {args.data_dir}")
        return

    print(f"Found {len(mat_files)} subject mat files.")
    for fname in mat_files:
        subj_name = fname.replace('.mat', '')   # e.g. Subject_01
        mat_path = os.path.join(args.data_dir, fname)
        out_img_dir = os.path.join(out_img_root, subj_name)
        out_msk_dir = os.path.join(out_msk_root, subj_name)
        process_subject(mat_path, out_img_dir, out_msk_dir, subj_name)

    print(f"\nDone. Output in:")
    print(f"  {out_img_root}")
    print(f"  {out_msk_root}")


if __name__ == '__main__':
    main()
