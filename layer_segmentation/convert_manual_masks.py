#!/usr/bin/env /data/vds/env_pt/bin/python
"""
convert_manual_masks.py
=======================
Convert OCT Manual Delineations 2018 .vol + .mat boundary files to
per-B-scan PNG images and pixel masks.

Structure:
  vol/   : *.vol files (Heidelberg Spectralis, read with heyexReader)
  delineation/ : *.mat files — key ``control_pts`` shape (49, 11)
                 49 control points × 11 boundary y-values per B-scan
                 (the 11 boundaries are constant across image columns
                  for each B-scan, so we broadcast to all columns).

Actually ``control_pts`` is interpreted as: for each B-scan index b,
``control_pts[b, :]`` gives the 11 boundary y-coordinates as constant
values across all columns of that B-scan.

Classes (12 total):
    0  – vitreous (above boundary 0)
    1-10 – retinal layers between consecutive boundaries
    11 – choroid/sclera (below boundary 10)

Output:
    processed_images/<vol_stem>/bscan_NNN.png
    processed_masks/<vol_stem>/bscan_NNN.png

Usage:
    python convert_manual_masks.py \
        --data_dir /data/vds/mmk/Codes/oct_data_synthesis/DATA/OCT_Manual_Delineations-2018_June_29
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image

# Add heyexReader to path
HEYEX_PATH = '/data/vds/env_pt/lib/python3.10/site-packages'
if HEYEX_PATH not in sys.path:
    sys.path.insert(0, HEYEX_PATH)

try:
    import heyexReader
except ImportError:
    raise ImportError(
        "heyexReader not found. Expected at "
        "/data/vds/env_pt/lib/python3.10/site-packages/heyexReader/")

try:
    import scipy.io as sio
    from scipy.interpolate import interp1d
except ImportError:
    raise ImportError("scipy is required: pip install scipy")


NUM_BOUNDARIES = 11
NUM_CLASSES = 12   # 0=vitreous, 1-10=layers, 11=choroid


def boundaries_to_mask(img_h, img_w, bnd_ys):
    """
    Generate pixel mask for one B-scan given constant boundary y-values.

    Args:
        img_h, img_w (int): B-scan dimensions.
        bnd_ys (np.ndarray): Shape (11,) — one y-value per boundary,
                             broadcast across all columns.

    Returns:
        mask (np.ndarray): uint8, shape (H, W), values 0-11.
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    # Broadcast: same boundary y-values for every column
    for row in range(img_h):
        n_above = int(np.sum(bnd_ys <= row))
        class_id = min(n_above, NUM_CLASSES - 1)
        mask[row, :] = class_id
    return mask


def boundaries_to_mask_interp(img_h, img_w, control_pts_row):
    boundary_y = []
    col_indices = np.arange(img_w, dtype=np.float64)

    for bnd in control_pts_row:
        if not isinstance(bnd, np.ndarray) or bnd.size == 0:
            if boundary_y:
                boundary_y.append(boundary_y[-1].copy())
            else:
                boundary_y.append(np.zeros(img_w))
            continue

        xs = bnd[:, 0]
        ys = bnd[:, 1]
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]

        interp_y = np.interp(col_indices, xs, ys, left=ys[0], right=ys[-1])
        boundary_y.append(interp_y)

    boundary_y = np.stack(boundary_y, axis=0)  # (11, W)
    boundary_y = np.sort(boundary_y, axis=0)

    rows = np.arange(img_h)[:, None, None]  # (H, 1, 1)
    bys = boundary_y[None, :, :]            # (1, 11, W)
    mask = np.sum(rows >= bys, axis=1, dtype=np.uint8)  # (H, W)
    mask = np.minimum(mask, NUM_CLASSES - 1)
    return mask




def process_vol(vol_path, mat_path, out_img_dir, out_msk_dir, stem):
    """Process one .vol + .mat pair and write PNG files."""
    # Read B-scans from .vol file
    try:
        vol = heyexReader.volFile(vol_path)
    except Exception as e:
        print(f"  [ERROR] Cannot open {vol_path}: {e}")
        return 0

    # Load boundaries
    mat = sio.loadmat(mat_path)
    if 'control_pts' not in mat:
        print(f"  [WARN] 'control_pts' key missing in {mat_path}, skipping.")
        return 0

    control_pts = mat['control_pts']   # expected (49, 11)
    if control_pts.ndim != 2 or control_pts.shape[1] != NUM_BOUNDARIES:
        print(f"  [WARN] Unexpected control_pts shape {control_pts.shape} "
              f"in {mat_path}, skipping.")
        return 0

    n_bscans_mat = control_pts.shape[0]   # 49

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_msk_dir, exist_ok=True)

    saved = 0
    try:
        oct_data = vol.oct  # shape (Z, H, W)
        n_bscans = oct_data.shape[0]
    except Exception as e:
        print(f"  [ERROR] Cannot read B-scans from {vol_path}: {e}")
        return 0

    for b_idx in range(n_bscans):
        if b_idx >= n_bscans_mat:
            break

        img_np = oct_data[b_idx].astype(np.float32)

        # Normalise to uint8
        img_min, img_max = img_np.min(), img_np.max()
        if img_max > img_min:
            img_u8 = ((img_np - img_min) / (img_max - img_min) * 255
                      ).astype(np.uint8)
        else:
            img_u8 = np.zeros_like(img_np, dtype=np.uint8)

        img_h, img_w = img_u8.shape[:2]

        # Get the 11 boundary y-values for this B-scan
        bnd_ys = control_pts[b_idx, :]   # (11,)
        mask = boundaries_to_mask_interp(img_h, img_w, bnd_ys)

        out_img = os.path.join(out_img_dir, f"bscan_{b_idx:03d}.png")
        out_msk = os.path.join(out_msk_dir, f"bscan_{b_idx:03d}.png")

        Image.fromarray(img_u8).save(out_img)
        Image.fromarray(mask).save(out_msk)
        saved += 1

    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Convert Manual Delineations 2018 .vol/.mat to PNG masks")
    parser.add_argument(
        '--data_dir', type=str,
        default=('/data/vds/mmk/Codes/oct_data_synthesis/DATA/'
                 'OCT_Manual_Delineations-2018_June_29'),
        help="Root directory containing vol/ and delineation/ sub-folders")
    args = parser.parse_args()

    vol_dir = os.path.join(args.data_dir, 'vol')
    mat_dir = os.path.join(args.data_dir, 'delineation')
    out_img_root = os.path.join(args.data_dir, 'processed_images')
    out_msk_root = os.path.join(args.data_dir, 'processed_masks')

    vol_files = sorted([f for f in os.listdir(vol_dir)
                        if f.lower().endswith('.vol')])
    print(f"Found {len(vol_files)} .vol files.")

    total_saved = 0
    for vf in vol_files:
        stem = os.path.splitext(vf)[0]
        vol_path = os.path.join(vol_dir, vf)
        mat_path = os.path.join(mat_dir, stem + '.mat')

        if not os.path.isfile(mat_path):
            print(f"  [WARN] No matching .mat for {vf}, skipping.")
            continue

        out_img_dir = os.path.join(out_img_root, stem)
        out_msk_dir = os.path.join(out_msk_root, stem)

        print(f"  Processing {stem} ...", end='', flush=True)
        n = process_vol(vol_path, mat_path, out_img_dir, out_msk_dir, stem)
        print(f" {n} B-scans saved.")
        total_saved += n

    print(f"\nTotal B-scans saved: {total_saved}")
    print(f"Images: {out_img_root}")
    print(f"Masks:  {out_msk_root}")


if __name__ == '__main__':
    main()
