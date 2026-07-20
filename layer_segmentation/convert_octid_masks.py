#!/usr/bin/env /data/vds/env_pt/bin/python
"""
convert_octid_masks.py
======================
Convert OCTID boundary CSV files to per-subject pixel mask PNGs.

Usage:
    python convert_octid_masks.py \
        --data_dir /data/vds/mmk/Codes/oct_data_synthesis/DATA/OCTID

Output:
    processed_images/<name>.png   — grayscale image saved as PNG
    processed_masks/<name>.png    — uint8 mask, values 0-7
"""

import os
import sys
import csv
import glob
import argparse
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Boundary helpers
# ---------------------------------------------------------------------------

def load_boundary_csv(csv_path):
    """Return (x_arr, y_arr) float arrays from a boundary CSV."""
    xs, ys = [], []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                try:
                    xs.append(float(row[0]))
                    ys.append(float(row[1]))
                except ValueError:
                    continue   # skip header
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def boundaries_to_mask(img_h, img_w, boundary_csvs):
    """
    Convert 7 boundary curves to a uint8 pixel mask.

    Classes:
        0 – vitreous (above boundary 0)
        1 – between boundary 0 and 1
        ...
        6 – between boundary 5 and 6
        7 – choroid (below boundary 6)

    Args:
        img_h, img_w (int): Image dimensions.
        boundary_csvs (list[str]): Ordered list of 7 CSV paths (b0..b6).

    Returns:
        mask (np.ndarray): uint8 array, shape (H, W), values 0-7.
    """
    col_indices = np.arange(img_w, dtype=np.float64)

    boundary_y = []
    for csv_path in boundary_csvs:
        xs, ys = load_boundary_csv(csv_path)
        if len(xs) < 2:
            # Degenerate: constant at image mid
            boundary_y.append(np.full(img_w, img_h / 2.0))
            continue
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]
        interp_y = np.interp(col_indices, xs, ys,
                             left=ys[0], right=ys[-1])
        boundary_y.append(interp_y)

    boundary_y = np.stack(boundary_y, axis=0)   # (7, W)

    rows = np.arange(img_h)[:, None, None]  # (H, 1, 1)
    bys = boundary_y[None, :, :]            # (1, 7, W)
    mask = np.sum(rows >= bys, axis=1, dtype=np.uint8)  # (H, W)
    mask = np.minimum(mask, 7)
    return mask



# ---------------------------------------------------------------------------
# Subject discovery
# ---------------------------------------------------------------------------

def find_octid_subjects(data_dir):
    """
    Walk Manual-Segmenation/Manual_Segmentation looking for
    <name>_octSegmentation/ folders.

    Returns list of dicts:
        { 'name', 'img_path', 'csv_paths' (7 ordered paths) }
    """
    base = os.path.join(data_dir,
                        'Manual-Segmenation',
                        'Manual_Segmentation')
    if not os.path.isdir(base):
        # Try alternate spelling without the extra 'a'
        alt = os.path.join(data_dir,
                           'Manual-Segmentation',
                           'Manual_Segmentation')
        if os.path.isdir(alt):
            base = alt
        else:
            raise FileNotFoundError(
                f"Cannot find Manual Segmentation folder in {data_dir}. "
                f"Tried:\n  {base}\n  {alt}")

    subjects = []
    for entry in sorted(os.listdir(base)):
        entry_path = os.path.join(base, entry)
        if not (os.path.isdir(entry_path) and
                entry.endswith('_octSegmentation')):
            continue
        name = entry.replace('_octSegmentation', '')

        # JPEG image
        jpg_candidates = (glob.glob(os.path.join(entry_path, name + '.jpeg'))
                          + glob.glob(os.path.join(entry_path, name + '.jpg')))
        if not jpg_candidates:
            jpg_candidates = (glob.glob(os.path.join(entry_path, '*.jpeg'))
                              + glob.glob(os.path.join(entry_path, '*.jpg')))
        if not jpg_candidates:
            print(f"  [WARN] No JPEG found in {entry_path}, skipping.")
            continue
        img_path = sorted(jpg_candidates)[0]

        # Boundary CSVs: *path_0_0.csv through *path_0_6.csv
        csvs = []
        for b in range(7):
            pattern = os.path.join(entry_path, f'*path_0_{b}.csv')
            matches = glob.glob(pattern)
            if not matches:
                print(f"  [WARN] Missing boundary {b} for {name}, skipping.")
                break
            csvs.append(sorted(matches)[0])

        if len(csvs) < 7:
            continue

        subjects.append({
            'name': name,
            'img_path': img_path,
            'csv_paths': csvs,
        })

    return subjects


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert OCTID boundary CSVs to pixel mask PNGs")
    parser.add_argument(
        '--data_dir', type=str,
        default='/data/vds/mmk/Codes/oct_data_synthesis/DATA/OCTID',
        help="Root OCTID data directory")
    args = parser.parse_args()

    out_img_dir = os.path.join(args.data_dir, 'processed_images')
    out_msk_dir = os.path.join(args.data_dir, 'processed_masks')
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_msk_dir, exist_ok=True)

    subjects = find_octid_subjects(args.data_dir)
    print(f"Found {len(subjects)} OCTID subjects.")

    for subj in subjects:
        name = subj['name']
        img_path = subj['img_path']
        csvs = subj['csv_paths']

        # Load image
        img = Image.open(img_path).convert('L')
        img_w, img_h = img.size   # PIL: (width, height)

        print(f"  Processing {name}  ({img_w}×{img_h}) ...", end='', flush=True)

        # Generate mask
        mask = boundaries_to_mask(img_h, img_w, csvs)

        # Save outputs
        out_img = os.path.join(out_img_dir, f"{name}.png")
        out_msk = os.path.join(out_msk_dir, f"{name}.png")

        img.save(out_img)
        Image.fromarray(mask).save(out_msk)
        print(" done.")

    print(f"\nSaved {len(subjects)} image+mask pairs to:")
    print(f"  {out_img_dir}")
    print(f"  {out_msk_dir}")


if __name__ == '__main__':
    main()
