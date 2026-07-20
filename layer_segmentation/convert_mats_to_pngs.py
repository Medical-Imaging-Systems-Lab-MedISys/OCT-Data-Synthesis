import os
import glob
import numpy as np
import scipy.io as sio
import cv2
from tqdm import tqdm

def process_mat_volume(mat_path, out_dir):
    try:
        mat_data = sio.loadmat(mat_path)
    except Exception as e:
        print(f"Error loading {mat_path}: {e}")
        return
        
    if 'images' in mat_data:
        vol = mat_data['images']
        # Typically shape is (height, width, num_slices)
        if vol.ndim == 3:
            h, w, slices = vol.shape
            base_name = os.path.splitext(os.path.basename(mat_path))[0]
            os.makedirs(out_dir, exist_ok=True)
            for i in range(slices):
                slice_img = vol[:, :, i]
                # Normalize to 0-255 if necessary
                if slice_img.max() <= 1.0:
                    slice_img = (slice_img * 255.0)
                slice_img = slice_img.astype(np.uint8)
                out_path = os.path.join(out_dir, f"{base_name}_slice_{i:03d}.png")
                cv2.imwrite(out_path, slice_img)
    elif 'I2' in mat_data: # kmader dataset
        img = mat_data['I2']
        if img.max() <= 1.0:
            img = (img * 255.0)
        img = img.astype(np.uint8)
        os.makedirs(out_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(mat_path))[0]
        out_path = os.path.join(out_dir, f"{base_name}.png")
        cv2.imwrite(out_path, img)
    else:
        # Fallback to search for largest 2D or 3D array
        largest_arr = None
        largest_size = 0
        for k, v in mat_data.items():
            if not k.startswith('__') and isinstance(v, np.ndarray):
                if v.ndim >= 2 and v.size > largest_size:
                    largest_arr = v
                    largest_size = v.size
        
        if largest_arr is not None:
            if largest_arr.ndim == 3:
                h, w, slices = largest_arr.shape
                base_name = os.path.splitext(os.path.basename(mat_path))[0]
                os.makedirs(out_dir, exist_ok=True)
                for i in range(slices):
                    slice_img = largest_arr[:, :, i]
                    if slice_img.max() <= 1.0:
                        slice_img = slice_img * 255.0
                    elif slice_img.max() <= 2.0:
                        slice_img = (slice_img / slice_img.max()) * 255.0
                    slice_img = slice_img.astype(np.uint8)
                    out_path = os.path.join(out_dir, f"{base_name}_slice_{i:03d}.png")
                    cv2.imwrite(out_path, slice_img)
            elif largest_arr.ndim == 2:
                img = largest_arr
                if img.max() <= 1.0:
                    img = img * 255.0
                elif img.max() <= 2.0:
                    img = (img / img.max()) * 255.0
                img = img.astype(np.uint8)
                os.makedirs(out_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(mat_path))[0]
                out_path = os.path.join(out_dir, f"{base_name}.png")
                cv2.imwrite(out_path, img)

def main():
    base_dir = "/data/vds/mmk/Codes/oct_data_synthesis/DATA"
    datasets = [
        "2011_IOVS_Chiu",
        "2015_BOE_Chiu2",
        "kmader_eye_oct/heiderlberg_oct"
    ]
    
    for ds in datasets:
        ds_path = os.path.join(base_dir, ds)
        mat_files = []
        for root, _, files in os.walk(ds_path):
            if "png_extracted" in root:
                continue
            for f in files:
                if f.endswith('.mat'):
                    mat_files.append(os.path.join(root, f))
                    
        print(f"Found {len(mat_files)} MAT files in {ds_path}")
        for mat_path in tqdm(mat_files, desc=f"Converting {ds}"):
            rel_dir = os.path.dirname(os.path.relpath(mat_path, ds_path))
            out_dir = os.path.join(ds_path, "png_extracted", rel_dir)
            process_mat_volume(mat_path, out_dir)

if __name__ == "__main__":
    main()
