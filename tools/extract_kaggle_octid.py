import os
import glob
import zipfile
import numpy as np
from PIL import Image

base_dir = "/home/mmk/Codes/oct_data_synthesis/DATA"

# 1. Extract Kaggle .npy files
kaggle_dir = os.path.join(base_dir, "Kaggle Retinal Segmentation Dataset")
npy_files = glob.glob(os.path.join(kaggle_dir, "**", "*.npy"), recursive=True)

for npy_file in npy_files:
    filename = os.path.basename(npy_file)
    name_no_ext = os.path.splitext(filename)[0]
    out_dir = os.path.join(os.path.dirname(npy_file), name_no_ext)
    
    if os.path.exists(out_dir):
        print(f"Skipping {filename}, already extracted.")
        continue
        
    os.makedirs(out_dir, exist_ok=True)
    print(f"Extracting {filename} (Kaggle)...")
    
    try:
        arr = np.load(npy_file, allow_pickle=True)
    except Exception as e:
        print(f"Skipping {filename}: {e}")
        continue
    arr = np.squeeze(arr)
    
    slice_axis = -1
    if arr.ndim > 2:
        if arr.shape[0] > arr.shape[-1] and arr.shape[0] > 4:
            slice_axis = 0
            
    num_slices = arr.shape[slice_axis] if arr.ndim > 2 else 1
    
    for i in range(num_slices):
        if arr.ndim > 2:
            if slice_axis == -1 or slice_axis == 2:
                slice_arr = arr[..., i]
            else:
                slice_arr = arr[i, ...]
        else:
            slice_arr = arr
            
        slice_arr = np.nan_to_num(slice_arr)
        ptp = float(np.max(slice_arr)) - float(np.min(slice_arr))
        if ptp > 0:
            slice_arr = (slice_arr - np.min(slice_arr)) * (255.0 / ptp)
        slice_arr = slice_arr.astype(np.uint8)
        
        # Additional normalization if it looks like a mask
        if np.max(slice_arr) < 10 and ptp > 0:
            slice_arr = (slice_arr * (255.0 / np.max(slice_arr))).astype(np.uint8)
            
        img = Image.fromarray(slice_arr)
        img.save(os.path.join(out_dir, f"slice_{i:04d}.png"))
        
    print(f"Finished {filename}")

# 2. Extract OCTID .zip files
octid_dir = os.path.join(base_dir, "OCTID")
zip_files = glob.glob(os.path.join(octid_dir, "**", "*.zip"), recursive=True)

for zf_path in zip_files:
    filename = os.path.basename(zf_path)
    name_no_ext = os.path.splitext(filename)[0]
    out_dir = os.path.join(os.path.dirname(zf_path), name_no_ext)
    
    if os.path.exists(out_dir):
        print(f"Skipping {filename}, already extracted.")
        continue
        
    os.makedirs(out_dir, exist_ok=True)
    print(f"Unzipping {filename} (OCTID)...")
    
    try:
        with zipfile.ZipFile(zf_path, 'r') as zf:
            zf.extractall(out_dir)
        print(f"Finished {filename}")
    except Exception as e:
        print(f"Failed to unzip {filename}: {e}")

print("All extractions completed.")
