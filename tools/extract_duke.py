import os
import glob
import numpy as np
from PIL import Image
from scipy.io import loadmat

data_dir = "/home/mmk/Codes/oct_data_synthesis/DATA/Duke DME Chiu 2015"
mat_files = glob.glob(os.path.join(data_dir, "*.mat"))

for mat_file in mat_files:
    filename = os.path.basename(mat_file)
    subject_name = os.path.splitext(filename)[0]
    subject_dir = os.path.join(data_dir, subject_name)
    os.makedirs(subject_dir, exist_ok=True)
    
    print(f"Extracting {filename}...")
    try:
        mat = loadmat(mat_file)
    except Exception as e:
        print(f"Failed to load {filename}: {e}")
        continue
        
    for key, val in mat.items():
        if key.startswith("__"):
            continue
        if isinstance(val, np.ndarray) and val.size > 0:
            arr = np.squeeze(val)
            if arr.ndim < 2:
                continue
                
            # Determine slice axis (usually the smallest dimension for medical stacks, or specifically the last dim for Duke)
            if arr.ndim > 2:
                # Duke is typically (H, W, N)
                slice_axis = -1
                if arr.shape[0] < arr.shape[-1] and arr.shape[0] < 100: # heuristic
                    slice_axis = 0
            else:
                # 2D array, just one slice
                arr = np.expand_dims(arr, axis=-1)
                slice_axis = -1
                
            num_slices = arr.shape[slice_axis]
            
            key_dir = os.path.join(subject_dir, key)
            os.makedirs(key_dir, exist_ok=True)
            
            for i in range(num_slices):
                if slice_axis == -1 or slice_axis == 2:
                    slice_arr = arr[..., i]
                else:
                    slice_arr = arr[i, ...]
                    
                slice_arr = np.nan_to_num(slice_arr)
                ptp = float(np.max(slice_arr)) - float(np.min(slice_arr))
                if ptp > 0:
                    slice_arr = (slice_arr - np.min(slice_arr)) * (255.0 / ptp)
                slice_arr = slice_arr.astype(np.uint8)
                
                # If it's a mask, it might need to be explicitly normalized again if ptp is small
                if np.max(slice_arr) < 10 and ptp > 0:
                    slice_arr = (slice_arr * (255.0 / np.max(slice_arr))).astype(np.uint8)
                    
                img = Image.fromarray(slice_arr)
                img.save(os.path.join(key_dir, f"{key}_{i:03d}.png"))
                
    print(f"Finished {subject_name}")
