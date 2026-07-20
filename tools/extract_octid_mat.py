import os
import glob
import numpy as np
import scipy.io as sio
from PIL import Image
import csv

data_dir = "/home/mmk/Codes/oct_data_synthesis/DATA/OCTID/Manual-Segmenation/Manual_Segmentation"
mat_files = glob.glob(os.path.join(data_dir, "*.mat"))

def extract_structured_array(arr, base_dir, prefix=""):
    if arr.dtype.names is not None:
        for name in arr.dtype.names:
            sub_arr = arr[name]
            # sub_arr might be a shape (1, 1) or (1, N) object array
            extract_item(sub_arr, base_dir, prefix=f"{prefix}_{name}" if prefix else name)

def extract_item(item, base_dir, prefix=""):
    # item could be an object array containing other arrays
    if isinstance(item, np.ndarray):
        if item.dtype.names is not None:
            extract_structured_array(item, base_dir, prefix)
        elif item.dtype == 'O':
            for i, elem in np.ndenumerate(item):
                extract_item(elem, base_dir, prefix=f"{prefix}_{i[0]}" if len(i)==1 else f"{prefix}_{i[0]}_{i[1]}")
        else:
            save_array(item, base_dir, prefix)
    elif isinstance(item, (list, tuple)):
        for i, elem in enumerate(item):
            extract_item(elem, base_dir, prefix=f"{prefix}_{i}")

def save_array(arr, base_dir, prefix=""):
    arr = np.squeeze(arr)
    if arr.size == 0:
        return
        
    os.makedirs(base_dir, exist_ok=True)
    
    if arr.ndim >= 2:
        # Save as image
        arr = np.nan_to_num(arr)
        ptp = float(np.max(arr)) - float(np.min(arr))
        if ptp > 0:
            arr = (arr - np.min(arr)) * (255.0 / ptp)
        arr = arr.astype(np.uint8)
        
        if np.max(arr) < 10 and ptp > 0:
            arr = (arr * (255.0 / np.max(arr))).astype(np.uint8)
            
        img = Image.fromarray(arr)
        img.save(os.path.join(base_dir, f"{prefix}.png"))
    elif arr.ndim == 1:
        # Save as CSV
        with open(os.path.join(base_dir, f"{prefix}.csv"), 'w') as f:
            writer = csv.writer(f)
            for val in arr:
                writer.writerow([val])

for mat_file in mat_files:
    filename = os.path.basename(mat_file)
    name_no_ext = os.path.splitext(filename)[0]
    out_dir = os.path.join(data_dir, name_no_ext)
    
    print(f"Extracting {filename}...")
    try:
        mat = sio.loadmat(mat_file)
        for k, v in mat.items():
            if not k.startswith("__"):
                extract_item(v, out_dir, prefix=k)
    except Exception as e:
        print(f"Failed {filename}: {e}")

print("Done.")
