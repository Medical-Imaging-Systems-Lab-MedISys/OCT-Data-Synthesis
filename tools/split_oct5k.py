import os
import glob
import random
import shutil

source_dir = "/home/mmk/Codes/oct_data_synthesis/DATA/OCT5k/Masks/Masks_Automatic/Grading"
output_dir = "/home/mmk/Codes/oct_data_synthesis/DATA/OCT5k_split"

# Find all grey masks
mask_files = glob.glob(os.path.join(source_dir, "**", "*.png"), recursive=True)

if not mask_files:
    print("No mask files found in", source_dir)
    exit()

print(f"Found {len(mask_files)} mask files.")

# Shuffle with a fixed seed for reproducibility
random.seed(42)
random.shuffle(mask_files)

# Calculate splits: 70% train, 15% val, 15% test
total = len(mask_files)
train_end = int(total * 0.7)
val_end = int(total * 0.85)

train_files = mask_files[:train_end]
val_files = mask_files[train_end:val_end]
test_files = mask_files[val_end:]

splits = {
    "train": train_files,
    "val": val_files,
    "test": test_files
}

for split_name, files in splits.items():
    split_mask_dir = os.path.join(output_dir, split_name, "masks")
    os.makedirs(split_mask_dir, exist_ok=True)
    
    print(f"Copying {len(files)} files to {split_mask_dir}...")
    for idx, f in enumerate(files):
        # We need to give them unique names since they are all named 'Image (1).png' etc.
        # Let's use the parent directory structure to create a unique name
        # e.g. "AMD Part1_AMD (10)_Image (1).png"
        parts = f.replace(source_dir, "").strip(os.path.sep).split(os.path.sep)
        unique_name = "_".join(parts)
        
        dest = os.path.join(split_mask_dir, unique_name)
        shutil.copy2(f, dest)

print(f"Dataset successfully split into {output_dir}")
