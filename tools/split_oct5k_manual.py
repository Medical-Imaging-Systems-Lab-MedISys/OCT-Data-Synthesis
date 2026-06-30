import os
import glob
import random
import shutil

source_dir = "/home/mmk/Codes/oct_data_synthesis/DATA/OCT5k/Masks/Masks_Manual"
output_dir = "/home/mmk/Codes/oct_data_synthesis/DATA/OCT5k_split"

# Find all grey manual masks across all gradings
mask_files = glob.glob(os.path.join(source_dir, "**", "*.png"), recursive=True)

if not mask_files:
    print("No manual mask files found in", source_dir)
    exit()

# Extract the unique relative paths to prevent data leakage (a single image's gradings must all go to the same split)
# The structure is Masks_Manual/Grading_X/{relative_path}
unique_rel_paths = set()
for f in mask_files:
    rel_to_source = os.path.relpath(f, source_dir)
    # rel_to_source looks like "Grading_1/AMD Part1/..."
    parts = rel_to_source.split(os.path.sep)
    # The first part is the Grading folder, the rest is the unique image path
    rel_path = os.path.sep.join(parts[1:])
    unique_rel_paths.add(rel_path)

unique_rel_paths = sorted(list(unique_rel_paths))
print(f"Found {len(unique_rel_paths)} unique manual mask images (across multiple gradings).")

# Shuffle with a fixed seed
random.seed(42)
random.shuffle(unique_rel_paths)

# 70/15/15 split
total = len(unique_rel_paths)
train_end = int(total * 0.7)
val_end = int(total * 0.85)

splits = {
    "train": unique_rel_paths[:train_end],
    "val": unique_rel_paths[train_end:val_end],
    "test": unique_rel_paths[val_end:]
}

gradings = ["Grading_1", "Grading_2", "Grading_3"]

for split_name, rel_paths in splits.items():
    split_mask_dir = os.path.join(output_dir, split_name, "manual_masks")
    os.makedirs(split_mask_dir, exist_ok=True)
    
    count = 0
    for rel_path in rel_paths:
        unique_name = rel_path.replace(os.path.sep, "_")
        
        for grading in gradings:
            src_file = os.path.join(source_dir, grading, rel_path)
            if os.path.exists(src_file):
                dest_file = os.path.join(split_mask_dir, f"{grading}__{unique_name}")
                shutil.copy2(src_file, dest_file)
                count += 1
                
    print(f"Copied {count} files to {split_mask_dir}")

print("Manual dataset successfully segregated.")
