import os
import glob
import shutil

# Paths
source_images_dir = "/home/mmk/Codes/oct_data_synthesis/DATA/OCT5k/Images"
split_dir = "/home/mmk/Codes/oct_data_synthesis/DATA/OCT5k_split"

def main():
    if not os.path.exists(source_images_dir):
        print(f"Error: Could not find the source images directory at {source_images_dir}")
        print("Please ensure you have downloaded and extracted the raw OCT5k Images folder into DATA/OCT5k/Images")
        return

    # 1. Build a lookup table from all available original images
    # We will map the unique flattened name (which was used for masks) back to the actual image file path.
    all_images = glob.glob(os.path.join(source_images_dir, "**", "*.png"), recursive=True)
    if not all_images:
        print(f"No .png images found inside {source_images_dir}")
        return

    lookup_table = {}
    for img_path in all_images:
        rel_path = os.path.relpath(img_path, source_images_dir)
        unique_name = rel_path.replace(os.path.sep, "_")
        lookup_table[unique_name] = img_path

    print(f"Indexed {len(lookup_table)} original images.")

    # 2. Iterate through the splits and transfer the matching images
    splits = ["train", "test"]
    for split in splits:
        split_mask_dir = os.path.join(split_dir, split, "masks")
        split_image_dir = os.path.join(split_dir, split, "images")
        
        # Always create the images directory so the dataloader doesn't crash on os.listdir()
        os.makedirs(split_image_dir, exist_ok=True)
        
        if not os.path.exists(split_mask_dir):
            continue
            
        masks_in_split = os.listdir(split_mask_dir)
        count = 0
        missing = 0
        
        for mask_name in masks_in_split:
            if mask_name in lookup_table:
                src_img = lookup_table[mask_name]
                dest_img = os.path.join(split_image_dir, mask_name)
                shutil.copy2(src_img, dest_img)
                count += 1
            else:
                missing += 1
                
        print(f"[{split.upper()}] Copied {count} matching images. ({missing} missing)")

    print("Image segregation complete!")

if __name__ == "__main__":
    main()
