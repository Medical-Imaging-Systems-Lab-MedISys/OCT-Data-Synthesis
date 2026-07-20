import os
import glob
import shutil
import numpy as np
from PIL import Image, ImageDraw

data_dir = "/home/mmk/Codes/oct_data_synthesis/DATA/OCTID/Manual-Segmenation/Manual_Segmentation"

extracted_dirs = glob.glob(os.path.join(data_dir, "*_octSegmentation"))

for ex_dir in extracted_dirs:
    dir_name = os.path.basename(ex_dir)
    img_name = dir_name.replace("_octSegmentation", "")
    dest_img = os.path.join(ex_dir, img_name)
    
    if not os.path.exists(dest_img):
        print(f"Original image not found for {dir_name}, skipping drawing.")
        continue
        
    try:
        img = Image.open(dest_img).convert("RGB")
        img_w, img_h = img.size
        
        # Step 1: Collect all raw coordinates across all layers to find global maxes
        all_x_raw = []
        all_y_raw = []
        layer_data = []
        
        for layer_idx in range(7):
            x_file = os.path.join(ex_dir, f"imageLayer_retinalLayers_0_0_pathX_0_{layer_idx}.csv")
            y_file = os.path.join(ex_dir, f"imageLayer_retinalLayers_0_0_pathY_0_{layer_idx}.csv")
            
            if os.path.exists(x_file) and os.path.exists(y_file):
                x_data = np.loadtxt(x_file, delimiter=",")
                y_data = np.loadtxt(y_file, delimiter=",")
                
                # Make sure it's an array
                if x_data.ndim == 0: x_data = np.array([x_data])
                if y_data.ndim == 0: y_data = np.array([y_data])
                
                if x_data.size > 0 and y_data.size > 0 and x_data.size == y_data.size:
                    all_x_raw.extend(x_data[~np.isnan(x_data)])
                    all_y_raw.extend(y_data[~np.isnan(y_data)])
                    layer_data.append((layer_idx, x_data, y_data))
                    
        if not layer_data:
            continue
            
        global_max_x_raw = np.max(all_x_raw)
        global_max_y_raw = np.max(all_y_raw)
        
        # Step 2: Apply rotation right (90 deg clockwise)
        # In MATLAB matrix coords, rotation by 90 right means X_new = max_Y - Y_old, Y_new = X_old
        # We also want to scale to the image size
        all_x_new = global_max_y_raw - np.array(all_y_raw)
        all_y_new = np.array(all_x_raw)
        
        global_max_x_new = np.max(all_x_new)
        global_max_y_new = np.max(all_y_new)
        
        scale_x = img_w / global_max_x_new if global_max_x_new > 0 else 1.0
        scale_y = img_h / global_max_y_new if global_max_y_new > 0 else 1.0
        
        # Draw
        draw = ImageDraw.Draw(img)
        colors = ["red", "lime", "cyan", "yellow", "magenta", "orange", "blue", "purple"]
        
        for layer_idx, x_data, y_data in layer_data:
            points = []
            for x, y in zip(x_data, y_data):
                if not np.isnan(x) and not np.isnan(y):
                    x_new = global_max_y_raw - y
                    y_new = x
                    points.append((int(x_new * scale_x), int(y_new * scale_y)))
            
            if len(points) > 1:
                draw.line(points, fill=colors[layer_idx % len(colors)], width=2)
            elif len(points) == 1:
                draw.point(points[0], fill=colors[layer_idx % len(colors)])
                
        img.save(os.path.join(ex_dir, "segmentation_overlay.png"))
        print(f"Processed {dir_name}")
        
    except Exception as e:
        print(f"Error processing {dir_name}: {e}")

print("Rotation and scaling overlay creation completed.")
