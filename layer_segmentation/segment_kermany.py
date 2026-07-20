import os
import sys
import argparse
import glob
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import mlflow
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from model import RETFoundSegmenter

# BGR colors for NR206 colorization
NR206_COLORS = [
    [0, 0, 0],       # 0: Background
    [0, 0, 255],     # 1: Red (ILM)
    [0, 128, 128],   # 2: Olive (NFL)
    [0, 255, 255],   # 3: Yellow (IPL/INL)
    [0, 128, 0],     # 4: DarkGreen (OPL)
    [0, 255, 0],     # 5: BrightGreen (ONL)
    [255, 255, 0],   # 6: Cyan (ELM/IS)
    [255, 0, 0],     # 7: Blue (OS/RPE)
    [255, 0, 255]    # 8: Magenta (RPE/Chor)
]

def colorize_mask_nr206(mask_np):
    h, w = mask_np.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for i, c in enumerate(NR206_COLORS):
        # Convert BGR back to RGB for matplotlib/PIL grid representation
        rgb_color = [c[2], c[1], c[0]]
        color_mask[mask_np == i] = rgb_color
    return color_mask

def colorize_mask_oct5k(mask_np):
    # OCT5k only has 6 classes (0 to 5)
    h, w = mask_np.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for i, c in enumerate(NR206_COLORS[:6]):
        rgb_color = [c[2], c[1], c[0]]
        color_mask[mask_np == i] = rgb_color
    return color_mask

def preprocess_image(image_path, img_size=256):
    img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img.size
    
    # Apply normalization matching PyTorch/RETFound
    # Resize, convert to tensor, normalize
    img_resized = img.resize((img_size, img_size))
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_np - mean) / std
    img_tensor = torch.as_tensor(img_normalized).permute(2, 0, 1).float()
    return img_tensor, orig_w, orig_h

def main():
    parser = argparse.ArgumentParser(description="Segment Kermany 2018 OCT Images using trained RETFound models")
    parser.add_argument('--kermany_dir', type=str, required=True, help="Path to Kermany 2018 dataset root")
    parser.add_argument('--run_id', type=str, required=True, help="MLflow Run ID of the trained model checkpoints")
    parser.add_argument('--model_type', type=str, required=True, choices=['nr206', 'oct5k'], help="Which model to use")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save predicted masks")
    parser.add_argument('--run_name', type=str, required=True, help="MLflow run name")
    parser.add_argument('--img_size', type=int, default=256, help="Image size expected by model")
    parser.add_argument('--gpu', type=str, default=None, help="GPU index to bind process")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"Force bound to GPU: {args.gpu}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set up MLflow
    mlflow.set_tracking_uri('http://10.24.38.15:5000')
    mlflow.set_experiment('Kermany2018_Segmentation')

    # 1. Download Model Checkpoint from MLflow
    client = mlflow.tracking.MlflowClient()
    print(f"Querying checkpoints in run {args.run_id}...")
    try:
        artifacts = client.list_artifacts(args.run_id, "checkpoints")
        pth_files = [art.path for art in artifacts if art.path.endswith('.pth')]
        if not pth_files:
            raise RuntimeError("No .pth files found in checkpoints/")
        checkpoint_path = pth_files[0]
        print(f"Downloading checkpoint: {checkpoint_path}")
        local_ckpt_path = client.download_artifacts(args.run_id, checkpoint_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

    # 2. Load Model
    num_classes = 9 if args.model_type == 'nr206' else 6
    model = RETFoundSegmenter(num_classes=num_classes, img_size=args.img_size)
    model.load_state_dict(torch.load(local_ckpt_path, map_location='cpu'))
    model = model.to(device)
    model.eval()
    print("Loaded model weights successfully.")

    # 3. Scan for Images recursively in Kermany test split
    # Standard split paths: kermany_dir/test/**/* (.jpeg or .jpg or .png)
    search_pattern = os.path.join(args.kermany_dir, "test", "**", "*.jpeg")
    images_list = glob.glob(search_pattern, recursive=True)
    
    # If test folder is empty or not matching, search all subfolders
    if not images_list:
        search_pattern = os.path.join(args.kermany_dir, "**", "*.jpeg")
        images_list = glob.glob(search_pattern, recursive=True)

    if not images_list:
        print(f"No .jpeg images found in {args.kermany_dir}. Exiting.")
        sys.exit(1)

    print(f"Found {len(images_list)} images to segment.")

    # Select 10 index indices for visual grid logging
    grid_indices = np.linspace(0, len(images_list) - 1, min(10, len(images_list)), dtype=int)
    sample_grids = []

    with mlflow.start_run(run_name=args.run_name):
        mlflow.set_tag("source_model", args.model_type)
        mlflow.set_tag("dataset", "kermany2018")
        
        with torch.no_grad():
            for idx, img_path in enumerate(tqdm(images_list, desc="Segmenting")):
                # Preprocess image
                img_tensor, orig_w, orig_h = preprocess_image(img_path, args.img_size)
                
                # Inference
                outputs = model(img_tensor.unsqueeze(0).to(device))
                preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                
                # Resize predictions back to original shape
                preds_resized = cv2.resize(preds, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                
                # Save mask file preserving relative directory tree (category/filename)
                rel_path = os.path.relpath(img_path, args.kermany_dir)
                out_path = os.path.join(args.output_dir, os.path.splitext(rel_path)[0] + ".png")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                
                if args.model_type == 'nr206':
                    # Save as BGRA PNG matching NR206 colormap
                    color_mask_bgr = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
                    for i, c in enumerate(NR206_COLORS):
                        color_mask_bgr[preds_resized == i] = c
                    alpha = np.full((orig_h, orig_w, 1), 255, dtype=np.uint8)
                    color_mask_bgra = np.concatenate([color_mask_bgr, alpha], axis=2)
                    cv2.imwrite(out_path, color_mask_bgra)
                else:
                    # Save as grayscale mask 0-5 for OCT5k
                    cv2.imwrite(out_path, preds_resized)

                # Sample for visual grid
                if idx in grid_indices:
                    # Convert original input image to normalized float numpy
                    img_pil = Image.open(img_path).convert('RGB').resize((args.img_size, args.img_size))
                    img_np = np.array(img_pil).astype(np.float32) / 255.0
                    
                    if args.model_type == 'nr206':
                        pred_color = colorize_mask_nr206(preds)
                    else:
                        pred_color = colorize_mask_oct5k(preds)
                        
                    sample_grids.append((img_np, pred_color, os.path.basename(img_path)))

        # Plot and save sample grid
        fig, axes = plt.subplots(len(sample_grids), 2, figsize=(8, 3.5 * len(sample_grids)))
        for i, (img, pred, name) in enumerate(sample_grids):
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"Input: {name}")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(pred)
            axes[i, 1].set_title("Predicted Mask")
            axes[i, 1].axis('off')
            
        plt.tight_layout()
        grid_path = f"kermany_eval_{args.run_name}_grid.png"
        plt.savefig(grid_path)
        plt.close()
        
        mlflow.log_artifact(grid_path, artifact_path="evaluation_grids")
        if os.path.exists(grid_path):
            os.remove(grid_path)

        print("Segmentation complete! Predicted masks saved and visual grid logged to MLflow.")

if __name__ == '__main__':
    main()
