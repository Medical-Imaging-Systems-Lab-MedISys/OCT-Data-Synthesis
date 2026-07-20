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
try:
    import heyexReader
except ImportError:
    heyexReader = None

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
        rgb_color = [c[2], c[1], c[0]]
        color_mask[mask_np == i] = rgb_color
    return color_mask

def colorize_mask_oct5k(mask_np):
    h, w = mask_np.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for i, c in enumerate(NR206_COLORS[:6]):
        rgb_color = [c[2], c[1], c[0]]
        color_mask[mask_np == i] = rgb_color
    return color_mask

def preprocess_image(img_pil, img_size=256):
    orig_w, orig_h = img_pil.size
    img_resized = img_pil.resize((img_size, img_size))
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_np - mean) / std
    img_tensor = torch.as_tensor(img_normalized).permute(2, 0, 1).float()
    return img_tensor, orig_w, orig_h

def main():
    parser = argparse.ArgumentParser(description="Segment New Custom OCT Datasets using trained RETFound models")
    parser.add_argument('--dataset_dir', type=str, required=True, help="Path to dataset root folder")
    parser.add_argument('--dataset_type', type=str, required=True, choices=['c8', 'ucsd', 'manual_delineations'], help="Dataset format type")
    parser.add_argument('--run_id', type=str, required=True, help="MLflow Run ID of the trained model checkpoint")
    parser.add_argument('--model_type', type=str, required=True, choices=['nr206', 'oct5k'], help="Which model to use")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save predicted masks")
    parser.add_argument('--run_name', type=str, required=True, help="MLflow run name")
    parser.add_argument('--img_size', type=int, default=256, help="Image size expected by model")
    parser.add_argument('--gpu', type=str, default=None, help="GPU index to bind process")
    parser.add_argument('--limit', type=int, default=-1, help="Maximum number of images to segment (-1 to segment all)")
    parser.add_argument('--gt_dir', type=str, default=None, help="Optional directory containing GT masks for evaluation grid")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"Force bound to GPU: {args.gpu}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set up MLflow
    mlflow.set_tracking_uri('http://10.24.38.15:5000')
    mlflow.set_experiment('New_Datasets_Segmentation')

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

    # 3. Gather records/images based on dataset_type
    records = []
    if args.dataset_type in ['c8', 'ucsd']:
        valid_exts = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}
        for root, dirs, files in os.walk(args.dataset_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in valid_exts:
                    records.append(os.path.join(root, file))
    elif args.dataset_type == 'manual_delineations':
        # Scan for .vol volume files
        for root, dirs, files in os.walk(args.dataset_dir):
            for file in files:
                if file.lower().endswith('.vol'):
                    records.append(os.path.join(root, file))
        if not records:
            # Fallback to scan vol subfolder
            records = glob.glob(os.path.join(args.dataset_dir, "vol", "*.vol"))

    if not records:
        print(f"No records found for dataset type {args.dataset_type} in {args.dataset_dir}. Exiting.")
        sys.exit(1)

    print(f"Found {len(records)} records/volumes to process.")
    
    if args.limit != -1:
        records = records[:args.limit]
        print(f"Limit specified: segmenting first {len(records)} records.")

    # Select 10 indices for visual grid logging
    grid_indices = np.linspace(0, len(records) - 1, min(10, len(records)), dtype=int)
    sample_grids = []

    with mlflow.start_run(run_name=args.run_name):
        mlflow.set_tag("source_model", args.model_type)
        mlflow.set_tag("dataset_type", args.dataset_type)
        
        with torch.no_grad():
            for record_idx, rec in enumerate(tqdm(records, desc="Processing records")):
                if args.dataset_type == 'manual_delineations':
                    # Parse Heidelberg .vol volume file
                    if heyexReader is None:
                        raise RuntimeError("heyexReader library is not installed!")
                    vol = heyexReader.volFile(rec)
                    num_bscans = vol.oct.shape[0]
                    vol_name = os.path.splitext(os.path.basename(rec))[0]
                    
                    for b_idx in range(num_bscans):
                        slice_data = vol.oct[b_idx]
                        slice_min = slice_data.min()
                        slice_max = slice_data.max()
                        if slice_max > slice_min:
                            slice_img = ((slice_data - slice_min) / (slice_max - slice_min) * 255.0).astype(np.uint8)
                        else:
                            slice_img = np.zeros_like(slice_data, dtype=np.uint8)
                        
                        img_pil = Image.fromarray(slice_img).convert('L').convert('RGB')
                        img_tensor, orig_w, orig_h = preprocess_image(img_pil, args.img_size)
                        
                        outputs = model(img_tensor.unsqueeze(0).to(device))
                        preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                        preds_resized = cv2.resize(preds, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                        
                        # Save
                        filename = f"{vol_name}_slice_{b_idx}.png"
                        out_path = os.path.join(args.output_dir, filename)
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        
                        if args.model_type == 'nr206':
                            color_mask_bgr = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
                            for i, c in enumerate(NR206_COLORS):
                                color_mask_bgr[preds_resized == i] = c
                            alpha = np.full((orig_h, orig_w, 1), 255, dtype=np.uint8)
                            color_mask_bgra = np.concatenate([color_mask_bgr, alpha], axis=2)
                            cv2.imwrite(out_path, color_mask_bgra)
                        else:
                            cv2.imwrite(out_path, preds_resized)
                            
                        # Add a sample grid occasionally
                        if record_idx in grid_indices and b_idx == num_bscans // 2:
                            img_np = np.array(img_pil.resize((args.img_size, args.img_size))).astype(np.float32) / 255.0
                            pred_color = colorize_mask_nr206(preds) if args.model_type == 'nr206' else colorize_mask_oct5k(preds)
                            sample_grids.append((img_np, pred_color, filename, None))
                else:
                    # Parse standard image file
                    img_pil = Image.open(rec).convert('L').convert('RGB')
                    img_tensor, orig_w, orig_h = preprocess_image(img_pil, args.img_size)
                    
                    outputs = model(img_tensor.unsqueeze(0).to(device))
                    preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                    preds_resized = cv2.resize(preds, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    
                    # Save preserving tree structure
                    rel_path = os.path.relpath(rec, args.dataset_dir)
                    out_path = os.path.join(args.output_dir, os.path.splitext(rel_path)[0] + ".png")
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    
                    if args.model_type == 'nr206':
                        color_mask_bgr = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
                        for i, c in enumerate(NR206_COLORS):
                            color_mask_bgr[preds_resized == i] = c
                        alpha = np.full((orig_h, orig_w, 1), 255, dtype=np.uint8)
                        color_mask_bgra = np.concatenate([color_mask_bgr, alpha], axis=2)
                        cv2.imwrite(out_path, color_mask_bgra)
                    else:
                        cv2.imwrite(out_path, preds_resized)
                        
                    if record_idx in grid_indices:
                        img_np = np.array(img_pil.resize((args.img_size, args.img_size))).astype(np.float32) / 255.0
                        pred_color = colorize_mask_nr206(preds) if args.model_type == 'nr206' else colorize_mask_oct5k(preds)
                        gt_color = None
                        if args.gt_dir is not None:
                            gt_path = os.path.join(args.gt_dir, os.path.splitext(os.path.basename(rec))[0] + ".png")
                            if os.path.exists(gt_path):
                                gt_pil = Image.open(gt_path).resize((args.img_size, args.img_size), Image.NEAREST)
                                gt_color = np.array(gt_pil)
                        sample_grids.append((img_np, pred_color, os.path.basename(rec), gt_color))

        # Plot and save sample grid
        if sample_grids:
            has_gt = any(gt is not None for _, _, _, gt in sample_grids)
            cols = 3 if has_gt else 2
            fig, axes = plt.subplots(len(sample_grids), cols, figsize=(4 * cols, 3.5 * len(sample_grids)))
            # Handle case of single sample
            if len(sample_grids) == 1:
                axes = np.expand_dims(axes, axis=0)
            for i, (img, pred, name, gt) in enumerate(sample_grids):
                axes[i, 0].imshow(img)
                axes[i, 0].set_title(f"Input: {name}")
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(pred)
                axes[i, 1].set_title("Predicted Mask")
                axes[i, 1].axis('off')
                
                if has_gt:
                    if gt is not None:
                        axes[i, 2].imshow(gt)
                        axes[i, 2].set_title("GT Mask")
                    axes[i, 2].axis('off')
                
            plt.tight_layout()
            grid_path = f"custom_eval_{args.run_name}_grid.png"
            plt.savefig(grid_path)
            plt.close()
            
            mlflow.log_artifact(grid_path, artifact_path="evaluation_grids")
            if os.path.exists(grid_path):
                os.remove(grid_path)

        print("Segmentation complete! Predicted masks saved and visual grid logged to MLflow.")

if __name__ == '__main__':
    main()
