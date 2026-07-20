import sys
import os
# Pre-scan arguments for --gpu to set CUDA_VISIBLE_DEVICES before torch is imported
for i, arg in enumerate(sys.argv):
    if arg == '--gpu' and i + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i+1]
        print(f"Forced CUDA_VISIBLE_DEVICES to {sys.argv[i+1]} from CLI")
        break

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import mlflow
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from model import RETFoundSegmenter
from dataset_nr206 import NR206Dataset
from dataset_oct5k import OCT5kDataset

# BGRA colors for NR206 (matching dataset_nr206.py list structure)
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

def calculate_metrics(preds, targets, num_classes=4):
    # preds: (H, W) tensor
    # targets: (H, W) tensor
    dice_scores = []
    iou_scores = []
    for c in range(num_classes):
        pred_c = (preds == c)
        target_c = (targets == c)
        intersection = (pred_c & target_c).float().sum()
        union = pred_c.float().sum() + target_c.float().sum()
        
        # Dice
        if union == 0:
            dice = 1.0
        else:
            dice = (2.0 * intersection / union).item()
        dice_scores.append(dice)
        
        # IoU
        union_iou = (pred_c | target_c).float().sum()
        if union_iou == 0:
            iou = 1.0
        else:
            iou = (intersection / union_iou).item()
        iou_scores.append(iou)
    return dice_scores, iou_scores

def map_mask(mask_np, dataset_name):
    # Map mask of either 'nr206' or 'oct5k' to common 4 classes:
    # 0: Background
    # 1: ILM
    # 2: OPL
    # 3: IS/OS / RPE / Photoreceptor
    mapped = np.zeros_like(mask_np)
    if dataset_name == 'nr206':
        mapped[mask_np == 1] = 1
        mapped[mask_np == 4] = 2
        mapped[mask_np == 6] = 3
        mapped[mask_np == 7] = 3
        mapped[mask_np == 8] = 3
    elif dataset_name == 'oct5k':
        mapped[mask_np == 1] = 1
        mapped[mask_np == 2] = 2
        mapped[mask_np == 3] = 3
        mapped[mask_np == 4] = 3
        mapped[mask_np == 5] = 3
    return mapped

def colorize_mask_common(mask_np):
    colors = [
        [0, 0, 0],       # 0: Background (Black)
        [255, 0, 0],     # 1: Red (ILM)
        [0, 255, 0],     # 2: Green (OPL)
        [0, 0, 255]      # 3: Blue (RPE)
    ]
    h, w = mask_np.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for i, c in enumerate(colors):
        color_mask[mask_np == i] = c
    return color_mask

def colorize_mask_original(mask_np):
    h, w = mask_np.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for i, c in enumerate(NR206_COLORS):
        # Convert BGR back to RGB for matplotlib display
        rgb_color = [c[2], c[1], c[0]]
        color_mask[mask_np == i] = rgb_color
    return color_mask

def main():
    parser = argparse.ArgumentParser(description="Cross Evaluation of Segmentation Models")
    parser.add_argument('--run_id', type=str, required=True, help="MLflow Run ID of the model to download")
    parser.add_argument('--model_type', type=str, required=True, choices=['finetune', 'frozen'])
    parser.add_argument('--source_dataset', type=str, required=True, choices=['oct5k', 'nr206'])
    parser.add_argument('--target_dataset', type=str, required=True, choices=['oct5k', 'nr206'])
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the target dataset root")
    parser.add_argument('--output_dir', type=str, required=True, help="Where to save prediction masks")
    parser.add_argument('--run_name', type=str, required=True, help="MLflow run name")
    parser.add_argument('--img_size', type=int, default=224, help="Image size expected by the model")
    parser.add_argument('--gpu', type=str, default=None, help="GPU index to force-bind process")
    parser.add_argument('--original_layers_grid', action='store_true', help="Save visual grid with original, unmapped layers")
    args = parser.parse_args()

    # Set up MLflow
    mlflow.set_tracking_uri('http://10.24.38.15:5000')
    mlflow.set_experiment('Cross_Evaluation_Segmentation')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Download model checkpoint from MLflow artifacts
    client = mlflow.tracking.MlflowClient()
    print(f"Querying artifacts in checkpoints/ for run {args.run_id}...")
    try:
        artifacts = client.list_artifacts(args.run_id, "checkpoints")
        pth_files = [art.path for art in artifacts if art.path.endswith('.pth')]
        if not pth_files:
            raise RuntimeError("No .pth files found in checkpoints/ directory!")
        checkpoint_path = pth_files[0]
        print(f"Downloading checkpoint: {checkpoint_path}")
        local_ckpt_path = client.download_artifacts(args.run_id, checkpoint_path)
    except Exception as e:
        raise RuntimeError(f"Failed to query/download checkpoint from MLflow run artifacts: {e}")

    # 2. Load model
    num_classes_source = 6 if args.source_dataset == 'oct5k' else 9
    model = RETFoundSegmenter(num_classes=num_classes_source, img_size=args.img_size)
    model.load_state_dict(torch.load(local_ckpt_path, map_location='cpu'))
    model = model.to(device)
    model.eval()
    print(f"Loaded model state dict successfully with {num_classes_source} classes.")

    # 3. Load Target Dataset Splits (all samples)
    if args.target_dataset == 'nr206':
        train_ds = NR206Dataset(image_dir=f"{args.data_dir}/train", mask_dir=f"{args.data_dir}/train_labels", img_size=args.img_size, remove_watermark=True, use_augmentations=False)
        test_ds = NR206Dataset(image_dir=f"{args.data_dir}/test", mask_dir=f"{args.data_dir}/test_labels", img_size=args.img_size, remove_watermark=True, use_augmentations=False)
        dataset = ConcatDataset([train_ds, test_ds])
        
        # Build mapping of dataset index to filepath details
        splits_info = []
        for img in train_ds.images:
            splits_info.append(('train', img, f"{args.data_dir}/train_labels"))
        for img in test_ds.images:
            splits_info.append(('test', img, f"{args.data_dir}/test_labels"))
    else:
        train_ds = OCT5kDataset(image_dir=f"{args.data_dir}/train/images", mask_dir=f"{args.data_dir}/train/masks", img_size=args.img_size, use_augmentations=False)
        test_ds = OCT5kDataset(image_dir=f"{args.data_dir}/test/images", mask_dir=f"{args.data_dir}/test/masks", img_size=args.img_size, use_augmentations=False)
        dataset = ConcatDataset([train_ds, test_ds])
        
        # Build mapping of dataset index to filepath details
        splits_info = []
        for img in train_ds.images:
            splits_info.append(('train', img, f"{args.data_dir}/train/masks"))
        for img in test_ds.images:
            splits_info.append(('test', img, f"{args.data_dir}/test/masks"))

    os.makedirs(args.output_dir, exist_ok=True)

    all_dice_scores = []
    all_iou_scores = []
    
    sample_grids = []
    num_samples = len(dataset)
    grid_indices = np.linspace(0, num_samples - 1, min(10, num_samples), dtype=int)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.set_tag("source_model", args.source_dataset)
        mlflow.set_tag("model_type", args.model_type)
        mlflow.set_tag("target_dataset", args.target_dataset)
        
        with torch.no_grad():
            for idx in range(num_samples):
                image_tensor, mask_tensor = dataset[idx]
                
                # Inference
                outputs = model(image_tensor.unsqueeze(0).to(device))
                preds = torch.argmax(outputs, dim=1).squeeze(0).cpu()
                
                # Map prediction and target masks to the common 4 classes for evaluation
                mapped_preds = torch.as_tensor(map_mask(preds.numpy(), args.source_dataset))
                mapped_target = torch.as_tensor(map_mask(mask_tensor.numpy(), args.target_dataset))
                
                # Calculate metrics (4 classes space)
                dice, iou = calculate_metrics(mapped_preds, mapped_target, num_classes=4)
                all_dice_scores.append(dice)
                all_iou_scores.append(iou)

                # Get original file properties to restore original shape
                split, filename, mask_dir_path = splits_info[idx]
                if args.target_dataset == 'nr206':
                    orig_mask = cv2.imread(os.path.join(mask_dir_path, filename))
                    orig_h, orig_w = orig_mask.shape[:2]
                else:
                    orig_mask = Image.open(os.path.join(mask_dir_path, filename))
                    orig_w, orig_h = orig_mask.size

                # Resize prediction using Nearest Neighbor
                preds_np = preds.numpy().astype(np.uint8)
                preds_resized = cv2.resize(preds_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

                # Save mask to disk
                if args.source_dataset == 'nr206':
                    # Save as BGRA matching original NR206 colors
                    color_mask_bgr = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
                    for i, c in enumerate(NR206_COLORS):
                        color_mask_bgr[preds_resized == i] = c
                    alpha = np.full((orig_h, orig_w, 1), 255, dtype=np.uint8)
                    color_mask_bgra = np.concatenate([color_mask_bgr, alpha], axis=2)
                    cv2.imwrite(os.path.join(args.output_dir, filename), color_mask_bgra)
                else:
                    # Save as single-channel grayscale 0-5 for OCT5k
                    cv2.imwrite(os.path.join(args.output_dir, filename), preds_resized)

                # Sample for visual grid
                if idx in grid_indices:
                    img_np = image_tensor.permute(1, 2, 0).numpy()
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_np = img_np * std + mean
                    img_np = np.clip(img_np, 0.0, 1.0)
                    
                    if args.original_layers_grid:
                        target_color = colorize_mask_original(mask_tensor.numpy())
                        pred_color = colorize_mask_original(preds.numpy())
                    else:
                        target_color = colorize_mask_common(mapped_target.numpy())
                        pred_color = colorize_mask_common(mapped_preds.numpy())
                    sample_grids.append((img_np, target_color, pred_color, filename))

        # Metrics aggregation
        mean_dice_per_class = np.mean(all_dice_scores, axis=0)
        mean_iou_per_class = np.mean(all_iou_scores, axis=0)
        overall_mean_dice = np.mean(mean_dice_per_class)
        overall_mean_iou = np.mean(mean_iou_per_class)

        # Log to MLflow
        mlflow.log_metric("overall_mean_dice", overall_mean_dice)
        mlflow.log_metric("overall_mean_iou", overall_mean_iou)
        for c in range(4):
            mlflow.log_metric(f"class_{c}_dice", mean_dice_per_class[c])
            mlflow.log_metric(f"class_{c}_iou", mean_iou_per_class[c])

        print(f"Evaluation finished for {args.run_name}.")
        print(f"Overall Mean Dice: {overall_mean_dice:.4f} | Overall Mean IoU: {overall_mean_iou:.4f}")

        # Plot and save sample grid
        fig, axes = plt.subplots(len(sample_grids), 3, figsize=(12, 3.5 * len(sample_grids)))
        for i, (img, target, pred, name) in enumerate(sample_grids):
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"Input: {name}")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(target)
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred)
            axes[i, 2].set_title("Prediction")
            axes[i, 2].axis('off')
            
        plt.tight_layout()
        grid_path = f"cross_eval_{args.run_name}_grid.png"
        plt.savefig(grid_path)
        plt.close()
        
        mlflow.log_artifact(grid_path, artifact_path="evaluation_grids")
        os.remove(grid_path)

if __name__ == '__main__':
    main()
