import sys
import os
# Pre-scan arguments for --gpu to set CUDA_VISIBLE_DEVICES before torch is imported
for i, arg in enumerate(sys.argv):
    if arg == '--gpu' and i + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i+1]
        print(f"Forced CUDA_VISIBLE_DEVICES to {sys.argv[i+1]} from CLI")
        break

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import mlflow
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Agg')

from model import RETFoundSegmenter
from dataset_aroi import AROIDataset

# --- Hyperparameters ---
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_CLASSES = 8   # AROI has 8 classes (0-7)


def calculate_dice(preds, targets, num_classes):
    preds = torch.argmax(preds, dim=1)
    dice = 0.0
    for c in range(num_classes):
        pred_c = (preds == c)
        target_c = (targets == c)
        intersection = (pred_c & target_c).float().sum()
        union = pred_c.float().sum() + target_c.float().sum()
        if union == 0:
            dice += 1.0
        else:
            dice += (2.0 * intersection / union).item()
    return dice / num_classes


def train():
    parser = argparse.ArgumentParser(
        description="Train RETFound Segmenter on AROI dataset")
    parser.add_argument('--data_dir', type=str, required=True,
                        help="Root AROI directory (parent of AROI - online/)")
    parser.add_argument('--weights_path', type=str,
                        default="./RETFound_oct_weights.pth",
                        help="Path to RETFound weights")
    parser.add_argument('--freeze_backbone', action='store_true',
                        help="Freeze RETFound encoder weights")
    parser.add_argument('--run_name', type=str,
                        default="RETFound_AROI_Finetune_Aug_256",
                        help="MLflow run name")
    parser.add_argument('--use_augmentations', action='store_true',
                        help="Apply Albumentations data augmentation")
    parser.add_argument('--img_size', type=int, default=256,
                        help="Input image size for model and transforms")
    parser.add_argument('--gpu', type=str, default=None,
                        help="GPU index to force-bind training process")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Prepare Data
    train_dataset = AROIDataset(
        data_dir=args.data_dir,
        split='train',
        img_size=args.img_size,
        use_augmentations=args.use_augmentations,
    )
    val_dataset = AROIDataset(
        data_dir=args.data_dir,
        split='val',
        img_size=args.img_size,
        use_augmentations=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 2. Initialize Model
    model = RETFoundSegmenter(num_classes=NUM_CLASSES,
                              img_size=args.img_size,
                              pretrained_path=args.weights_path)

    if args.freeze_backbone:
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("Froze RETFound encoder backbone weights.")

    model = model.to(device)

    # 3. Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE,
                            weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    # MLflow tracking
    mlflow.set_tracking_uri("http://10.24.38.15:5000")
    mlflow.set_experiment("AROI_Segmentation")

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            "BATCH_SIZE": BATCH_SIZE,
            "EPOCHS": EPOCHS,
            "LEARNING_RATE": LEARNING_RATE,
            "NUM_CLASSES": NUM_CLASSES,
        })
        mlflow.set_tag(
            "mlflow.note.content",
            "RETFound ViT-Large backbone with convolutional decoder head. "
            "Dataset: Annotated Retinal OCT Images (AROI). "
            "NUM_CLASSES=8 (labels 0-7). "
            "Augmentations: HorizontalFlip, RandomBrightnessContrast, "
            "GaussianBlur, ShiftScaleRotate. "
            f"Image size: {args.img_size}x{args.img_size}. "
            "Preprocessing: uint8 OCT slices converted to 3-channel by "
            "repetition, normalised with ImageNet mean/std. "
            "80/20 subject-level split (24 subjects: 19 train, 5 val, seed=42)."
        )

        # 4. Training Loop
        best_val_loss = float('inf')

        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            train_dice_total = 0.0

            loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
            for images, masks in loop:
                images = images.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item() * images.size(0)
                dice = calculate_dice(outputs, masks, NUM_CLASSES)
                train_dice_total += dice * images.size(0)

                loop.set_postfix(loss=loss.item(), dice=dice)

            epoch_train_loss = train_loss / len(train_loader.dataset)
            epoch_train_dice = train_dice_total / len(train_loader.dataset)

            # Validation
            model.eval()
            val_loss = 0.0
            val_dice_total = 0.0

            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)

                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, masks)

                    val_loss += loss.item() * images.size(0)
                    dice = calculate_dice(outputs, masks, NUM_CLASSES)
                    val_dice_total += dice * images.size(0)

            epoch_val_loss = val_loss / len(val_loader.dataset)
            epoch_val_dice = val_dice_total / len(val_loader.dataset)

            print(f"Epoch [{epoch+1}/{EPOCHS}] - "
                  f"Train Loss: {epoch_train_loss:.4f}, Train Dice: {epoch_train_dice:.4f} | "
                  f"Val Loss: {epoch_val_loss:.4f}, Val Dice: {epoch_val_dice:.4f}")

            # Log metrics
            mlflow.log_metrics({
                "train_loss": epoch_train_loss,
                "train_dice": epoch_train_dice,
                "val_loss": epoch_val_loss,
                "val_dice": epoch_val_dice,
            }, step=epoch)

            # Save checkpoint
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                os.makedirs("checkpoints", exist_ok=True)
                ckpt_path = f"checkpoints/{args.run_name}_best.pth"
                torch.save(model.state_dict(), ckpt_path)
                mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
                print(f"Saved best model checkpoint to {ckpt_path}")

        # 5. Log final visualization grid
        model.eval()
        sample_grids = []
        grid_indices = np.linspace(0, len(val_dataset) - 1, min(10, len(val_dataset)), dtype=int)
        
        with torch.no_grad():
            for idx in grid_indices:
                img_tensor, mask_tensor = val_dataset[idx]
                img_input = img_tensor.unsqueeze(0).to(device)
                outputs = model(img_input)
                preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
                
                # Unnormalize image for plotting
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = img_tensor.permute(1, 2, 0).numpy()
                img_unnorm = np.clip(img_np * std + mean, 0.0, 1.0)
                
                sample_grids.append((img_unnorm, preds, mask_tensor.numpy(), f"sample_{idx}"))

        if sample_grids:
            fig, axes = plt.subplots(len(sample_grids), 3, figsize=(12, 3.5 * len(sample_grids)))
            if len(sample_grids) == 1:
                axes = np.expand_dims(axes, axis=0)
            for i, (img, pred, gt, name) in enumerate(sample_grids):
                axes[i, 0].imshow(img)
                axes[i, 0].set_title(f"Input: {name}")
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(pred, vmin=0, vmax=NUM_CLASSES-1, cmap='tab10')
                axes[i, 1].set_title("Predicted Mask")
                axes[i, 1].axis('off')
                
                axes[i, 2].imshow(gt, vmin=0, vmax=NUM_CLASSES-1, cmap='tab10')
                axes[i, 2].set_title("Ground Truth Mask")
                axes[i, 2].axis('off')
                
            plt.tight_layout()
            grid_path = f"eval_{args.run_name}_grid.png"
            plt.savefig(grid_path)
            plt.close()
            
            mlflow.log_artifact(grid_path, artifact_path="evaluation_grids")
            if os.path.exists(grid_path):
                os.remove(grid_path)

        print("Training completed successfully.")


if __name__ == '__main__':
    train()
