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
matplotlib.use('Agg')

from model import RETFoundSegmenter
from dataset_retouch import RETOUCHDataset

# --- Hyperparameters ---
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_CLASSES = 4   # 0=background, 1=IRF, 2=SRF, 3=PED


def calculate_dice(preds, targets, num_classes):
    preds = torch.argmax(preds, dim=1)  # (B, H, W)
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
        description="Train RETFound Segmenter on RETOUCH dataset")
    parser.add_argument('--data_dir', type=str, required=True,
                        help="Path to the root RETOUCH directory (contains "
                             "TrainingCirrus / TrainingSpectralis / "
                             "TrainingTopcon)")
    parser.add_argument('--weights_path', type=str,
                        default="./RETFound_oct_weights.pth",
                        help="Path to RETFound weights")
    parser.add_argument('--freeze_backbone', action='store_true',
                        help="Freeze RETFound encoder weights")
    parser.add_argument('--run_name', type=str,
                        default="RETFound_RETOUCH_Finetune_Aug_256",
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
    train_dataset = RETOUCHDataset(
        data_dir=args.data_dir,
        split='train',
        img_size=args.img_size,
        use_augmentations=args.use_augmentations,
    )
    val_dataset = RETOUCHDataset(
        data_dir=args.data_dir,
        split='val',
        img_size=args.img_size,
        use_augmentations=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)

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
    mlflow.set_experiment("RETOUCH_Segmentation")

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
            "Dataset: RETOUCH (Cirrus + Spectralis + Topcon scanners). "
            "NUM_CLASSES=4 (0=background, 1=IRF, 2=SRF, 3=PED). "
            "Augmentations: HorizontalFlip, RandomBrightnessContrast, "
            "GaussianBlur, ShiftScaleRotate. "
            f"Image size: {args.img_size}x{args.img_size}. "
            "Preprocessing: 3D MHD/RAW volumes sliced to 2D grayscale, "
            "converted to 3-channel by repetition, normalised with "
            "ImageNet mean/std. Subject-level 80/20 train/val split."
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

                train_loss += loss.item()
                dice = calculate_dice(outputs, masks, NUM_CLASSES)
                train_dice_total += dice
                loop.set_postfix(loss=loss.item(), dice=dice)

            avg_train_loss = train_loss / len(train_loader)
            avg_train_dice = train_dice_total / len(train_loader)

            # 5. Validation Loop
            model.eval()
            val_loss = 0.0
            val_dice_total = 0.0

            with torch.no_grad():
                for i, (images, masks) in enumerate(val_loader):
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
                    val_dice_total += calculate_dice(outputs, masks,
                                                     NUM_CLASSES)

                    # Log validation grid every 5 epochs (first batch only)
                    if i == 0 and (epoch + 1) % 5 == 0:
                        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
                        for j in range(min(4, images.size(0))):
                            img_disp = images[j].cpu().permute(
                                1, 2, 0).numpy()
                            img_disp = ((img_disp - img_disp.min()) /
                                        (img_disp.max() - img_disp.min() +
                                         1e-8))

                            axes[0, j].imshow(img_disp)
                            axes[0, j].set_title("Input")
                            axes[0, j].axis('off')

                            axes[1, j].imshow(
                                masks[j].cpu().numpy(),
                                cmap='nipy_spectral',
                                vmin=0, vmax=NUM_CLASSES - 1)
                            axes[1, j].set_title("GT Mask")
                            axes[1, j].axis('off')

                            pred = torch.argmax(
                                outputs[j], dim=0).cpu().numpy()
                            axes[2, j].imshow(
                                pred, cmap='nipy_spectral',
                                vmin=0, vmax=NUM_CLASSES - 1)
                            axes[2, j].set_title("Predicted Mask")
                            axes[2, j].axis('off')

                        plt.tight_layout()
                        grid_path = (f"val_grid_{args.run_name}"
                                     f"_epoch_{epoch+1}.png")
                        plt.savefig(grid_path)
                        mlflow.log_artifact(grid_path,
                                            artifact_path="validation_grids")
                        plt.close(fig)
                        if os.path.exists(grid_path):
                            os.remove(grid_path)

            avg_val_loss = val_loss / len(val_loader)
            avg_val_dice = val_dice_total / len(val_loader)

            print(f"Train Loss: {avg_train_loss:.4f} | "
                  f"Train Dice: {avg_train_dice:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val Dice: {avg_val_dice:.4f}")

            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "train_dice": avg_train_dice,
                "val_loss": avg_val_loss,
                "val_dice": avg_val_dice,
            }, step=epoch + 1)

            # Save Best Model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_to_save = (model.module
                                 if hasattr(model, 'module') else model)
                checkpoint_name = f"{args.run_name}_best_model.pth"
                torch.save(model_to_save.state_dict(), checkpoint_name)
                try:
                    os.sync()
                except Exception:
                    pass
                time.sleep(5)
                mlflow.log_artifact(checkpoint_name,
                                    artifact_path="checkpoints")
                print(f"Saved new best model as {checkpoint_name}!")


if __name__ == "__main__":
    train()
