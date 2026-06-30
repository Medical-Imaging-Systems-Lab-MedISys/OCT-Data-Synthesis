import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import RETFoundSegmenter
from dataset_oct5k import OCT5kDataset

import argparse

# --- Hyperparameters ---
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_CLASSES = 6 # Background, ILM, OPL, IS/OS, IBRPE, OBRPE

def train():
    parser = argparse.ArgumentParser(description="Train RETFound Segmenter on OCT5k")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the root of the split dataset")
    parser.add_argument('--weights_path', type=str, default="./RETFound_oct_weights.pth", help="Path to RETFound weights")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Prepare Data
    train_dataset = OCT5kDataset(image_dir=f'{args.data_dir}/train/images', mask_dir=f'{args.data_dir}/train/masks')
    val_dataset = OCT5kDataset(image_dir=f'{args.data_dir}/test/images', mask_dir=f'{args.data_dir}/test/masks')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 2. Initialize Model
    model = RETFoundSegmenter(num_classes=NUM_CLASSES, pretrained_path=args.weights_path)
    
    # Enable Multi-GPU if available
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        
    model = model.to(device)

    # 3. Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() # For Mixed Precision

    # 4. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            # Mixed Precision Forward pass
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        
        # 5. Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Handle saving properly if using DataParallel
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), "best_duke_retfound_model.pth")
            print("Saved new best model!")

if __name__ == "__main__":
    train()