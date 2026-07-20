import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import albumentations as A

# AROI database image shape is 512x1024.
# AROI unique values: [0, 1, 2, 3, 4, 5, 6, 7]

def get_aroi_split_indices(data_dir, val_ratio=0.2, seed=42):
    """
    AROI structure:
      AROI - online/24 patient/patientX/raw/labeled/patientX_rawYYYY.png
      AROI - online/24 patient/patientX/mask/number/patientX_rawYYYY.png
    """
    base_dir = os.path.join(data_dir, "AROI - online", "24 patient")
    if not os.path.isdir(base_dir):
        # Alternative path
        base_dir = os.path.join(data_dir, "AROI", "AROI - online", "24 patient")
        if not os.path.isdir(base_dir):
            base_dir = os.path.join(data_dir, "24 patient")
    
    # We want subject-level split for validation to prevent data leakage
    patients = sorted([d for d in os.listdir(base_dir) if d.startswith("patient")])
    
    rng = np.random.default_rng(seed)
    indices = np.arange(len(patients))
    rng.shuffle(indices)
    n_val = max(1, int(len(patients) * val_ratio))
    
    train_patients = [patients[i] for i in indices[:-n_val]]
    val_patients = [patients[i] for i in indices[-n_val:]]
    
    def get_pairs(patient_list):
        pairs = []
        for p in patient_list:
            raw_dir = os.path.join(base_dir, p, "raw", "labeled")
            mask_dir = os.path.join(base_dir, p, "mask", "number")
            if not os.path.isdir(raw_dir) or not os.path.isdir(mask_dir):
                continue
            for f in sorted(os.listdir(raw_dir)):
                if f.lower().endswith(".png"):
                    img_path = os.path.join(raw_dir, f)
                    msk_path = os.path.join(mask_dir, f)
                    if os.path.isfile(msk_path):
                        pairs.append((img_path, msk_path))
        return pairs

    return get_pairs(train_patients), get_pairs(val_patients)

class AROIDataset(Dataset):
    """
    AROI Retinal OCT Segmentation Dataset.
    """
    def __init__(self, data_dir, split='train', img_size=256, use_augmentations=False, val_ratio=0.2, seed=42):
        self.img_size = img_size
        self.use_augmentations = use_augmentations
        
        train_pairs, val_pairs = get_aroi_split_indices(data_dir, val_ratio=val_ratio, seed=seed)
        self.pairs = train_pairs if split == 'train' else val_pairs
        
        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if self.use_augmentations:
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussianBlur(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
            ], additional_targets={'mask': 'mask'})

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, msk_path = self.pairs[idx]
        
        img_np = np.array(Image.open(img_path).convert('L'))
        mask_np = np.array(Image.open(msk_path).convert('L'))
        
        if self.use_augmentations:
            augmented = self.aug(image=img_np, mask=mask_np)
            img_np = augmented['image']
            mask_np = augmented['mask']
            
        image = self.img_transform(Image.fromarray(img_np))
        
        msk_pil = Image.fromarray(mask_np)
        msk_resized = msk_pil.resize((self.img_size, self.img_size), Image.NEAREST)
        mask = torch.as_tensor(np.array(msk_resized), dtype=torch.long)
        
        return image, mask
