import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import albumentations as A

COLORS = [
    [0, 0, 255],     # 1: Red
    [0, 128, 128],   # 2: Olive
    [0, 255, 255],   # 3: Yellow
    [0, 128, 0],     # 4: DarkGreen
    [0, 255, 0],     # 5: BrightGreen
    [255, 255, 0],   # 6: Cyan
    [255, 0, 0],     # 7: Blue
    [255, 0, 255]    # 8: Magenta
]

class NR206Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=224, remove_watermark=True, use_augmentations=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        self.img_size = img_size
        self.remove_watermark = remove_watermark
        self.use_augmentations = use_augmentations
        
        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if self.use_augmentations:
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # Load image as numpy array
        img_np = np.array(Image.open(img_path).convert('L'))
        
        # Cover the watermark if requested
        if self.remove_watermark and img_np.shape[0] >= 500 and img_np.shape[1] >= 750:
            clean_patch = img_np[350:, 600:]
            img_np[350:, :150] = np.flip(clean_patch, axis=1)
            
        # Load mask using OpenCV to match exact colors
        mask_bgr = cv2.imread(mask_path)
        if mask_bgr is None:
            raise FileNotFoundError(f"Mask not found at {mask_path}")
        
        # Map colors to class indices (0 to 8)
        class_map = np.zeros(mask_bgr.shape[:2], dtype=np.uint8)
        for i, c in enumerate(COLORS, start=1):
            match = (mask_bgr[:, :, 0] == c[0]) & \
                    (mask_bgr[:, :, 1] == c[1]) & \
                    (mask_bgr[:, :, 2] == c[2])
            class_map[match] = i

        # Apply Albumentations if enabled
        if self.use_augmentations:
            augmented = self.aug(image=img_np, mask=class_map)
            img_np = augmented['image']
            class_map = augmented['mask']

        # Convert back to PIL Image and apply standard transforms
        image = Image.fromarray(img_np)
        image = self.img_transform(image)

        # Resize mask to self.img_size using Nearest Neighbor to preserve integers
        class_map_pil = Image.fromarray(class_map)
        class_map_resized = class_map_pil.resize((self.img_size, self.img_size), Image.NEAREST)
        mask_tensor = torch.as_tensor(np.array(class_map_resized), dtype=torch.long)

        return image, mask_tensor
