import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import albumentations as A
import numpy as np

class OCT5kDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=224, use_augmentations=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.img_size = img_size
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

        image_np = np.array(Image.open(img_path).convert('L'))
        mask_np = np.array(Image.open(mask_path).convert('L'))

        # Apply Albumentations if enabled
        if self.use_augmentations:
            augmented = self.aug(image=image_np, mask=mask_np)
            image_np = augmented['image']
            mask_np = augmented['mask']

        # Convert back to PIL Image and apply transforms
        image = Image.fromarray(image_np)
        image = self.img_transform(image)
        
        mask_pil = Image.fromarray(mask_np)
        mask_resized = mask_pil.resize((self.img_size, self.img_size), Image.NEAREST)
        mask = torch.as_tensor(np.array(mask_resized), dtype=torch.long)
        
        return image, mask