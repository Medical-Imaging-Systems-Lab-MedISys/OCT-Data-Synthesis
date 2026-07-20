import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import albumentations as A


# Grayscale pixel values present in the Kaggle Retinal Segmentation masks
# mapped to integer class IDs 0-7
KAGGLE_GRAY_VALUES = [0, 36, 72, 109, 145, 182, 218, 255]


def build_gray_to_class_lut():
    """
    Build a 256-element look-up table: gray_value → class_id.
    Values not in the canonical list are mapped to the nearest entry.
    """
    lut = np.zeros(256, dtype=np.uint8)
    vals = np.array(KAGGLE_GRAY_VALUES)
    for g in range(256):
        lut[g] = int(np.argmin(np.abs(vals - g)))
    return lut


GRAY_TO_CLASS_LUT = build_gray_to_class_lut()


def get_split_indices(image_dir, mask_dir, val_ratio=0.2, seed=42):
    """
    Return sorted lists of (image_path, mask_path) for train and val splits.
    Filenames are sorted for reproducibility.
    """
    img_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    paired = []
    for fname in img_files:
        img_path = os.path.join(image_dir, fname)
        msk_path = os.path.join(mask_dir, fname)
        if os.path.isfile(msk_path):
            paired.append((img_path, msk_path))

    rng = np.random.default_rng(seed)
    indices = np.arange(len(paired))
    rng.shuffle(indices)
    n_val = max(1, int(len(paired) * val_ratio))

    train_pairs = [paired[i] for i in indices[:-n_val]]
    val_pairs = [paired[i] for i in indices[-n_val:]]
    return train_pairs, val_pairs


class KaggleRetinalDataset(Dataset):
    """
    Kaggle Retinal OCT Segmentation Dataset.

    Mask grayscale pixel values [0, 36, 72, 109, 145, 182, 218, 255] are
    mapped to integer class IDs 0-7 via a pre-built look-up table.

    Args:
        image_dir (str): Directory containing PNG images.
        mask_dir (str): Directory containing PNG mask images.
        split (str): 'train' or 'val'.
        img_size (int): Target spatial size for both H and W.
        use_augmentations (bool): Whether to apply Albumentations pipeline.
        val_ratio (float): Fraction of data reserved for validation.
        seed (int): Random seed for the split.
    """

    def __init__(self, image_dir, mask_dir, split='train', img_size=256,
                 use_augmentations=False, val_ratio=0.2, seed=42):
        self.img_size = img_size
        self.use_augmentations = use_augmentations

        train_pairs, val_pairs = get_split_indices(image_dir, mask_dir,
                                                   val_ratio=val_ratio,
                                                   seed=seed)
        self.pairs = train_pairs if split == 'train' else val_pairs

        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        if self.use_augmentations:
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussianBlur(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1,
                                   rotate_limit=15, p=0.5),
            ], additional_targets={'mask': 'mask'})

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, msk_path = self.pairs[idx]

        img_np = np.array(Image.open(img_path).convert('L'))     # (H, W) uint8
        gray_np = np.array(Image.open(msk_path).convert('L'))    # (H, W) uint8

        # Map grayscale values to class IDs using the LUT
        class_map = GRAY_TO_CLASS_LUT[gray_np]   # (H, W) uint8, values 0-7

        if self.use_augmentations:
            augmented = self.aug(image=img_np, mask=class_map)
            img_np = augmented['image']
            class_map = augmented['mask']

        image = self.img_transform(Image.fromarray(img_np))

        msk_pil = Image.fromarray(class_map)
        msk_resized = msk_pil.resize((self.img_size, self.img_size),
                                     Image.NEAREST)
        mask_tensor = torch.as_tensor(np.array(msk_resized), dtype=torch.long)

        return image, mask_tensor
