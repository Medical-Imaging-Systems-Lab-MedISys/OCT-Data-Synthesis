import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import albumentations as A


class ManualDelineationDataset(Dataset):
    """
    OCT Manual Delineations 2018 dataset.

    Reads pre-processed PNG images from ``processed_images/`` and integer
    PNG masks from ``processed_masks/``.  Sub-folders correspond to
    individual .vol scan files.

    Classes (12 total):
        0  – vitreous (above boundary 0)
        1-10 – retinal layers between consecutive boundaries (11 boundaries)
        11 – choroid/sclera (below boundary 10)

    80/20 subject-level split based on sub-folder names.

    Args:
        data_dir (str): Root of the Manual Delineations data tree (parent of
                        ``processed_images/`` and ``processed_masks/``).
        split (str): 'train' or 'val'.
        img_size (int): Spatial size for resize.
        use_augmentations (bool): Albumentations pipeline on/off.
        val_ratio (float): Fraction of subjects reserved for validation.
        seed (int): RNG seed.
    """

    def __init__(self, data_dir, split='train', img_size=256,
                 use_augmentations=False, val_ratio=0.2, seed=42):
        self.img_size = img_size
        self.use_augmentations = use_augmentations

        img_root = os.path.join(data_dir, 'processed_images')
        msk_root = os.path.join(data_dir, 'processed_masks')

        subjects = sorted(os.listdir(img_root))

        rng = np.random.default_rng(seed)
        idx = np.arange(len(subjects))
        rng.shuffle(idx)
        n_val = max(1, int(len(subjects) * val_ratio))

        if split == 'val':
            chosen = [subjects[i] for i in idx[-n_val:]]
        else:
            chosen = [subjects[i] for i in idx[:-n_val]]

        self.pairs = []
        for subj in chosen:
            img_dir = os.path.join(img_root, subj)
            msk_dir = os.path.join(msk_root, subj)
            if not (os.path.isdir(img_dir) and os.path.isdir(msk_dir)):
                continue
            for fname in sorted(os.listdir(img_dir)):
                if not fname.lower().endswith('.png'):
                    continue
                msk_path = os.path.join(msk_dir, fname)
                if os.path.isfile(msk_path):
                    self.pairs.append((os.path.join(img_dir, fname),
                                       msk_path))

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

        img_np = np.array(Image.open(img_path).convert('L'))
        msk_np = np.array(Image.open(msk_path).convert('L'))

        if self.use_augmentations:
            augmented = self.aug(image=img_np, mask=msk_np)
            img_np = augmented['image']
            msk_np = augmented['mask']

        image = self.img_transform(Image.fromarray(img_np))

        msk_pil = Image.fromarray(msk_np)
        msk_resized = msk_pil.resize((self.img_size, self.img_size),
                                     Image.NEAREST)
        mask_tensor = torch.as_tensor(np.array(msk_resized), dtype=torch.long)

        return image, mask_tensor
