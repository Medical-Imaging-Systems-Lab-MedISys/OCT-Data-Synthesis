import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import albumentations as A


def read_mhd_raw(mhd_path):
    """
    Parse a .mhd header and load the associated .raw binary volume.
    Returns a numpy array reshaped to (DimZ, DimY, DimX) as uint8.
    """
    header = {}
    with open(mhd_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                key, _, val = line.partition('=')
                header[key.strip()] = val.strip()

    dims = list(map(int, header['DimSize'].split()))   # [X, Y, Z]
    elem_type = header.get('ElementType', 'MET_UCHAR')
    dtype = np.uint8 if 'UCHAR' in elem_type else np.float32

    raw_file = header.get('ElementDataFile', '')
    if not os.path.isabs(raw_file):
        raw_file = os.path.join(os.path.dirname(mhd_path), raw_file)

    data = np.fromfile(raw_file, dtype=dtype)
    # DimSize is X Y Z → reshape to (Z, Y, X)
    data = data.reshape((dims[2], dims[1], dims[0]))
    return data


def collect_slices(data_dir, split='train', val_ratio=0.2, seed=42):
    """
    Walk through all three scanner folders in data_dir (TrainingCirrus,
    TrainingSpectralis, TrainingTopcon), collect per-subject paths, do an
    80/20 subject-level split, then return flat list of (oct_mhd, ref_mhd) tuples.
    """
    scanner_dirs = [
        os.path.join(data_dir, 'TrainingCirrus'),
        os.path.join(data_dir, 'TrainingSpectralis'),
        os.path.join(data_dir, 'TrainingTopcon'),
    ]

    subjects = []
    for scanner_dir in scanner_dirs:
        if not os.path.isdir(scanner_dir):
            continue
        for subj in sorted(os.listdir(scanner_dir)):
            subj_path = os.path.join(scanner_dir, subj)
            oct_mhd = os.path.join(subj_path, 'oct.mhd')
            ref_mhd = os.path.join(subj_path, 'reference.mhd')
            if os.path.isfile(oct_mhd) and os.path.isfile(ref_mhd):
                subjects.append((oct_mhd, ref_mhd))

    # Reproducible subject-level split
    rng = np.random.default_rng(seed)
    indices = np.arange(len(subjects))
    rng.shuffle(indices)
    n_val = max(1, int(len(subjects) * val_ratio))
    if split == 'val':
        chosen = [subjects[i] for i in indices[-n_val:]]
    else:
        chosen = [subjects[i] for i in indices[:-n_val]]

    return chosen


class RETOUCHDataset(Dataset):
    """
    Dataset for the RETOUCH challenge.

    Reads 3D MHD/RAW volumes, extracts 2D axial slices on-the-fly.
    Label values: 0=background, 1=IRF, 2=SRF, 3=PED → 4 classes (0-3).

    Args:
        data_dir (str): Root directory containing TrainingCirrus /
                        TrainingSpectralis / TrainingTopcon sub-folders.
        split (str): 'train' or 'val'.
        img_size (int): Target spatial size (both H and W).
        use_augmentations (bool): Whether to apply Albumentations pipeline.
        val_ratio (float): Fraction of subjects reserved for validation.
        seed (int): Random seed for the subject-level split.
    """

    def __init__(self, data_dir, split='train', img_size=256,
                 use_augmentations=False, val_ratio=0.2, seed=42):
        self.img_size = img_size
        self.use_augmentations = use_augmentations
        self.split = split

        subject_pairs = collect_slices(data_dir, split=split,
                                       val_ratio=val_ratio, seed=seed)

        # Pre-load slice index: list of (oct_volume_path, ref_volume_path, slice_idx)
        self.samples = []
        for oct_mhd, ref_mhd in subject_pairs:
            # Read header to get number of slices without loading data
            n_slices = self._get_n_slices(oct_mhd)
            for s in range(n_slices):
                self.samples.append((oct_mhd, ref_mhd, s))

        # Cache loaded volumes to avoid re-reading the same file repeatedly
        self._volume_cache = {}

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

    # ------------------------------------------------------------------
    def _get_n_slices(self, mhd_path):
        with open(mhd_path, 'r') as f:
            for line in f:
                if line.strip().startswith('DimSize'):
                    dims = list(map(int, line.split('=')[1].strip().split()))
                    return dims[2]   # Z dimension
        return 0

    def _load_volume(self, mhd_path):
        if mhd_path not in self._volume_cache:
            self._volume_cache[mhd_path] = read_mhd_raw(mhd_path)
        return self._volume_cache[mhd_path]

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        oct_mhd, ref_mhd, slice_idx = self.samples[idx]

        oct_vol = self._load_volume(oct_mhd)   # (Z, Y, X)
        ref_vol = self._load_volume(ref_mhd)   # (Z, Y, X)

        img_np = oct_vol[slice_idx].astype(np.uint8)   # (H, W)
        msk_np = ref_vol[slice_idx].astype(np.uint8)   # (H, W), values 0-3

        if self.use_augmentations:
            augmented = self.aug(image=img_np, mask=msk_np)
            img_np = augmented['image']
            msk_np = augmented['mask']

        # Image → PIL → normalised 3-channel tensor
        image = self.img_transform(Image.fromarray(img_np))

        # Mask → resize with nearest-neighbour → long tensor
        msk_pil = Image.fromarray(msk_np)
        msk_resized = msk_pil.resize((self.img_size, self.img_size),
                                     Image.NEAREST)
        mask_tensor = torch.as_tensor(np.array(msk_resized), dtype=torch.long)

        return image, mask_tensor
