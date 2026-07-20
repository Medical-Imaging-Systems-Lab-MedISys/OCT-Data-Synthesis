import os
import csv
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import albumentations as A


def load_boundary_csv(csv_path):
    """
    Load a boundary CSV file.  Each row contains (x, y) coordinates of one
    control point on a boundary curve.  Returns arrays x_arr, y_arr.
    """
    xs, ys = [], []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                try:
                    xs.append(float(row[0]))
                    ys.append(float(row[1]))
                except ValueError:
                    continue   # skip header rows
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def boundaries_to_mask(img_h, img_w, boundary_csvs):
    """
    Convert 7 boundary curves to a pixel-wise segmentation mask.

    Classes:
        0  – vitreous (above boundary 0)
        1  – layer between boundary 0 and 1
        ...
        6  – layer between boundary 5 and 6
        7  – choroid / below boundary 6

    Args:
        img_h, img_w (int): Image height and width.
        boundary_csvs (list[str]): Ordered list of 7 CSV paths (b0 .. b6).

    Returns:
        mask (np.ndarray): uint8 array of shape (H, W), values 0-7.
    """
    col_indices = np.arange(img_w, dtype=np.float64)

    # Interpolate each boundary to get one y-value per column
    boundary_y = []
    for csv_path in boundary_csvs:
        xs, ys = load_boundary_csv(csv_path)
        if len(xs) < 2:
            # Degenerate: use constant mid-row
            boundary_y.append(np.full(img_w, img_h / 2.0))
            continue
        # Sort by x
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]
        interp_y = np.interp(col_indices, xs, ys,
                             left=ys[0], right=ys[-1])
        boundary_y.append(interp_y)   # shape (W,)

    boundary_y = np.stack(boundary_y, axis=0)   # (7, W)

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    row_indices = np.arange(img_h, dtype=np.float64)

    for col in range(img_w):
        bys = boundary_y[:, col]   # 7 y-values
        for row in range(img_h):
            # Count how many boundaries are strictly above this row
            n_above = int(np.sum(bys <= row))
            mask[row, col] = min(n_above, 7)

    return mask


def find_octid_subjects(raw_data_dir):
    """
    Scan the OCTID manual-segmentation folder structure and return a list of
    dicts, each describing one subject:
      {
        'img_path': path to .jpeg,
        'csv_paths': sorted list of 7 boundary CSV paths,
        'name':  stem name of the subject
      }
    """
    base = os.path.join(raw_data_dir,
                        'Manual-Segmenation',
                        'Manual_Segmentation')
    subjects = []
    for entry in sorted(os.listdir(base)):
        entry_path = os.path.join(base, entry)
        if not (os.path.isdir(entry_path) and
                entry_path.endswith('_octSegmentation')):
            continue
        name = entry.replace('_octSegmentation', '')

        # JPEG image
        jpg_candidates = glob.glob(
            os.path.join(entry_path, name + '.jpeg')) + \
            glob.glob(os.path.join(entry_path, name + '.jpg'))
        if not jpg_candidates:
            jpg_candidates = glob.glob(os.path.join(entry_path, '*.jpeg')) + \
                             glob.glob(os.path.join(entry_path, '*.jpg'))
        if not jpg_candidates:
            continue
        img_path = sorted(jpg_candidates)[0]

        # Boundary CSVs: path_0_0 through path_0_6
        csvs = []
        for b in range(7):
            pattern = os.path.join(
                entry_path,
                f'*path_0_{b}.csv')
            matches = glob.glob(pattern)
            if not matches:
                break
            csvs.append(sorted(matches)[0])

        if len(csvs) < 7:
            continue   # skip incomplete subjects

        subjects.append({
            'img_path': img_path,
            'csv_paths': csvs,
            'name': name,
        })
    return subjects


def get_split(subjects, split='train', val_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(subjects))
    rng.shuffle(indices)
    n_val = max(1, int(len(subjects) * val_ratio))
    if split == 'val':
        return [subjects[i] for i in indices[-n_val:]]
    return [subjects[i] for i in indices[:-n_val]]


class OCTIDDataset(Dataset):
    """
    OCTID retinal layer segmentation dataset.

    Reads pre-processed PNG images from ``processed_images/`` and integer
    PNG masks from ``processed_masks/``.  Both directories mirror the
    original subject names.

    Classes (8 total):
        0 – vitreous (above boundary 0)
        1-6 – retinal layers between consecutive boundaries
        7 – choroid / below boundary 6

    Args:
        data_dir (str): Root of the OCTID data tree (containing
                        ``processed_images/`` and ``processed_masks/``).
        split (str): 'train' or 'val'.
        img_size (int): Spatial size for resize.
        use_augmentations (bool): Albumentations pipeline on/off.
        val_ratio (float): Validation split fraction.
        seed (int): RNG seed.
    """

    def __init__(self, data_dir, split='train', img_size=256,
                 use_augmentations=False, val_ratio=0.2, seed=42):
        self.img_size = img_size
        self.use_augmentations = use_augmentations

        img_dir = os.path.join(data_dir, 'processed_images')
        msk_dir = os.path.join(data_dir, 'processed_masks')

        all_imgs = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith('.png')
        ])

        all_pairs = [(os.path.join(img_dir, f),
                      os.path.join(msk_dir, f)) for f in all_imgs
                     if os.path.isfile(os.path.join(msk_dir, f))]

        # Subject-level split using the filename stem
        names = sorted(set(f.rsplit('_', 1)[0] if '_' in f else f
                           for f, _ in all_pairs))
        rng = np.random.default_rng(seed)
        idx = np.arange(len(names))
        rng.shuffle(idx)
        n_val = max(1, int(len(names) * val_ratio))
        if split == 'val':
            chosen_names = {names[i] for i in idx[-n_val:]}
        else:
            chosen_names = {names[i] for i in idx[:-n_val]}

        self.pairs = [
            (ip, mp) for ip, mp in all_pairs
            if any(os.path.basename(ip).startswith(n) for n in chosen_names)
        ]

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
