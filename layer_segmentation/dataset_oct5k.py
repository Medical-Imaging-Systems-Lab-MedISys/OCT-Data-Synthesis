import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class OCT5kDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=224):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.img_size = img_size
        
        # RETFound expects 3-channel inputs (it was trained on both Color Fundus and OCT).
        # We duplicate the grayscale OCT to 3 channels and normalize.
        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Masks use Nearest Neighbor to preserve integer class labels
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=Image.NEAREST)
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name) # Assumes mask has the same filename

        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        image = self.img_transform(image)
        
        mask = self.mask_transform(mask)
        mask = torch.as_tensor(import_numpy_array(mask), dtype=torch.long)
        
        return image, mask

# Helper function
import numpy as np
def import_numpy_array(pil_image):
    return np.array(pil_image)