import os
import cv2
import numpy as np
from PIL import Image
import sys

# Import functions from the conditional-flow-matching script
sys.path.append(os.path.join(os.path.dirname(__file__), "conditional-flow-matching"))
from train_val_2_cropped import synthesize_from_mask, crop_and_pad_curved

# -------------------------------------------------------------------
# Global Seeding for Reproducibility
# -------------------------------------------------------------------
import random
import numpy as np
import torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
try:
    import pytorch_lightning as pl
    pl.seed_everything(42, workers=True)
except ImportError:
    pass
# -------------------------------------------------------------------


def main():
    labels_dir = "NR206/test_labels"
    output_dir = "NR206/val_priors"
    img_size = 256

    os.makedirs(output_dir, exist_ok=True)
    
    # Get all mask filenames
    filenames = sorted([f for f in os.listdir(labels_dir) if f.endswith('.png')])
    
    print(f"Found {len(filenames)} validation masks. Generating priors...")
    
    for filename in filenames:
        mask_path = os.path.join(labels_dir, filename)
        
        # Load mask in BGRA
        mask_bgra = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask_bgra is None:
            print(f"Warning: Could not read {mask_path}")
            continue
            
        # Ensure it has 4 channels
        if mask_bgra.shape[2] == 3:
            mask_bgra = cv2.cvtColor(mask_bgra, cv2.COLOR_BGR2BGRA)
            
        # Synthesize with fixed gamma (e.g. 1.0) to get a clean prior
        synth_np = synthesize_from_mask(mask_bgra, min_gamma=1.0, max_gamma=1.0)
        
        # Squash to 256x256
        synth_np_squashed = cv2.resize(synth_np, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        mask_squashed = cv2.resize(mask_bgra, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        
        # Apply curved crop and pad
        synth_np_padded = crop_and_pad_curved(synth_np_squashed, mask_squashed)
        
        # Save output
        out_path = os.path.join(output_dir, filename)
        synth_img = Image.fromarray(synth_np_padded, mode='L')
        synth_img.save(out_path)
        
    print(f"Successfully generated {len(filenames)} validation priors in {output_dir}")

if __name__ == "__main__":
    main()
