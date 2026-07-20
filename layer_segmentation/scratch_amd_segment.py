import os
import torch
import numpy as np
import cv2
from PIL import Image
import mlflow
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Add layer_segmentation to path to import model
import sys
sys.path.append('/data/vds/mmk/Codes/oct_data_synthesis/layer_segmentation')
from model import RETFoundSegmenter
from segment_new_datasets import preprocess_image, colorize_mask_nr206

def main():
    img_path = '/data/vds/mmk/Codes/oct_data_synthesis/DATA/AMD-SD/images/126/126_19.png'
    out_path = '/data/vds/mmk/Codes/oct_data_synthesis/layer_segmentation/amd_126_19_segmented.png'
    
    print("Loading image...")
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read {img_path}")
        
    h, w, c = img.shape
    half = w // 2
    left = img[:, :half, :]
    right = img[:, half:, :]
    
    # Convert left to PIL RGB for preprocessing
    left_pil = Image.fromarray(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
    
    print("Setting up MLflow and Model...")
    mlflow.set_tracking_uri('http://10.24.38.15:5000')
    client = mlflow.tracking.MlflowClient()
    run_id = "a8d99dbe233442e48fc391c7b02c5b74"
    
    artifacts = client.list_artifacts(run_id, "checkpoints")
    pth_files = [art.path for art in artifacts if art.path.endswith('.pth')]
    checkpoint_path = pth_files[0]
    local_ckpt_path = client.download_artifacts(run_id, checkpoint_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RETFoundSegmenter(num_classes=9, img_size=256)
    model.load_state_dict(torch.load(local_ckpt_path, map_location='cpu'))
    model = model.to(device)
    model.eval()
    
    print("Segmenting...")
    img_tensor, orig_w, orig_h = preprocess_image(left_pil, 256)
    with torch.no_grad():
        outputs = model(img_tensor.unsqueeze(0).to(device))
        preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        preds_resized = cv2.resize(preds, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
    pred_color = colorize_mask_nr206(preds_resized)
    
    print("Plotting results...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input OCT (Left Half)")
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(pred_color, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Predicted Mask (NR206 model)")
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(right, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Original GT (Right Half)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Done! Saved to {out_path}")

if __name__ == '__main__':
    main()
