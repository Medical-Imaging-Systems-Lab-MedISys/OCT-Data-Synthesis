import os
import cv2
import numpy as np
import base64
from flask import Flask, render_template_string

app = Flask(__name__)

def crop_and_pad_curved(image, mask_bgra):
    H, W = image.shape[:2]
    
    # 1. Find b8 curve (bottom of retina)
    is_bg = (mask_bgra[:, :, 0] == 0) & (mask_bgra[:, :, 1] == 0) & (mask_bgra[:, :, 2] == 0)
    is_retina = ~is_bg
    has_retina = np.any(is_retina, axis=0)
    b8 = np.full(W, H - 1, dtype=np.int32)
    if np.any(has_retina):
        b8[has_retina] = H - 1 - np.argmax(is_retina[::-1, :][:, has_retina], axis=0)
    
    # Add a tiny margin so we don't cut into the actual color
    b8 = np.clip(b8 + 3, 0, H - 1)
    
    # We want to crop the image height to max(b8) + some margin
    max_y = np.max(b8[has_retina]) if np.any(has_retina) else H
    max_y = min(H, max_y + 5)
    
    cropped_h = max_y
    max_dim = max(cropped_h, W)
    pad_h = max_dim - cropped_h
    pad_w = max_dim - W
    
    # 2. Extract safe noise patch from original image BEFORE cropping
    safe_bottom = H - 20
    safe_top = max(0, safe_bottom - 50)
    bottom_patch = image[safe_top:safe_bottom]
    patch_height = bottom_patch.shape[0]
    
    # 3. Build a full max_dim x max_dim background of tiled noise
    tiles_needed = int(np.ceil(max_dim / patch_height)) if patch_height > 0 else 1
    tiles = []
    for i in range(tiles_needed):
        shift = np.random.randint(0, W) if W > 0 else 0
        shifted = np.roll(bottom_patch, shift, axis=1)
        if i % 2 == 1:
            shifted = np.flip(shifted, axis=0)
        tiles.append(shifted)
        
    tiled_bg = np.concatenate(tiles, axis=0)[:max_dim, :W]
    if pad_w > 0:
        if len(image.shape) == 3:
            tiled_bg = np.pad(tiled_bg, ((0, 0), (0, pad_w), (0, 0)), mode='symmetric')
        else:
            tiled_bg = np.pad(tiled_bg, ((0, 0), (0, pad_w)), mode='symmetric')
            
    # 4. Create keep mask for the curved retina region
    y_coords = np.arange(max_dim)[:, None]
    keep_mask = y_coords <= b8[None, :]
    if pad_w > 0:
        keep_mask = np.pad(keep_mask, ((0, 0), (0, pad_w)), mode='constant', constant_values=False)
        
    # 5. Composite: copy original image over the tiled background
    cropped_img = image[:cropped_h]
    if len(image.shape) == 3:
        padded_img = np.pad(cropped_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        keep_mask_3d = np.expand_dims(keep_mask, axis=-1)
        final_image = np.where(keep_mask_3d, padded_img, tiled_bg)
    else:
        padded_img = np.pad(cropped_img, ((0, pad_h), (0, pad_w)), mode='constant')
        final_image = np.where(keep_mask, padded_img, tiled_bg)
        
    return final_image

def image_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/')
def index():
    data_dir = "./NR206/train"
    labels_dir = "./NR206/train_labels"
    
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.png')])[:5]
    
    html = """
    <html><head>
    <style>
        body { font-family: 'Inter', sans-serif; background: #121212; color: #fff; margin: 0; padding: 20px; }
        h1 { text-align: center; color: #00e5ff; }
        .container { display: flex; flex-direction: column; gap: 30px; }
        .row { display: flex; gap: 20px; background: #1e1e1e; padding: 20px; border-radius: 12px; }
        .col { flex: 1; text-align: center; }
        img { max-width: 100%; border: 1px solid #333; border-radius: 8px; }
    </style>
    </head><body>
    <h1>OCT Cropping & Padding Previews</h1>
    <div class="container">
    """
    
    for f in files:
        real_path = os.path.join(data_dir, f)
        lbl_path = os.path.join(labels_dir, f)
        
        real_img = cv2.imread(real_path, cv2.IMREAD_GRAYSCALE)
        mask_bgra = cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED)
        
        # Remove watermark
        clean_patch = real_img[350:, 600:]
        real_img[350:, :150] = np.flip(clean_patch, axis=1)
        
        if len(mask_bgra.shape) == 3 and mask_bgra.shape[2] == 3:
            alpha = np.full((mask_bgra.shape[0], mask_bgra.shape[1], 1), 255, dtype=np.uint8)
            mask_bgra = np.concatenate([mask_bgra, alpha], axis=2)
            
        # BEFORE RESIZING (Correct method)
        real_padded = crop_and_pad_curved(real_img, mask_bgra)
        
        # Resize to 256 for display to match training input
        real_resized = cv2.resize(real_padded, (256, 256), interpolation=cv2.INTER_LINEAR)
        
        # AFTER RESIZING (Squashed method)
        real_squashed = cv2.resize(real_img, (256, 256), interpolation=cv2.INTER_LINEAR)
        mask_squashed = cv2.resize(mask_bgra, (256, 256), interpolation=cv2.INTER_LINEAR)
        real_after = crop_and_pad_curved(real_squashed, mask_squashed)
        
        html += f"""
        <div class="row">
            <div class="col">
                <h3>Original GT</h3>
                <img src="data:image/png;base64,{image_to_base64(real_img)}">
            </div>
            <div class="col">
                <h3>Cropped BEFORE Resizing</h3>
                <p>Aspect ratio preserved. Natural speckle.</p>
                <img src="data:image/png;base64,{image_to_base64(real_resized)}">
            </div>
            <div class="col">
                <h3>Cropped AFTER Resizing</h3>
                <p>Squashed aspect ratio. Mismatched padding noise.</p>
                <img src="data:image/png;base64,{image_to_base64(real_after)}">
            </div>
        </div>
        """
    
    html += "</div></body></html>"
    return render_template_string(html)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
