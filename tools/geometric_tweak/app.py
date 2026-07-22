import os
import cv2
import numpy as np
import socket
from flask import Flask, jsonify, request, send_file, render_template_string, Response
import base64
from io import BytesIO
from PIL import Image
import random

app = Flask(__name__)

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
DATA_DIR = os.path.join(REPO_ROOT, "DATA/NR206/train")
MASK_DIR = os.path.join(REPO_ROOT, "DATA/NR206/train_labels")

def find_free_port(start_port=3003):
    port = start_port
    while port < 65535:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            port += 1
    return start_port

def crop_and_pad_curved(image, mask_bgra, orig_image=None):
    H, W = image.shape[:2]
    is_bg = (mask_bgra[:, :, 0] == 0) & (mask_bgra[:, :, 1] == 0) & (mask_bgra[:, :, 2] == 0)
    is_retina = ~is_bg
    has_retina = np.any(is_retina, axis=0)
    b8 = np.full(W, H - 1, dtype=np.int32)
    if np.any(has_retina):
        b8[has_retina] = H - 1 - np.argmax(is_retina[::-1, :][:, has_retina], axis=0)
    
    b8 = np.clip(b8 + 3, 0, H - 1)
    max_y = np.max(b8[has_retina]) if np.any(has_retina) else H
    max_y = min(H, max_y + 5)
    
    cropped_h = max_y
    max_dim = max(cropped_h, W)
    pad_h = max_dim - cropped_h
    pad_w = max_dim - W
    
    safe_bottom = H - 20
    safe_top = max(0, safe_bottom - 50)
    
    if orig_image is not None:
        if orig_image.shape[:2] != (H, W):
            orig_resized = cv2.resize(orig_image, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            orig_resized = orig_image
        bottom_patch = orig_resized[safe_top:safe_bottom]
    else:
        bottom_patch = image[safe_top:safe_bottom]
        
    patch_height = bottom_patch.shape[0]
    
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
            
    y_coords = np.arange(max_dim)[:, None]
    keep_mask = y_coords <= b8[None, :]
    if pad_w > 0:
        keep_mask = np.pad(keep_mask, ((0, 0), (0, pad_w)), mode='constant', constant_values=False)
        
    cropped_img = image[:cropped_h]
    if len(image.shape) == 3:
        padded_img = np.pad(cropped_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        keep_mask_3d = np.expand_dims(keep_mask, axis=-1)
        final_image = np.where(keep_mask_3d, padded_img, tiled_bg)
    else:
        padded_img = np.pad(cropped_img, ((0, pad_h), (0, pad_w)), mode='constant')
        final_image = np.where(keep_mask, padded_img, tiled_bg)
        
    return final_image

def get_base64_img(img_array):
    if len(img_array.shape) == 2:
        img_pil = Image.fromarray(img_array.astype(np.uint8), mode='L')
    elif img_array.shape[2] == 4:
        img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGBA))
    else:
        img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    
    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def load_sample(filename):
    img_path = os.path.join(DATA_DIR, filename)
    mask_path = os.path.join(MASK_DIR, filename)
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Dynamically remove watermark from bottom-left before geometric modification
    if img is not None and img.shape[0] >= 350 and img.shape[1] >= 600:
        clean_patch = img[350:, 600:]
        if clean_patch.size > 0:
            h_target = img[350:, :150].shape[0]
            w_target = img[350:, :150].shape[1]
            img[350:, :150] = np.flip(clean_patch, axis=1)[:h_target, :w_target]
            
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    
    if mask is not None and (len(mask.shape) == 2 or mask.shape[2] == 3):
        # ensure mask is BGRA
        if len(mask.shape) == 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
        elif mask.shape[2] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2BGRA)
            
    return img, mask

def find_default_center(mask):
    """
    Finds the center of the fovea by identifying where the first retinal layer (Red, [0, 0, 255])
    reaches its minimum thickness (very thin middle position).
    """
    H, W = mask.shape[:2]
    # Red (ILM) layer is BGR color [0, 0, 255]
    red_mask = (mask[:, :, 0] == 0) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 255)
    thickness = np.sum(red_mask, axis=0) # shape (W,)
    
    # Smooth thickness values using a moving average window to eliminate noise/discretization spikes
    window_size = 21
    if W > window_size:
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(thickness, kernel, mode='same')
    else:
        smoothed = thickness
        
    # Search within the middle 60% of the image to avoid edge/boundary artifacts
    start = int(0.2 * W)
    end = int(0.8 * W)
    
    if start < end:
        min_idx = start + np.argmin(smoothed[start:end])
    else:
        min_idx = W // 2
        
    return float(min_idx) / W

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/random")
def get_random():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(('.png', '.jpg'))]
    filename = random.choice(files)
    
    _, mask = load_sample(filename)
    default_center = 0.5
    if mask is not None:
        default_center = find_default_center(mask)
        
    return jsonify({
        "filename": filename,
        "default_center": default_center
    })

@app.route("/api/batch_list")
def get_batch_list():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(('.png', '.jpg'))]
    if len(files) < 10:
        selected_files = files
    else:
        selected_files = random.sample(files, 10)
        
    batch_data = []
    for filename in selected_files:
        # Load mask to get the natural fovea center for this image
        _, mask = load_sample(filename)
        c_fovea = 0.5
        if mask is not None:
            c_fovea = find_default_center(mask)
            
        # Generate 10 randomized parameters per file
        variants = []
        for _ in range(10):
            amp = random.uniform(40, 150)
            # Center constrained to fovea center +/- 5% relative width
            center = float(np.clip(c_fovea + random.uniform(-0.05, 0.05), 0.10, 0.90))
            # Increase bend width (40% to 80%) to make the bends smoother (less steep)
            width = random.uniform(0.40, 0.80)
            tilt = random.uniform(-35, 35)
            variants.append({
                "amplitude": round(amp, 2),
                "center": round(center, 4),
                "width": round(width, 4),
                "tilt": round(tilt, 2)
            })
        batch_data.append({
            "filename": filename,
            "variants": variants
        })
        
    return jsonify({"batch": batch_data})

@app.route("/api/render_augment")
def render_augment():
    filename = request.args.get('filename')
    amplitude = float(request.args.get('amplitude', 95.0))
    center = float(request.args.get('center', 0.5))
    width = float(request.args.get('width', 0.4))
    tilt = float(request.args.get('tilt', 0.0))
    
    img, mask = load_sample(filename)
    if img is None or mask is None:
        return "Failed to load sample", 400
        
    H, W = img.shape[:2]
    
    # 1. Apply column shift (bending + tilt)
    shifted_img = np.zeros_like(img)
    shifted_mask = np.zeros_like(mask)
    if len(mask.shape) == 3 and mask.shape[2] == 4:
        shifted_mask[:,:,3] = 255 # initialize alpha
        
    x = np.arange(W)
    center_px = center * W
    width_px = width * W
    
    dy_bend = amplitude * np.exp(-((x - center_px)**2) / (2 * (width_px**2) + 1e-6))
    dy_tilt = tilt * (x - W/2) / (W/2)
    dy = dy_bend + dy_tilt
    dy = np.round(dy).astype(int)
    
    for i in range(W):
        shift = dy[i]
        if shift > 0:
            shifted_img[shift:, i] = img[:-shift, i]
            shifted_mask[shift:, i] = mask[:-shift, i]
        elif shift < 0:
            shifted_img[:shift, i] = img[-shift:, i]
            shifted_mask[:shift, i] = mask[-shift:, i]
        else:
            shifted_img[:, i] = img[:, i]
            shifted_mask[:, i] = mask[:, i]
            
    is_bg_shifted = (shifted_mask[:,:,0] == 0) & (shifted_mask[:,:,1] == 0) & (shifted_mask[:,:,2] == 0)
    is_ret_shifted = ~is_bg_shifted
    has_ret_shifted = np.any(is_ret_shifted, axis=0)
    
    max_dy = int(np.max(dy))
    min_dy = int(np.min(dy))
    
    if np.any(has_ret_shifted):
        top_y_per_col = np.argmax(is_ret_shifted, axis=0)
        min_top_y = np.min(top_y_per_col[has_ret_shifted])
        top_crop = min(max_dy, max(0, min_top_y - 5))
    else:
        top_crop = max(0, max_dy)
        
    bottom_crop = max(0, -min_dy)
    
    if top_crop + bottom_crop < shifted_img.shape[0]:
        shifted_img = shifted_img[top_crop:shifted_img.shape[0] - bottom_crop, :]
        shifted_mask = shifted_mask[top_crop:shifted_mask.shape[0] - bottom_crop, :]
        
    target_size = (256, 256)
    img_squashed = cv2.resize(shifted_img, target_size, interpolation=cv2.INTER_LINEAR)
    mask_squashed = cv2.resize(shifted_mask, target_size, interpolation=cv2.INTER_NEAREST)
    
    orig_img_256 = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    final_img = crop_and_pad_curved(img_squashed, mask_squashed, orig_image=orig_img_256)
    
    _, encoded_img = cv2.imencode('.png', final_img)
    return Response(encoded_img.tobytes(), mimetype='image/png')

@app.route("/api/process", methods=['POST'])
def process_image():
    data = request.json
    filename = data.get('filename')
    amplitude = float(data.get('amplitude', 0.0))
    center = float(data.get('center', 0.5)) # relative to width
    width = float(data.get('width', 0.2)) # relative to width
    tilt = float(data.get('tilt', 0.0))
    
    img, mask = load_sample(filename)
    if img is None or mask is None:
        return jsonify({"error": "Failed to load sample"}), 400
        
    H, W = img.shape[:2]
    
    # 1. Apply column shift (bending + tilt)
    shifted_img = np.zeros_like(img)
    shifted_mask = np.zeros_like(mask)
    if len(mask.shape) == 3 and mask.shape[2] == 4:
        shifted_mask[:,:,3] = 255 # initialize alpha
        
    x = np.arange(W)
    center_px = center * W
    width_px = width * W
    
    # Gaussian bend
    dy_bend = amplitude * np.exp(-((x - center_px)**2) / (2 * (width_px**2) + 1e-6))
    # Linear tilt (pivot around center of the image W/2)
    dy_tilt = tilt * (x - W/2) / (W/2)
    
    dy = dy_bend + dy_tilt
    dy = np.round(dy).astype(int)
    
    for i in range(W):
        shift = dy[i]
        if shift > 0:
            shifted_img[shift:, i] = img[:-shift, i]
            shifted_mask[shift:, i] = mask[:-shift, i]
        elif shift < 0:
            shifted_img[:shift, i] = img[-shift:, i]
            shifted_mask[:shift, i] = mask[-shift:, i]
        else:
            shifted_img[:, i] = img[:, i]
            shifted_mask[:, i] = mask[:, i]
            
    # Crop ONLY the pure black zero-padded bands created by the shift.
    # We crop top_crop from the top (removing downward shift blackness)
    # and bottom_crop from the bottom (removing upward shift blackness).
    is_bg_shifted = (shifted_mask[:,:,0] == 0) & (shifted_mask[:,:,1] == 0) & (shifted_mask[:,:,2] == 0)
    is_ret_shifted = ~is_bg_shifted
    has_ret_shifted = np.any(is_ret_shifted, axis=0)
    
    max_dy = int(np.max(dy))
    min_dy = int(np.min(dy))
    
    if np.any(has_ret_shifted):
        top_y_per_col = np.argmax(is_ret_shifted, axis=0)
        min_top_y = np.min(top_y_per_col[has_ret_shifted])
        # Ensure we don't slice into the retina even if amplitude/tilt is large
        top_crop = min(max_dy, max(0, min_top_y - 5))
    else:
        top_crop = max(0, max_dy)
        
    bottom_crop = max(0, -min_dy)
        
    if top_crop + bottom_crop < shifted_img.shape[0]:
        shifted_img = shifted_img[top_crop:shifted_img.shape[0] - bottom_crop, :]
        shifted_mask = shifted_mask[top_crop:shifted_mask.shape[0] - bottom_crop, :]
            
    # 2. Resize to 256x256 then Crop and pad curved as in CFM training
    target_size = (256, 256)
    
    img_squashed = cv2.resize(shifted_img, target_size, interpolation=cv2.INTER_LINEAR)
    mask_squashed = cv2.resize(shifted_mask, target_size, interpolation=cv2.INTER_NEAREST)
    
    # Resize original watermark-cleaned image to match H, W for bottom patch extraction
    orig_img_256 = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    
    final_img = crop_and_pad_curved(img_squashed, mask_squashed, orig_image=orig_img_256)
    final_mask = crop_and_pad_curved(mask_squashed, mask_squashed)
    
    return jsonify({
        "original_img": get_base64_img(img),
        "shifted_img": get_base64_img(shifted_img),
        "shifted_mask": get_base64_img(shifted_mask),
        "final_img": get_base64_img(final_img),
        "final_mask": get_base64_img(final_mask)
    })

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OCT Geometric Tweak Studio</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #0f172a;
            --bg-panel: rgba(30, 41, 59, 0.7);
            --border-color: rgba(148, 163, 184, 0.15);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #3b82f6;
            --accent-hover: #60a5fa;
            --panel-blur: blur(16px);
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: 'Inter', sans-serif;
            background: radial-gradient(circle at top right, #1e1b4b, var(--bg-base) 40%);
            color: var(--text-main);
            min-height: 100vh;
            padding: 2rem;
            display: flex;
            flex-direction: column;
            gap: 2rem;
        }

        .header {
            text-align: center;
            padding: 2rem 0;
            background: var(--bg-panel);
            border-radius: 20px;
            border: 1px solid var(--border-color);
            backdrop-filter: var(--panel-blur);
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            animation: fadeInDown 0.6s ease-out;
        }
        
        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(to right, #60a5fa, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Tabs Navigation styling */
        .tabs-header {
            display: flex;
            gap: 1rem;
            justify-content: center;
            margin: 0 auto;
            max-width: 1600px;
            width: 100%;
        }
        
        .tab-btn {
            background: rgba(255,255,255,0.05);
            border: 1px solid var(--border-color);
            color: var(--text-muted);
            padding: 0.75rem 1.5rem;
            border-radius: 12px;
            cursor: pointer;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.2s;
            backdrop-filter: var(--panel-blur);
        }
        
        .tab-btn:hover {
            background: rgba(255,255,255,0.1);
            color: var(--text-main);
            transform: translateY(-1px);
        }
        
        .tab-btn.active {
            background: var(--accent);
            color: white;
            border-color: var(--accent);
            box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.3);
        }

        .container {
            display: grid;
            grid-template-columns: 350px 1fr;
            gap: 2rem;
            max-width: 1600px;
            margin: 0 auto;
            width: 100%;
        }

        .controls {
            background: var(--bg-panel);
            border-radius: 20px;
            border: 1px solid var(--border-color);
            backdrop-filter: var(--panel-blur);
            padding: 2rem;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            height: fit-content;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        .control-group { display: flex; flex-direction: column; gap: 0.5rem; }
        
        label {
            font-weight: 600;
            font-size: 0.9rem;
            color: var(--text-muted);
            display: flex;
            justify-content: space-between;
        }
        
        input[type="range"] {
            -webkit-appearance: none;
            width: 100%;
            height: 6px;
            background: rgba(255,255,255,0.1);
            border-radius: 3px;
            outline: none;
        }
        
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: var(--accent);
            cursor: pointer;
            transition: transform 0.1s;
        }
        
        input[type="range"]::-webkit-slider-thumb:hover { transform: scale(1.2); }
        
        .btn {
            background: var(--accent);
            color: white;
            border: none;
            padding: 1rem;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }
        
        .btn:hover { background: var(--accent-hover); transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.3); }

        .preview-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1.5rem;
        }

        .image-card {
            background: var(--bg-panel);
            border-radius: 20px;
            border: 1px solid var(--border-color);
            backdrop-filter: var(--panel-blur);
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
            align-items: center;
            transition: transform 0.3s;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }
        
        .image-card:hover { transform: translateY(-5px); }
        
        .image-card h3 { font-size: 1.1rem; color: var(--text-main); font-weight: 600; text-align: center; }
        
        .img-wrapper {
            position: relative;
            width: 100%;
            aspect-ratio: 4/3;
            background: #000;
            border-radius: 12px;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .img-wrapper img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            transition: opacity 0.3s;
        }
        
        .overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain; opacity: 0.5; mix-blend-mode: screen; pointer-events: none;}
        
        .loader {
            border: 3px solid rgba(255,255,255,0.1);
            border-top: 3px solid var(--accent);
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            display: none;
            position: absolute;
        }

        /* Batch Augmentation CSS */
        .batch-row {
            background: var(--bg-panel);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            backdrop-filter: var(--panel-blur);
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.3);
        }
        
        .batch-row-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.75rem;
        }
        
        .batch-row-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: var(--accent-hover);
        }
        
        .batch-scroller {
            display: flex;
            gap: 1.25rem;
            overflow-x: auto;
            padding: 0.5rem 0 1.25rem 0;
            scrollbar-width: thin;
            scrollbar-color: rgba(255,255,255,0.2) transparent;
        }
        
        .batch-scroller::-webkit-scrollbar {
            height: 6px;
        }
        
        .batch-scroller::-webkit-scrollbar-thumb {
            background: rgba(255,255,255,0.25);
            border-radius: 3px;
        }
        
        .batch-card {
            min-width: 220px;
            width: 220px;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 0.75rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            transition: transform 0.2s, border-color 0.2s;
        }
        
        .batch-card:hover {
            transform: translateY(-3px);
            border-color: rgba(255,255,255,0.3);
            box-shadow: 0 10px 20px rgba(0,0,0,0.5);
        }
        
        .batch-card img {
            width: 100%;
            aspect-ratio: 1;
            object-fit: cover;
            border-radius: 8px;
            background: #000;
        }
        
        .batch-card-info {
            font-size: 0.8rem;
            color: var(--text-muted);
            display: flex;
            flex-direction: column;
            gap: 0.3rem;
        }
        
        .batch-card-info span {
            display: flex;
            justify-content: space-between;
        }
        
        .batch-card-info strong {
            color: var(--text-main);
        }

        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        @keyframes fadeInDown { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }

    </style>
</head>
<body>
    <div class="header">
        <h1>OCT Geometric Tweak Studio</h1>
        <p style="color: var(--text-muted); margin-top: 0.5rem;" id="filename-display">Interactive Planning System</p>
    </div>

    <!-- Tabs Navigation -->
    <div class="tabs-header">
        <button class="tab-btn active" onclick="switchTab('studio')">Interactive Tweak Studio</button>
        <button class="tab-btn" onclick="switchTab('gallery')">Batch Augmentation Gallery (10x10)</button>
    </div>

    <!-- Interactive Studio Tab -->
    <div id="tab-studio" class="tab-content">
        <div class="container">
            <div class="controls">
                <button class="btn" onclick="loadRandom()">
                    <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg>
                    Load Random Sample
                </button>
                
                <hr style="border-color: var(--border-color); margin: 1rem 0;">
                
                <div class="control-group">
                    <label>Bend Amplitude (Pixels) <span id="val-amp">0</span></label>
                    <input type="range" id="amp" min="0" max="200" value="0" oninput="updateVal('amp'); scheduleUpdate()">
                </div>
                
                <div class="control-group">
                    <label>Bend Center (%) <span id="val-center">50</span></label>
                    <input type="range" id="center" min="10" max="90" value="50" oninput="updateVal('center'); scheduleUpdate()">
                </div>
                
                <div class="control-group">
                    <label>Bend Width <span id="val-width">40</span></label>
                    <input type="range" id="width" min="5" max="100" value="40" oninput="updateVal('width'); scheduleUpdate()">
                </div>
                
                <div class="control-group">
                    <label>Tilt (Pixels) <span id="val-tilt">0</span></label>
                    <input type="range" id="tilt" min="-100" max="100" value="0" oninput="updateVal('tilt'); scheduleUpdate()">
                </div>
            </div>

            <div class="preview-grid">
                <div class="image-card">
                    <h3>Original Image</h3>
                    <div class="img-wrapper">
                        <img id="img-orig" src="">
                        <div class="loader" id="loader-orig"></div>
                    </div>
                </div>
                
                <div class="image-card">
                    <h3>Shifted (Constant Width)</h3>
                    <div class="img-wrapper">
                        <img id="img-shift" src="">
                        <img class="overlay" id="img-shift-mask" src="">
                        <div class="loader" id="loader-shift"></div>
                    </div>
                </div>
                
                <div class="image-card">
                    <h3>Final Cropped (CFM Input)</h3>
                    <div class="img-wrapper" style="aspect-ratio: 1; background: #000;">
                        <img id="img-final" src="" style="width: 100%; height: 100%;">
                        <div class="loader" id="loader-final"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Batch Augmentation Gallery Tab -->
    <div id="tab-gallery" class="tab-content" style="display: none; max-width: 1600px; margin: 0 auto; width: 100%;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
            <div>
                <h2 style="font-size: 1.5rem; font-weight: 700;">10x10 Randomized Augmentation Grid</h2>
                <p style="color: var(--text-muted); font-size: 0.9rem; margin-top: 0.25rem;">
                    Generating 10 augmented variants per sample. Ranges: Amplitude 40-150 | Center: Fovea Center ±5% | Width 40%-80% | Tilt -35 to +35.
                </p>
            </div>
            <button class="btn" onclick="loadBatch()">
                <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg>
                Regenerate Batch
            </button>
        </div>
        <div id="batch-container" style="display: flex; flex-direction: column; gap: 2.5rem;">
            <!-- Populated dynamically via JS -->
        </div>
    </div>

    <script>
        let currentFilename = "";
        let updateTimeout = null;

        function updateVal(id) {
            document.getElementById('val-' + id).innerText = document.getElementById(id).value;
        }

        function switchTab(tabId) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.style.display = 'none');
            
            if (tabId === 'studio') {
                document.querySelector('.tab-btn[onclick="switchTab(\\'studio\\')"]').classList.add('active');
                document.getElementById('tab-studio').style.display = 'block';
                if(currentFilename) {
                    document.getElementById('filename-display').innerText = currentFilename;
                } else {
                    document.getElementById('filename-display').innerText = "Interactive Planning System";
                }
            } else {
                document.querySelector('.tab-btn[onclick="switchTab(\\'gallery\\')"]').classList.add('active');
                document.getElementById('tab-gallery').style.display = 'block';
                document.getElementById('filename-display').innerText = "Batch Augmentation Previewer";
                if (document.getElementById('batch-container').children.length === 0) {
                    loadBatch();
                }
            }
        }

        async function loadBatch() {
            const container = document.getElementById('batch-container');
            container.innerHTML = `
                <div style="text-align: center; padding: 5rem 0; color: var(--text-muted);">
                    <div class="loader" style="display: inline-block; position: relative; width: 50px; height: 50px; border-width: 4px; margin-bottom: 1.5rem;"></div>
                    <p style="font-size: 1.1rem; font-weight: 500;">Generating 100 augmented samples on-the-fly...</p>
                </div>
            `;
            
            // show the inline loader
            container.querySelector('.loader').style.display = 'inline-block';
            
            try {
                const res = await fetch('/api/batch_list');
                const data = await res.json();
                
                container.innerHTML = '';
                data.batch.forEach((sample, idx) => {
                    const row = document.createElement('div');
                    row.className = 'batch-row';
                    
                    const header = document.createElement('div');
                    header.className = 'batch-row-header';
                    header.innerHTML = `
                        <div class="batch-row-title">Sample ${idx + 1}: ${sample.filename}</div>
                        <div style="font-size: 0.9rem; color: var(--text-muted);">Original Size: 256x256 CFM Input</div>
                    `;
                    row.appendChild(header);
                    
                    const scroller = document.createElement('div');
                    scroller.className = 'batch-scroller';
                    
                    sample.variants.forEach((v, vIdx) => {
                        const card = document.createElement('div');
                        card.className = 'batch-card';
                        
                        const imgUrl = `/api/render_augment?filename=${encodeURIComponent(sample.filename)}&amplitude=${v.amplitude}&center=${v.center}&width=${v.width}&tilt=${v.tilt}`;
                        
                        card.innerHTML = `
                            <img src="${imgUrl}" loading="lazy" alt="Augmented Variant">
                            <div class="batch-card-info">
                                <span><strong>Variant #${vIdx + 1}</strong></span>
                                <span>Amp: <strong>${v.amplitude}px</strong></span>
                                <span>Center: <strong>${Math.round(v.center * 100)}%</strong></span>
                                <span>Width: <strong>${Math.round(v.width * 100)}%</strong></span>
                                <span>Tilt: <strong>${v.tilt}px</strong></span>
                            </div>
                        `;
                        scroller.appendChild(card);
                    });
                    
                    row.appendChild(scroller);
                    container.appendChild(row);
                });
            } catch (err) {
                container.innerHTML = `<p style="color: #ef4444; text-align: center; padding: 3rem; font-weight: 600;">Error generating batch: ${err}</p>`;
            }
        }

        async function loadRandom() {
            const res = await fetch('/api/random');
            const data = await res.json();
            currentFilename = data.filename;
            document.getElementById('filename-display').innerText = currentFilename;
            
            // reset sliders
            document.getElementById('amp').value = 0;
            updateVal('amp');
            document.getElementById('tilt').value = 0;
            updateVal('tilt');
            document.getElementById('width').value = 40;
            updateVal('width');
            
            // Set bend center slider to calculated default center from backend fovea detector
            const centerVal = Math.round(data.default_center * 100);
            document.getElementById('center').value = centerVal;
            updateVal('center');
            
            updatePreview();
        }

        function scheduleUpdate() {
            if(updateTimeout) clearTimeout(updateTimeout);
            updateTimeout = setTimeout(updatePreview, 300); // 300ms debounce
        }

        async function updatePreview() {
            if(!currentFilename) return;
            
            ['orig', 'shift', 'final'].forEach(l => document.getElementById(`loader-${l}`).style.display = 'block');
            
            const payload = {
                filename: currentFilename,
                amplitude: parseFloat(document.getElementById('amp').value),
                center: parseFloat(document.getElementById('center').value) / 100.0,
                width: parseFloat(document.getElementById('width').value) / 100.0,
                tilt: parseFloat(document.getElementById('tilt').value)
            };
            
            const res = await fetch('/api/process', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            
            if(!data.error) {
                document.getElementById('img-orig').src = 'data:image/png;base64,' + data.original_img;
                document.getElementById('img-shift').src = 'data:image/png;base64,' + data.shifted_img;
                document.getElementById('img-shift-mask').src = 'data:image/png;base64,' + data.shifted_mask;
                document.getElementById('img-final').src = 'data:image/png;base64,' + data.final_img;
            }
            
            ['orig', 'shift', 'final'].forEach(l => document.getElementById(`loader-${l}`).style.display = 'none');
        }

        // Init
        switchTab('studio');
        loadRandom();
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    port = find_free_port(3001)
    print(f"Starting Geometric Tweak App on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
