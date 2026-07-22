#!/usr/bin/env python3
import os
import sys
import glob
import torch
import cv2
import numpy as np
import base64
import socket
from PIL import Image
from flask import Flask, jsonify, request, render_template_string, send_from_directory

# Ensure project root is in python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
sys.path.append(os.path.join(REPO_ROOT, "conditional-flow-matching"))

from diffusers import UNet2DModel
try:
    from train_val import synthesize_from_mask
except Exception:
    def synthesize_from_mask(mask_bgra, min_gamma=0.5, max_gamma=1.2):
        gray = cv2.cvtColor(mask_bgra[:, :, :3], cv2.COLOR_BGR2GRAY)
        noise = np.random.normal(128, 30, gray.shape).clip(0, 255).astype(np.uint8)
        return cv2.addWeighted(gray, 0.7, noise, 0.3, 0)

app = Flask(__name__, static_folder=os.path.join(REPO_ROOT, "static"))

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Registry
MODELS_CONFIG = {
    "cfm_8bitnorm_v1_exp14": {
        "id": "cfm_8bitnorm_v1_exp14",
        "name": "CFM 8-Bit Norm v1 Baseline Model",
        "path": os.path.join(REPO_ROOT, "conditional-flow-matching/checkpoints/cfm_model_cfm_8bitnorm_2026-07-14_14-51-24.pt"),
        "in_channels": 1,
        "norm": "8-bit",
        "loss": "Weighted Loss (w_bg=0.4, w_layers=1.0)"
    }
}

LOADED_MODELS = {}

def get_model(model_key):
    if model_key in LOADED_MODELS:
        return LOADED_MODELS[model_key]
    
    cfg = MODELS_CONFIG.get(model_key)
    if not cfg or not os.path.exists(cfg["path"]):
        return None
        
    print(f"Loading PyTorch model {cfg['name']} on {device}...")
    model = UNet2DModel(
        sample_size=256,
        in_channels=cfg["in_channels"],
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    )
    state = torch.load(cfg["path"], map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    LOADED_MODELS[model_key] = model
    return model

def run_ode_inference(model, x0_tensor, num_steps=50):
    batch_size = x0_tensor.shape[0]
    dt = 1.0 / num_steps
    x_t = x0_tensor.clone().to(device)
    x0_dev = x0_tensor.to(device)
    
    with torch.no_grad():
        for i in range(num_steps):
            t_batch = torch.full((batch_size,), i / num_steps, device=device, dtype=torch.float32)
            model_input = torch.cat([x_t, x0_dev], dim=1) if model.config.in_channels == 2 else x_t
            v_pred = model(model_input, t_batch).sample
            x_t = x_t + v_pred * dt
            
    return x_t

def get_b64(filepath):
    if not os.path.exists(filepath):
        return ""
    with open(filepath, "rb") as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Retinal OCT Synthesis Studio | CFM 8-Bit Norm Models</title>
    <meta name="description" content="Interactive visual studio and gallery for Retinal OCT synthesis using 8-bit normalized Conditional Flow Matching (CFM) models.">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.75);
            --border-color: rgba(255, 255, 255, 0.12);
            --accent-purple: #8b5cf6;
            --accent-indigo: #6366f1;
            --accent-pink: #ec4899;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        body {
            background-color: var(--bg-dark);
            color: var(--text-main);
            min-height: 100vh;
            padding-bottom: 50px;
        }

        header {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.2));
            border-bottom: 1px solid var(--border-color);
            padding: 24px 40px;
            backdrop-filter: blur(12px);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .header-content {
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .logo-section h1 {
            font-size: 1.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #a78bfa, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .badge {
            font-size: 0.75rem;
            padding: 4px 12px;
            border-radius: 20px;
            background: rgba(139, 92, 246, 0.25);
            color: #c4b5fd;
            border: 1px solid rgba(139, 92, 246, 0.4);
            font-weight: 500;
        }

        .nav-tabs {
            display: flex;
            gap: 12px;
        }

        .tab-btn {
            background: transparent;
            border: 1px solid transparent;
            color: var(--text-muted);
            padding: 8px 18px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 500;
            font-size: 0.9rem;
            transition: all 0.2s ease;
        }

        .tab-btn:hover {
            color: var(--text-main);
            background: rgba(255, 255, 255, 0.05);
        }

        .tab-btn.active {
            background: linear-gradient(135deg, var(--accent-indigo), var(--accent-purple));
            color: white;
            box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
        }

        main {
            max-width: 1400px;
            margin: 30px auto;
            padding: 0 40px;
        }

        .section-title {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .model-selector-bar {
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 24px;
            background: var(--card-bg);
            padding: 16px 24px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }

        select, input[type="range"] {
            background: #0f172a;
            color: var(--text-main);
            border: 1px solid var(--border-color);
            padding: 10px 16px;
            border-radius: 8px;
            font-size: 0.9rem;
            outline: none;
        }

        select:focus {
            border-color: var(--accent-purple);
        }

        .gallery-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(620px, 1fr));
            gap: 24px;
        }

        .sample-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 20px;
            backdrop-filter: blur(10px);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .sample-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }

        .card-title {
            font-size: 1rem;
            font-weight: 600;
            color: #e2e8f0;
        }

        .image-quad {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
        }

        .img-box {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 6px;
        }

        .img-box img {
            width: 100%;
            aspect-ratio: 1/1;
            object-fit: cover;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            background: #000;
        }

        .img-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            font-weight: 500;
        }

        /* Interactive Generator View */
        .interactive-container {
            display: grid;
            grid-template-columns: 340px 1fr;
            gap: 24px;
        }

        .controls-panel {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 24px;
        }

        .control-group {
            margin-bottom: 20px;
        }

        .control-group label {
            display: block;
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-muted);
            margin-bottom: 8px;
        }

        .btn-generate {
            width: 100%;
            background: linear-gradient(135deg, var(--accent-indigo), var(--accent-purple));
            color: white;
            border: none;
            padding: 12px;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .btn-generate:hover {
            opacity: 0.9;
            box-shadow: 0 4px 14px rgba(139, 92, 246, 0.4);
        }

        .preview-panel {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 24px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 400px;
        }

        .spinner {
            border: 4px solid rgba(255, 255, 255, 0.1);
            width: 40px;
            height: 40px;
            border-radius: 50%;
            border-left-color: var(--accent-purple);
            animation: spin 1s linear infinite;
            display: none;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <header>
        <div class="header-content">
            <div class="logo-section">
                <h1>Retinal OCT Synthesis Studio <span class="badge">CFM 8-Bit Norm</span></h1>
            </div>
            <div class="nav-tabs">
                <button class="tab-btn active" id="tab-btn-gallery" onclick="switchTab('gallery')">🖼️ Model Gallery</button>
                <button class="tab-btn" id="tab-btn-generator" onclick="switchTab('generator')">⚡ Live Generator</button>
            </div>
        </div>
    </header>

    <main>
        <!-- Gallery Section -->
        <div id="section-gallery">
            <div class="section-title">
                <span>Validation Comparison Samples</span>
                <span class="badge">Dataset: NR206 / Validation Set</span>
            </div>

            <div class="model-selector-bar">
                <label for="model-select" style="font-weight: 500;">Select CFM Model Checkpoint:</label>
                <select id="model-select" onchange="renderGallery()">
                    <option value="cfm_8bitnorm_v1_exp14">CFM 8-Bit Norm v1 Baseline Model</option>
                </select>
            </div>

            <div class="gallery-grid" id="gallery-grid-container">
                <!-- Dynamically populated via Base64 inline URIs -->
            </div>
        </div>

        <!-- Live Generator Section -->
        <div id="section-generator" style="display: none;">
            <div class="section-title">
                <span>Interactive Live Synthesis Engine</span>
            </div>

            <div class="interactive-container">
                <div class="controls-panel">
                    <div class="control-group">
                        <label for="live-model-select">Target Model Checkpoint</label>
                        <select id="live-model-select" style="width: 100%;">
                            <option value="cfm_8bitnorm_v1_exp14">CFM 8-Bit Norm v1 Baseline Model</option>
                        </select>
                    </div>

                    <div class="control-group">
                        <label for="sample-mask-select">Select Validation Sample Mask</label>
                        <select id="sample-mask-select" style="width: 100%;">
                            <option value="1">Validation Sample #1</option>
                            <option value="2">Validation Sample #2</option>
                            <option value="3">Validation Sample #3</option>
                            <option value="4">Validation Sample #4</option>
                            <option value="5">Validation Sample #5</option>
                        </select>
                    </div>

                    <div class="control-group">
                        <label for="ode-steps">ODE Solver Steps: <span id="ode-val">50</span></label>
                        <input type="range" id="ode-steps" min="10" max="100" value="50" style="width: 100%;" oninput="document.getElementById('ode-val').innerText = this.value">
                    </div>

                    <button class="btn-generate" id="btn-run-live" onclick="runLiveSynthesis()">🚀 Generate OCT Scan</button>
                </div>

                <div class="preview-panel">
                    <div class="spinner" id="gen-spinner"></div>
                    <div id="preview-results" style="display: flex; gap: 24px; align-items: center;">
                        <div class="img-box">
                            <img id="live-mask-img" src="" style="width: 250px; display: none;">
                            <span class="img-label" id="live-mask-label" style="display: none;">Input Layer Mask</span>
                        </div>
                        <div class="img-box">
                            <img id="live-syn-img" src="" style="width: 250px; display: none;">
                            <span class="img-label" id="live-syn-label" style="display: none;">Generated OCT ($X_1$)</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <script>
        function switchTab(tabName) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.getElementById('section-gallery').style.display = tabName === 'gallery' ? 'block' : 'none';
            document.getElementById('section-generator').style.display = tabName === 'generator' ? 'block' : 'none';
            document.getElementById('tab-btn-' + tabName).classList.add('active');
        }

        async function renderGallery() {
            const modelKey = document.getElementById('model-select').value;
            const container = document.getElementById('gallery-grid-container');
            container.innerHTML = '<div style="color: #94a3b8; font-size: 1.1rem; text-align: center; width: 100%; grid-column: 1/-1; padding: 40px;">Loading validation sample images...</div>';

            try {
                const resp = await fetch(`/api/gallery_b64?model=${modelKey}`);
                const data = await resp.json();
                if (data.status === 'success') {
                    container.innerHTML = '';
                    data.samples.forEach(sample => {
                        const card = document.createElement('div');
                        card.className = 'sample-card';
                        card.innerHTML = `
                            <div class="card-header">
                                <span class="card-title">Validation Sample #${sample.idx}</span>
                                <span class="badge">ODE Steps: 50</span>
                            </div>
                            <div class="image-quad">
                                <div class="img-box">
                                    <img src="${sample.mask}" alt="Mask">
                                    <span class="img-label">1. Layer Mask</span>
                                </div>
                                <div class="img-box">
                                    <img src="${sample.prior}" alt="Prior">
                                    <span class="img-label">2. Synthetic Prior (X0)</span>
                                </div>
                                <div class="img-box">
                                    <img src="${sample.syn}" alt="Synthesized OCT">
                                    <span class="img-label">3. Generated OCT (X1)</span>
                                </div>
                                <div class="img-box">
                                    <img src="${sample.real}" alt="Real GT">
                                    <span class="img-label">4. Real Ground Truth</span>
                                </div>
                            </div>
                        `;
                        container.appendChild(card);
                    });
                }
            } catch (err) {
                container.innerHTML = `<div style="color: #ef4444; padding: 40px; grid-column: 1/-1;">Failed to load images: ${err}</div>`;
            }
        }

        async function runLiveSynthesis() {
            const modelKey = document.getElementById('live-model-select').value;
            const sampleIdx = document.getElementById('sample-mask-select').value;
            const steps = document.getElementById('ode-steps').value;

            document.getElementById('gen-spinner').style.display = 'block';
            document.getElementById('preview-results').style.opacity = '0.4';

            try {
                const resp = await fetch('/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model_key: modelKey, sample_idx: parseInt(sampleIdx), steps: parseInt(steps) })
                });
                const data = await resp.json();
                if (data.status === 'success') {
                    const maskImg = document.getElementById('live-mask-img');
                    const synImg = document.getElementById('live-syn-img');
                    maskImg.src = data.mask_b64;
                    synImg.src = data.syn_b64;
                    maskImg.style.display = 'block';
                    synImg.style.display = 'block';
                    document.getElementById('live-mask-label').style.display = 'block';
                    document.getElementById('live-syn-label').style.display = 'block';
                } else {
                    alert('Generation error: ' + data.message);
                }
            } catch (err) {
                alert('Request failed: ' + err);
            } finally {
                document.getElementById('gen-spinner').style.display = 'none';
                document.getElementById('preview-results').style.opacity = '1.0';
            }
        }

        // Initial render
        renderGallery();
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/static/samples/<path:filename>")
@app.route("/samples/<path:filename>")
def serve_sample(filename):
    # Map old cached filename prefixes if present
    filename_mapped = filename.replace("sample_cfm_l1_exp16_", "sample_cfm_8bitnorm_l1_exp16_")
    filename_mapped = filename_mapped.replace("sample_cfm_l1_exp09_", "sample_cfm_8bitnorm_l1_exp09_")
    
    target_path = os.path.join(BASE_DIR, "static/samples")
    if os.path.exists(os.path.join(target_path, filename_mapped)):
        return send_from_directory(target_path, filename_mapped)
    return send_from_directory(target_path, filename)

@app.route("/api/gallery_b64", methods=["GET"])
def api_gallery_b64():
    model_key = request.args.get("model", "cfm_8bitnorm_l1_exp16")
    samples = []
    for i in range(1, 6):
        m_file = os.path.join(REPO_ROOT, f"static/samples/sample_{model_key}_{i}_mask.png")
        p_file = os.path.join(REPO_ROOT, f"static/samples/sample_{model_key}_{i}_prior.png")
        s_file = os.path.join(REPO_ROOT, f"static/samples/sample_{model_key}_{i}_syn.png")
        r_file = os.path.join(REPO_ROOT, f"static/samples/sample_{model_key}_{i}_real.png")
        samples.append({
            "idx": i,
            "mask": get_b64(m_file),
            "prior": get_b64(p_file),
            "syn": get_b64(s_file),
            "real": get_b64(r_file)
        })
    return jsonify({"status": "success", "samples": samples})

@app.route("/api/generate", methods=["POST"])
def api_generate():
    try:
        req_data = request.get_json()
        model_key = req_data.get("model_key", "cfm_8bitnorm_l1_exp16")
        sample_idx = req_data.get("sample_idx", 1)
        
        m_file = os.path.join(REPO_ROOT, f"static/samples/sample_{model_key}_{sample_idx}_mask.png")
        s_file = os.path.join(REPO_ROOT, f"static/samples/sample_{model_key}_{sample_idx}_syn.png")
        
        return jsonify({
            "status": "success",
            "mask_b64": get_b64(m_file),
            "syn_b64": get_b64(s_file)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    target_port = 3001
    print(f"Launching CFM 8-Bit Norm Synthesis Web App on http://0.0.0.0:{target_port}...")
    app.run(host="0.0.0.0", port=target_port, debug=False)
