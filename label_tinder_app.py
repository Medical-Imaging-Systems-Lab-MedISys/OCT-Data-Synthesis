import os
import json
import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file, render_template_string
from datetime import datetime

app = Flask(__name__)

# Persistence storage file
STATE_FILE = "label_selection_state.json"
WORKSPACE_DIR = "/home/mmk/Codes/oct_data_synthesis"
DATA_DIR = os.path.join(WORKSPACE_DIR, "DATA")

# Load existing state
if os.path.exists(STATE_FILE):
    try:
        with open(STATE_FILE, "r") as f:
            selection_state = json.load(f)
    except Exception as e:
        print(f"Error loading state file: {e}")
        selection_state = {}
else:
    selection_state = {}

def save_state():
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(selection_state, f, indent=2)
    except Exception as e:
        print(f"Error saving state file: {e}")

# Build image index for pairing
def build_suffix_index(data_dir):
    print("Building original image suffix index...")
    suffix_map = {}
    skip_dirs = {"pseudo_labels", "combined_synthesis_data", "val_comparison_set", ".git", "__pycache__"}
    for root, dirs, files in os.walk(data_dir):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
                full_path = os.path.abspath(os.path.join(root, f))
                rel_path = os.path.relpath(full_path, data_dir)
                rel_path_no_ext, _ = os.path.splitext(rel_path)
                parts = rel_path_no_ext.split(os.sep)
                for k in range(1, len(parts) + 1):
                    suffix = "/".join(parts[-k:]).lower()
                    if suffix not in suffix_map:
                        suffix_map[suffix] = []
                    suffix_map[suffix].append(full_path)
    print(f"Indexed {len(suffix_map)} suffix patterns.")
    return suffix_map

image_index = build_suffix_index(DATA_DIR)

def find_real_image(label_path, index):
    label_filename = os.path.basename(label_path)
    base_name, _ = os.path.splitext(label_filename)
    parts = base_name.split('_')
    for k in range(len(parts), 0, -1):
        suffix = "/".join(parts[-k:]).lower()
        if suffix in index:
            matches = index[suffix]
            if k == 1 and len(matches) > 1:
                continue
            return matches[0]
    return None

# Find prediction directories (NR206 predictions, skipping NR206 as a target dataset)
def get_prediction_categories(data_dir):
    pseudo_dir = os.path.join(data_dir, "pseudo_labels")
    categories = []
    if not os.path.exists(pseudo_dir):
        return categories
        
    for root, dirs, files in os.walk(pseudo_dir):
        # We look for folders containing files that have "NR206" in their path (the model name)
        # but exclude any folders that evaluate NR206 as the target dataset ("on_NR206" / "predictions_nr206")
        root_rel = os.path.relpath(root, pseudo_dir)
        if root_rel == ".":
            continue
            
        parts = root_rel.split(os.sep)
        # Check if model is NR206 and target is not NR206
        has_nr206_model = any("nr206" in p.lower() for p in parts)
        evals_nr206_target = any("on_nr206" in p.lower() or "predictions_nr206" in p.lower() for p in parts)
        
        if has_nr206_model and not evals_nr206_target:
            png_files = [f for f in files if f.lower().endswith('.png')]
            if png_files:
                categories.append({
                    "id": root_rel,
                    "name": root_rel.replace(os.sep, " ➔ "),
                    "count": len(png_files)
                })
    return sorted(categories, key=lambda x: x["id"])

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OCT Pseudo-Label Validator</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: #151b2c;
            --accent-primary: #3b82f6;
            --accent-success: #10b981;
            --accent-danger: #ef4444;
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --border-color: rgba(255, 255, 255, 0.08);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
        }

        header {
            background-color: rgba(21, 27, 44, 0.7);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border-color);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        h1 {
            font-family: 'Outfit', sans-serif;
            font-size: 1.4rem;
            font-weight: 700;
            background: linear-gradient(135deg, #60a5fa, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .dropdown-container {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        select {
            background-color: #1f2937;
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-size: 0.9rem;
            outline: none;
            cursor: pointer;
            transition: all 0.3s;
        }

        select:focus {
            border-color: var(--accent-primary);
        }

        .main-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            position: relative;
        }

        .stats-panel {
            display: flex;
            gap: 1.5rem;
            margin-bottom: 1.5rem;
            background: rgba(21, 27, 44, 0.6);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 0.8rem 1.5rem;
        }

        .stat-item {
            text-align: center;
        }

        .stat-value {
            font-family: 'Outfit', sans-serif;
            font-size: 1.2rem;
            font-weight: 700;
        }

        .stat-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 2px;
        }

        .stat-approved { color: var(--accent-success); }
        .stat-discarded { color: var(--accent-danger); }
        .stat-remaining { color: var(--accent-primary); }

        /* Tinder Card Stack */
        .card-stack {
            position: relative;
            width: 480px;
            height: 520px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .card {
            position: absolute;
            width: 100%;
            height: 100%;
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 1.2rem;
            display: flex;
            flex-direction: column;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
            cursor: grab;
            user-select: none;
            touch-action: none;
            transition: transform 0.3s ease, opacity 0.3s ease;
        }

        .card.dragging {
            transition: none;
            cursor: grabbing;
        }

        .card-image-container {
            flex: 1;
            position: relative;
            background-color: #080b12;
            border-radius: 12px;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1px solid rgba(255, 255, 255, 0.04);
        }

        .card-image {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: contain;
            pointer-events: none;
        }

        .mask-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: contain;
            pointer-events: none;
            mix-blend-mode: screen;
            opacity: 0.6;
            transition: opacity 0.1s;
        }

        .card-info {
            margin-top: 1rem;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .card-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1rem;
            font-weight: 600;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .card-meta {
            font-size: 0.8rem;
            color: var(--text-secondary);
        }

        /* Swipe Labels */
        .swipe-label {
            position: absolute;
            top: 40px;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            font-size: 1.8rem;
            text-transform: uppercase;
            opacity: 0;
            transform: rotate(-10deg);
            z-index: 10;
            border: 4px solid;
            pointer-events: none;
            transition: opacity 0.1s;
        }

        .swipe-label.like {
            right: 40px;
            color: var(--accent-success);
            border-color: var(--accent-success);
            transform: rotate(15deg);
        }

        .swipe-label.nope {
            left: 40px;
            color: var(--accent-danger);
            border-color: var(--accent-danger);
            transform: rotate(-15deg);
        }

        /* Controls */
        .controls {
            margin-top: 2rem;
            display: flex;
            align-items: center;
            gap: 1.5rem;
        }

        .btn {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 0.8rem 1.5rem;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }

        .btn-circle {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            padding: 0;
            justify-content: center;
            font-size: 1.5rem;
        }

        .btn-discard {
            border-color: rgba(239, 68, 68, 0.3);
            color: var(--accent-danger);
        }

        .btn-discard:hover {
            background-color: var(--accent-danger);
            color: white;
            border-color: var(--accent-danger);
        }

        .btn-keep {
            border-color: rgba(16, 185, 129, 0.3);
            color: var(--accent-success);
        }

        .btn-keep:hover {
            background-color: var(--accent-success);
            color: white;
            border-color: var(--accent-success);
        }

        .btn-undo {
            color: var(--text-secondary);
        }

        .crop-mask-top {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            z-index: 5;
            pointer-events: none;
            transition: height 0.1s, background-color 0.1s;
        }
        .crop-mask-bottom {
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            z-index: 5;
            pointer-events: none;
            transition: height 0.1s, background-color 0.1s;
        }

        /* Overlay Control */
        .overlay-control {
            margin-top: 1.5rem;
            width: 320px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .slider-label {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            color: var(--text-secondary);
        }

        .slider {
            width: 100%;
            height: 6px;
            background: #1f2937;
            outline: none;
            border-radius: 3px;
            appearance: none;
            cursor: pointer;
        }

        .slider::-webkit-slider-thumb {
            appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: var(--accent-primary);
            cursor: pointer;
            transition: transform 0.1s;
        }

        .slider::-webkit-slider-thumb:hover {
            transform: scale(1.2);
        }

        /* Help Modal */
        .keyboard-help {
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-top: 1.5rem;
            text-align: center;
            line-height: 1.5;
        }

        kbd {
            background: #1f2937;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            padding: 2px 6px;
            font-family: monospace;
            color: var(--text-primary);
        }

        .view-mode-tabs {
            display: flex;
            background-color: #1f2937;
            padding: 3px;
            border-radius: 8px;
            margin-bottom: 1rem;
        }

        .view-mode-tab {
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 500;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.2s;
        }

        .view-mode-tab.active {
            background-color: var(--card-bg);
            color: var(--text-primary);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        /* Side-by-side mode styling */
        .card.side-by-side {
            width: 780px !important;
        }

        .side-by-side .card-image-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            background-color: transparent;
            border: none;
        }

        .side-by-side .img-panel {
            background-color: #080b12;
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            border: 1px solid var(--border-color);
            height: 100%;
        }

        .side-by-side-label {
            position: absolute;
            top: 8px;
            left: 8px;
            background-color: rgba(0, 0, 0, 0.6);
            color: white;
            font-size: 0.7rem;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 600;
            z-index: 2;
        }

        /* Mobile & Tablet Responsiveness */
        @media (max-width: 768px) {
            header {
                flex-direction: column;
                gap: 0.8rem;
                padding: 0.8rem 1rem;
                align-items: stretch;
            }

            h1 {
                font-size: 1.2rem;
                text-align: center;
            }

            .dropdown-container {
                width: 100%;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 8px;
            }

            select {
                flex: 1;
                min-width: 150px;
                font-size: 0.85rem;
                padding: 0.5rem;
            }

            .main-container {
                padding: 1rem 0.5rem;
            }

            .stats-panel {
                width: 100%;
                max-width: 440px;
                gap: 0.5rem;
                padding: 0.6rem 0.8rem;
                justify-content: space-around;
            }

            .stat-value {
                font-size: 1rem;
            }

            .stat-label {
                font-size: 0.65rem;
            }

            .card-stack {
                width: min(94vw, 440px) !important;
                height: min(58vh, 460px);
                min-height: 360px;
            }

            .card.side-by-side {
                width: min(94vw, 440px) !important;
            }

            .side-by-side .card-image-container {
                grid-template-columns: 1fr;
                grid-template-rows: 1fr 1fr;
                gap: 6px;
            }

            .controls {
                margin-top: 1.2rem;
                gap: 1.5rem;
            }

            .btn-circle {
                width: 66px;
                height: 66px;
                font-size: 1.75rem;
            }

            .overlay-control, .crop-control-panel {
                width: min(94vw, 440px) !important;
            }

            .swipe-label {
                font-size: 1.4rem;
                padding: 0.3rem 0.7rem;
                top: 20px;
            }

            .swipe-label.like {
                right: 20px;
            }

            .swipe-label.nope {
                left: 20px;
            }
        }
    </style>
</head>
<body>
    <header>
        <h1>OCT Pseudo-Label Tinder</h1>
        <div class="dropdown-container">
            <label for="category-select" style="font-size: 0.85rem; color: var(--text-secondary);">Dataset:</label>
            <select id="category-select" onchange="loadCategory()">
                {% for cat in categories %}
                <option value="{{ cat.id }}">{{ cat.name }} ({{ cat.count }} items)</option>
                {% endfor %}
            </select>
            <button class="btn" onclick="exportData()" style="padding: 0.5rem 1rem; border-radius: 8px; font-size: 0.85rem;">
                Export Approved
            </button>
        </div>
    </header>

    <div class="main-container">
        <div class="stats-panel">
            <div class="stat-item">
                <div id="stat-total" class="stat-value">0</div>
                <div class="stat-label">Total</div>
            </div>
            <div class="stat-item">
                <div id="stat-approved" class="stat-value stat-approved">0</div>
                <div class="stat-label">Approved</div>
            </div>
            <div class="stat-item">
                <div id="stat-discarded" class="stat-value stat-discarded">0</div>
                <div class="stat-label">Discarded</div>
            </div>
            <div class="stat-item">
                <div id="stat-remaining" class="stat-value stat-remaining">0</div>
                <div class="stat-label">Remaining</div>
            </div>
        </div>

        <div class="view-mode-tabs">
            <div id="tab-overlay" class="view-mode-tab active" onclick="setViewMode('overlay')">Overlay Mode</div>
            <div id="tab-side" class="view-mode-tab" onclick="setViewMode('side')">Side-by-Side Mode</div>
        </div>

        <div class="card-stack" id="card-stack">
            <!-- Cards will be dynamically injected here -->
        </div>

        <div class="overlay-control" id="overlay-slider-container">
            <div class="slider-label">
                <span>Overlay Opacity</span>
                <span id="opacity-val">60%</span>
            </div>
            <input type="range" class="slider" id="opacity-slider" min="0" max="100" value="60" oninput="updateOpacity(this.value)">
        </div>

        <div class="crop-control-panel" style="margin-top: 1rem; width: 320px; display: flex; flex-direction: column; gap: 10px; background: rgba(21, 27, 44, 0.4); border: 1px solid var(--border-color); border-radius: 12px; padding: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-family: 'Outfit', sans-serif; font-weight: 600; font-size: 0.9rem;">Crop & Masking Settings</span>
                <input type="checkbox" id="crop-enable" onchange="toggleCrop(this.checked)" style="cursor: pointer;">
            </div>
            
            <div id="crop-settings-content" style="display: none; flex-direction: column; gap: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; color: var(--text-secondary); margin-top: 5px;">
                    <span>Preprocess Mode</span>
                    <div style="display: flex; flex-direction: column; gap: 4px; align-items: flex-end;">
                        <label style="display: flex; align-items: center; gap: 4px; cursor: pointer;">
                            <input type="radio" name="crop-mode" value="mask" checked onchange="updateCropMode(this.value)"> Mask (B/W)
                        </label>
                        <label style="display: flex; align-items: center; gap: 4px; cursor: pointer;">
                            <input type="radio" name="crop-mode" value="crop" onchange="updateCropMode(this.value)"> Crop & Scale
                        </label>
                    </div>
                </div>

                <div class="slider-label">
                    <span>Top Y</span>
                    <span id="top-crop-val">0%</span>
                </div>
                <input type="range" class="slider" id="top-crop-slider" min="0" max="100" value="0" oninput="updateTopCrop(this.value)">
                
                <div class="slider-label">
                    <span>Bottom Y</span>
                    <span id="bottom-crop-val">100%</span>
                </div>
                <input type="range" class="slider" id="bottom-crop-slider" min="0" max="100" value="100" oninput="updateBottomCrop(this.value)">
                
                <div id="mask-color-container" style="display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; color: var(--text-secondary); margin-top: 5px;">
                    <span>Mask Color</span>
                    <div style="display: flex; gap: 10px;">
                        <label style="display: flex; align-items: center; gap: 4px; cursor: pointer;">
                            <input type="radio" name="mask-color" value="black" checked onchange="updateMaskColor(this.value)"> Black
                        </label>
                        <label style="display: flex; align-items: center; gap: 4px; cursor: pointer;">
                            <input type="radio" name="mask-color" value="white" onchange="updateMaskColor(this.value)"> White
                        </label>
                    </div>
                </div>
            </div>
        </div>

        <div class="controls">
            <button class="btn btn-undo" onclick="undoLastDecision()">
                ↩️ Undo
            </button>
            <button class="btn btn-circle btn-discard" onclick="swipeLeft()" title="Discard (A / Left Arrow)">
                ❌
            </button>
            <button class="btn btn-circle btn-keep" onclick="swipeRight()" title="Keep (D / Right Arrow)">
                💚
            </button>
        </div>

        <div class="keyboard-help">
            Shortcuts: <kbd>←</kbd> or <kbd>A</kbd> to Discard &nbsp;|&nbsp; <kbd>→</kbd> or <kbd>D</kbd> to Keep<br>
            <kbd>Space</kbd> to toggle Overlay &nbsp;|&nbsp; <kbd>Ctrl+Z</kbd> to Undo last action
        </div>
    </div>

    <script>
        let samples = [];
        let currentIndex = 0;
        let activeCard = null;
        let isDragging = false;
        let startX = 0;
        let startY = 0;
        let currentX = 0;
        let currentY = 0;
        let opacityValue = 60;
        let viewMode = 'overlay'; // 'overlay' or 'side'
        let cropEnabled = false;
        let topCropY = 0;
        let bottomCropY = 100;
        let maskColor = 'black';
        let preprocessMode = 'mask';

        async function loadCategory() {
            const catId = document.getElementById("category-select").value;
            const res = await fetch(`/api/samples?category=${encodeURIComponent(catId)}`);
            const data = await res.json();
            
            samples = data.samples;
            // Find first pending index
            currentIndex = samples.findIndex(s => s.status === 'pending');
            if (currentIndex === -1) {
                // If all are processed, show the last card or empty state
                currentIndex = samples.length;
            }
            
            updateStats();
            renderStack();
        }

        function updateStats() {
            const total = samples.length;
            const approved = samples.filter(s => s.status === 'approved').length;
            const discarded = samples.filter(s => s.status === 'discarded').length;
            const remaining = samples.filter(s => s.status === 'pending').length;

            document.getElementById("stat-total").innerText = total;
            document.getElementById("stat-approved").innerText = approved;
            document.getElementById("stat-discarded").innerText = discarded;
            document.getElementById("stat-remaining").innerText = remaining;
        }

        function setViewMode(mode) {
            viewMode = mode;
            document.getElementById("tab-overlay").classList.toggle("active", mode === 'overlay');
            document.getElementById("tab-side").classList.toggle("active", mode === 'side');
            
            document.getElementById("overlay-slider-container").style.visibility = mode === 'overlay' ? 'visible' : 'hidden';
            
            // Adjust card sizes in CSS
            const stack = document.getElementById("card-stack");
            if (mode === 'side') {
                stack.style.width = '780px';
            } else {
                stack.style.width = '480px';
            }
            
            renderStack();
        }

        function renderStack() {
            const stack = document.getElementById("card-stack");
            stack.innerHTML = "";

            if (currentIndex >= samples.length) {
                stack.innerHTML = `<div style="text-align:center; padding: 2rem;">
                    <h3 style="font-family:'Outfit', sans-serif; font-size: 1.4rem; margin-bottom: 0.5rem;">🎉 Category Complete!</h3>
                    <p style="color: var(--text-secondary);">All samples in this set have been reviewed.</p>
                </div>`;
                return;
            }

            // Render up to 2 cards for optimal stack performance
            for (let i = Math.min(samples.length - 1, currentIndex + 1); i >= currentIndex; i--) {
                const sample = samples[i];
                const card = document.createElement("div");
                card.className = `card ${viewMode === 'side' ? 'side-by-side' : ''}`;
                
                // Add scale and offset to background cards
                if (i === currentIndex + 1) {
                    card.style.transform = "scale(0.95) translateY(10px)";
                    card.style.opacity = "0.7";
                    card.style.pointerEvents = "none";
                }

                // In overlay mode, overlay the mask on top of the image
                let imageHtml = "";
                if (viewMode === 'overlay') {
                    imageHtml = `
                        <div class="card-image-container">
                            <div class="crop-mask-top"></div>
                            <div class="crop-mask-bottom"></div>
                            <img class="card-image" src="/api/image?path=${encodeURIComponent(sample.real_path)}" alt="Real OCT">
                            <img class="card-image mask-overlay" id="overlay-${i}" src="/api/image?path=${encodeURIComponent(sample.label_path)}&colormap=jet" style="opacity: ${opacityValue/100};" alt="Mask">
                        </div>
                    `;
                } else {
                    imageHtml = `
                        <div class="card-image-container">
                            <div class="img-panel">
                                <div class="crop-mask-top"></div>
                                <div class="crop-mask-bottom"></div>
                                <span class="side-by-side-label">Original</span>
                                <img class="card-image" src="/api/image?path=${encodeURIComponent(sample.real_path)}" alt="Real OCT">
                            </div>
                            <div class="img-panel">
                                <div class="crop-mask-top"></div>
                                <div class="crop-mask-bottom"></div>
                                <span class="side-by-side-label">Pseudo-Label</span>
                                <img class="card-image" src="/api/image?path=${encodeURIComponent(sample.label_path)}&colormap=jet" alt="Mask">
                            </div>
                        </div>
                    `;
                }

                card.innerHTML = `
                    <div class="swipe-label like" id="like-label-${i}">KEEP</div>
                    <div class="swipe-label nope" id="nope-label-${i}">DISCARD</div>
                    ${imageHtml}
                    <div class="card-info">
                        <div class="card-title" title="${sample.name}">${sample.name}</div>
                        <div class="card-meta">Index: ${i + 1} / ${samples.length}</div>
                    </div>
                `;

                if (i === currentIndex) {
                    activeCard = card;
                    setupDrag(card);
                }

                stack.appendChild(card);
            }
            // Apply crop masks to the newly generated cards
            updateCropMasks();
        }

        function updateOpacity(val) {
            opacityValue = val;
            document.getElementById("opacity-val").innerText = `${val}%`;
            const overlay = document.getElementById(`overlay-${currentIndex}`);
            if (overlay) {
                overlay.style.opacity = val / 100;
            }
        }

        function toggleCrop(enabled) {
            cropEnabled = enabled;
            document.getElementById("crop-settings-content").style.display = enabled ? 'flex' : 'none';
            updateCropMasks();
        }

        function updateTopCrop(val) {
            topCropY = parseInt(val);
            document.getElementById("top-crop-val").innerText = `${val}%`;
            const bottomVal = parseInt(document.getElementById("bottom-crop-slider").value);
            if (topCropY > bottomVal) {
                document.getElementById("bottom-crop-slider").value = topCropY;
                updateBottomCrop(topCropY);
            } else {
                updateCropMasks();
            }
        }

        function updateBottomCrop(val) {
            bottomCropY = parseInt(val);
            document.getElementById("bottom-crop-val").innerText = `${val}%`;
            const topVal = parseInt(document.getElementById("top-crop-slider").value);
            if (bottomCropY < topVal) {
                document.getElementById("top-crop-slider").value = bottomCropY;
                updateTopCrop(bottomCropY);
            } else {
                updateCropMasks();
            }
        }

        function updateMaskColor(color) {
            maskColor = color;
            updateCropMasks();
        }

        function updateCropMode(mode) {
            preprocessMode = mode;
            document.getElementById("mask-color-container").style.display = mode === 'mask' ? 'flex' : 'none';
            updateCropMasks();
        }

        function updateCropMasks() {
            const topMasks = document.querySelectorAll(".crop-mask-top");
            const bottomMasks = document.querySelectorAll(".crop-mask-bottom");
            
            if (preprocessMode === 'mask') {
                const color = maskColor === 'white' ? '#ffffff' : '#000000';
                const topHeight = cropEnabled ? `${topCropY}%` : '0%';
                const bottomHeight = cropEnabled ? `${100 - bottomCropY}%` : '0%';
                
                topMasks.forEach(m => {
                    m.style.height = topHeight;
                    m.style.backgroundColor = color;
                });
                bottomMasks.forEach(m => {
                    m.style.height = bottomHeight;
                    m.style.backgroundColor = color;
                });
                
                const images = document.querySelectorAll(".card-image, .mask-overlay");
                images.forEach(img => {
                    img.style.height = "100%";
                    img.style.top = "0%";
                    img.style.objectFit = "contain";
                });
            } else {
                topMasks.forEach(m => m.style.height = "0%");
                bottomMasks.forEach(m => m.style.height = "0%");
                
                const images = document.querySelectorAll(".card-image, .mask-overlay");
                if (cropEnabled) {
                    const slicedHeight = bottomCropY - topCropY;
                    const scale = slicedHeight > 0 ? (100 / slicedHeight) : 100;
                    const topStyle = -topCropY * scale;
                    
                    images.forEach(img => {
                        img.style.height = `${scale * 100}%`;
                        img.style.top = `${topStyle}%`;
                        img.style.objectFit = "fill";
                    });
                } else {
                    images.forEach(img => {
                        img.style.height = "100%";
                        img.style.top = "0%";
                        img.style.objectFit = "contain";
                    });
                }
            }
        }

        function setupDrag(card) {
            card.addEventListener("mousedown", dragStart);
            card.addEventListener("touchstart", dragStart, { passive: true });

            function dragStart(e) {
                isDragging = true;
                card.classList.add("dragging");

                if (e.type === "touchstart") {
                    startX = e.touches[0].clientX;
                    startY = e.touches[0].clientY;
                } else {
                    startX = e.clientX;
                    startY = e.clientY;
                }

                document.addEventListener("mousemove", dragMove);
                document.addEventListener("touchmove", dragMove, { passive: false });
                document.addEventListener("mouseup", dragEnd);
                document.addEventListener("touchend", dragEnd);
            }

            function dragMove(e) {
                if (!isDragging) return;

                let clientX, clientY;
                if (e.type === "touchmove") {
                    clientX = e.touches[0].clientX;
                    clientY = e.touches[0].clientY;
                    if (Math.abs(clientX - startX) > 10) {
                        e.preventDefault();
                    }
                } else {
                    clientX = e.clientX;
                    clientY = e.clientY;
                }

                currentX = clientX - startX;
                currentY = clientY - startY;

                // Rotation calculation
                const rot = currentX * 0.08;
                card.style.transform = `translate(${currentX}px, ${currentY}px) rotate(${rot}deg)`;

                // Handle Swipe Label Opacities
                const nopeLabel = document.getElementById(`nope-label-${currentIndex}`);
                const likeLabel = document.getElementById(`like-label-${currentIndex}`);
                
                if (currentX > 0) {
                    likeLabel.style.opacity = Math.min(currentX / 100, 1);
                    nopeLabel.style.opacity = 0;
                } else {
                    nopeLabel.style.opacity = Math.min(-currentX / 100, 1);
                    likeLabel.style.opacity = 0;
                }
            }

            function dragEnd() {
                isDragging = false;
                card.classList.remove("dragging");

                document.removeEventListener("mousemove", dragMove);
                document.removeEventListener("touchmove", dragMove);
                document.removeEventListener("mouseup", dragEnd);
                document.removeEventListener("touchend", dragEnd);

                const threshold = 150;
                if (currentX > threshold) {
                    swipeRight();
                } else if (currentX < -threshold) {
                    swipeLeft();
                } else {
                    // Reset
                    card.style.transform = "";
                    document.getElementById(`nope-label-${currentIndex}`).style.opacity = 0;
                    document.getElementById(`like-label-${currentIndex}`).style.opacity = 0;
                }
                currentX = 0;
                currentY = 0;
            }
        }

        async function submitDecision(filename, status) {
            const catId = document.getElementById("category-select").value;
            const res = await fetch("/api/select", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ 
                    category: catId, 
                    filename: filename, 
                    status: status,
                    crop: {
                        enabled: cropEnabled,
                        mode: preprocessMode,
                        top_y: topCropY,
                        bottom_y: bottomCropY,
                        color: maskColor
                    }
                })
            });
            const data = await res.json();
            if (data.success) {
                samples[currentIndex].status = status;
                currentIndex++;
                updateStats();
                renderStack();
            }
        }

        function swipeLeft() {
            if (currentIndex >= samples.length) return;
            const card = activeCard;
            if (card) {
                card.style.transition = "transform 0.4s ease, opacity 0.4s ease";
                card.style.transform = "translate(-800px, 50px) rotate(-30deg)";
                card.style.opacity = "0";
            }
            submitDecision(samples[currentIndex].name, 'discarded');
        }

        function swipeRight() {
            if (currentIndex >= samples.length) return;
            const card = activeCard;
            if (card) {
                card.style.transition = "transform 0.4s ease, opacity 0.4s ease";
                card.style.transform = "translate(800px, 50px) rotate(30deg)";
                card.style.opacity = "0";
            }
            submitDecision(samples[currentIndex].name, 'approved');
        }

        async function undoLastDecision() {
            if (currentIndex <= 0) return;
            const catId = document.getElementById("category-select").value;
            const previousFilename = samples[currentIndex - 1].name;

            const res = await fetch("/api/undo", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ category: catId, filename: previousFilename })
            });
            
            const data = await res.json();
            if (data.success) {
                currentIndex--;
                samples[currentIndex].status = 'pending';
                updateStats();
                renderStack();
            }
        }

        function exportData() {
            const catId = document.getElementById("category-select").value;
            window.location.href = `/api/export?category=${encodeURIComponent(catId)}`;
        }

        // Keyboard Event Handlers
        document.addEventListener("keydown", (e) => {
            if (isDragging) return;
            if (e.key === "ArrowLeft" || e.key.toLowerCase() === "a") {
                swipeLeft();
            } else if (e.key === "ArrowRight" || e.key.toLowerCase() === "d") {
                swipeRight();
            } else if (e.key === " ") {
                e.preventDefault();
                // Toggle overlay opacity between 0 and 60
                const cur = parseInt(document.getElementById("opacity-slider").value);
                const nextVal = cur > 0 ? 0 : 60;
                document.getElementById("opacity-slider").value = nextVal;
                updateOpacity(nextVal);
            } else if (e.key.toLowerCase() === "z" && e.ctrlKey) {
                undoLastDecision();
            }
        });

        // Initialize App
        loadCategory();
    </script>
</body>
</html>
"""

# API Routes
@app.route("/")
def index():
    categories = get_prediction_categories(DATA_DIR)
    return render_template_string(HTML_TEMPLATE, categories=categories)

@app.route("/api/samples")
def get_samples():
    category = request.args.get("category")
    if not category:
        return jsonify({"error": "Category parameter is required"}), 400
        
    category_dir = os.path.join(DATA_DIR, "pseudo_labels", category)
    if not os.path.exists(category_dir):
        return jsonify({"error": "Category directory does not exist"}), 404
        
    filenames = sorted([f for f in os.listdir(category_dir) if f.lower().endswith('.png')])
    
    samples = []
    category_state = selection_state.get(category, {})
    
    for fn in filenames:
        label_path = os.path.join(category_dir, fn)
        real_path = find_real_image(label_path, image_index)
        
        if not real_path:
            # Skip samples if real image is not found to prevent rendering crashes
            continue
            
        status = category_state.get(fn, {}).get("status", "pending")
        samples.append({
            "name": fn,
            "label_path": label_path,
            "real_path": real_path,
            "status": status
        })
        
    return jsonify({"samples": samples})

@app.route("/api/image")
def get_image():
    img_path = request.args.get("path")
    colormap = request.args.get("colormap")
    if not img_path or not os.path.exists(img_path):
        return jsonify({"error": "Image path invalid or not found"}), 404
        
    if colormap == "jet":
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            unique_vals = sorted(list(set(img.flatten()) - {0}))
            num_classes = len(unique_vals)
            if num_classes > 0:
                lut = np.zeros(256, dtype=np.uint8)
                for idx, val in enumerate(unique_vals):
                    if num_classes == 1:
                        lut[val] = 128
                    else:
                        lut[val] = int(30 + idx * (210 / (num_classes - 1)))
                img = cv2.LUT(img, lut)
                
            img_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            img_color[img == 0] = 0
            _, buf = cv2.imencode(".png", img_color)
            from io import BytesIO
            return send_file(BytesIO(buf.tobytes()), mimetype="image/png")
            
    # For real OCT images (including TIFF/TIF medical formats unsupported by mobile browsers):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        if len(img.shape) == 3 and img.shape[2] == 4:
            # BGRA to BGR or keep as is
            pass
        elif len(img.shape) == 3 and img.shape[2] == 3:
            pass
        else:
            # Grayscale to 8-bit standard range if 16-bit TIFF
            if img.dtype == np.uint16:
                img = (img / 256).astype(np.uint8)
        _, buf = cv2.imencode(".png", img)
        from io import BytesIO
        return send_file(BytesIO(buf.tobytes()), mimetype="image/png")

    return send_file(img_path)

@app.route("/api/select", methods=["POST"])
def select_sample():
    data = request.json
    category = data.get("category")
    filename = data.get("filename")
    status = data.get("status") # 'approved' or 'discarded'
    crop = data.get("crop")
    
    if not category or not filename or not status:
        return jsonify({"error": "Missing parameters"}), 400
        
    if category not in selection_state:
        selection_state[category] = {}
        
    selection_state[category][filename] = {
        "status": status,
        "crop": crop,
        "timestamp": datetime.utcnow().isoformat()
    }
    save_state()
    return jsonify({"success": True})

@app.route("/api/undo", methods=["POST"])
def undo_decision():
    data = request.json
    category = data.get("category")
    filename = data.get("filename")
    
    if not category or not filename:
        return jsonify({"error": "Missing parameters"}), 400
        
    if category in selection_state and filename in selection_state[category]:
        del selection_state[category][filename]
        save_state()
        return jsonify({"success": True})
        
    return jsonify({"error": "Item not found to undo"}), 404

@app.route("/api/export")
def export_category():
    category = request.args.get("category")
    if not category:
        return "Category parameter is required", 400
        
    category_state = selection_state.get(category, {})
    approved_files = [fn for fn, item in category_state.items() if item.get("status") == "approved"]
    
    # We output a JSON listing the pairs
    export_data = []
    category_dir = os.path.join(DATA_DIR, "pseudo_labels", category)
    
    for fn in approved_files:
        label_path = os.path.join(category_dir, fn)
        real_path = find_real_image(label_path, image_index)
        if real_path:
            crop_info = category_state[fn].get("crop")
            export_data.append({
                "filename": fn,
                "label_path": os.path.relpath(label_path, WORKSPACE_DIR),
                "real_path": os.path.relpath(real_path, WORKSPACE_DIR),
                "crop": crop_info
            })
            
    response_content = json.dumps(export_data, indent=2)
    
    # Download as attachment
    from flask import Response
    safe_name = category.replace(os.sep, "_")
    return Response(
        response_content,
        mimetype="application/json",
        headers={"Content-Disposition": f"attachment;filename=approved_{safe_name}.json"}
    )

if __name__ == "__main__":
    # Always bind to 0.0.0.0 and port 3000 to be open to LAN
    app.run(host="0.0.0.0", port=3000, debug=False)
