import os
import glob
import json
import numpy as np
from flask import Flask, render_template, jsonify, request, send_from_directory

app = Flask(__name__)
DATA_DIR = "/home/mmk/Codes/oct_data_synthesis/DATA/OCTID/Manual-Segmenation/Manual_Segmentation"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/images')
def get_images():
    extracted_dirs = glob.glob(os.path.join(DATA_DIR, "*_octSegmentation"))
    images = []
    for ex_dir in extracted_dirs:
        dir_name = os.path.basename(ex_dir)
        img_name = dir_name.replace("_octSegmentation", "")
        images.append({
            "dir_name": dir_name,
            "img_name": img_name
        })
    return jsonify(images)

@app.route('/api/data/<dir_name>')
def get_data(dir_name):
    ex_dir = os.path.join(DATA_DIR, dir_name)
    if not os.path.exists(ex_dir):
        return jsonify({"error": "Not found"}), 404
        
    # Read existing alignment if present
    alignment = {"offset_x": 0, "offset_y": 0, "scale_x": 1.0, "scale_y": 1.0, "rotate": 0, "flip_x": False, "flip_y": False}
    align_file = os.path.join(ex_dir, "alignment.json")
    if os.path.exists(align_file):
        with open(align_file, 'r') as f:
            alignment = json.load(f)
            
    # Collect coordinates
    layers = []
    all_x = []
    all_y = []
    
    for layer_idx in range(7):
        x_file = os.path.join(ex_dir, f"imageLayer_retinalLayers_0_0_pathX_0_{layer_idx}.csv")
        y_file = os.path.join(ex_dir, f"imageLayer_retinalLayers_0_0_pathY_0_{layer_idx}.csv")
        
        if os.path.exists(x_file) and os.path.exists(y_file):
            x_data = np.loadtxt(x_file, delimiter=",").tolist()
            y_data = np.loadtxt(y_file, delimiter=",").tolist()
            
            if not isinstance(x_data, list): x_data = [x_data]
            if not isinstance(y_data, list): y_data = [y_data]
            
            layers.append({"layer_idx": layer_idx, "x": x_data, "y": y_data})
            all_x.extend([x for x in x_data if not np.isnan(x)])
            all_y.extend([y for y in y_data if not np.isnan(y)])

    global_max_y = max(all_y) if all_y else 0
    global_max_x = max(all_x) if all_x else 0
    
    # Read the exact transformation parameters (xrange, yrange, isResize)
    min_xrange = 0
    min_yrange = 0
    resize_scale = 1.0
    
    xrange_file = os.path.join(ex_dir, "imageLayer_params_0_0_xrange_0_0.csv")
    yrange_file = os.path.join(ex_dir, "imageLayer_params_0_0_yrange_0_0.csv")
    resize_file = os.path.join(ex_dir, "imageLayer_params_0_0_isResize_0_0.csv")
    
    if os.path.exists(xrange_file):
        xr = np.loadtxt(xrange_file, delimiter=",")
        min_xrange = np.min(xr) if xr.size > 0 else 0
    
    if os.path.exists(yrange_file):
        yr = np.loadtxt(yrange_file, delimiter=",")
        min_yrange = np.min(yr) if yr.size > 0 else 0
        
    if os.path.exists(resize_file):
        rz = np.loadtxt(resize_file, delimiter=",")
        if rz.size > 1:
            resize_scale = rz[1]
        elif rz.size == 1:
            resize_scale = float(rz)
            
    return jsonify({
        "layers": layers,
        "alignment": alignment,
        "global_max_y": global_max_y,
        "global_max_x": global_max_x,
        "min_xrange": min_xrange,
        "min_yrange": min_yrange,
        "resize_scale": resize_scale
    })

@app.route('/api/save/<dir_name>', methods=['POST'])
def save_alignment(dir_name):
    ex_dir = os.path.join(DATA_DIR, dir_name)
    data = request.json
    align_file = os.path.join(ex_dir, "alignment.json")
    with open(align_file, 'w') as f:
        json.dump(data, f)
    return jsonify({"success": True})

@app.route('/data/<path:filepath>')
def serve_file(filepath):
    return send_from_directory(DATA_DIR, filepath)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
