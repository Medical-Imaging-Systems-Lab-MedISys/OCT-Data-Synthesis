import os
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, abort, send_file
import markdown
import io
from PIL import Image

app = Flask(__name__)

# Path to the datasets
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'DATA'))

def get_datasets():
    if not os.path.exists(DATA_DIR):
        return []
    datasets = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    return sorted(datasets)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/datasets')
def api_datasets():
    return jsonify(get_datasets())

@app.route('/api/dataset/<name>')
def api_dataset(name):
    dataset_path = os.path.join(DATA_DIR, name)
    if not os.path.exists(dataset_path):
        return abort(404)
    
    tree = {}
    
    # Recursively find files up to a limit
    def scan_dir_tree(d, rel_base):
        try:
            entries = os.listdir(d)
        except PermissionError:
            return None
        
        node = {'images': [], 'masks': [], 'arrays': [], 'metadata': [], 'subdirs': {}}
        has_content = False
        
        for entry in sorted(entries):
            full_path = os.path.join(d, entry)
            rel_path = os.path.join(rel_base, entry)
            
            if os.path.isdir(full_path):
                child_node = scan_dir_tree(full_path, rel_path)
                if child_node:
                    node['subdirs'][entry] = child_node
                    has_content = True
            else:
                ext = os.path.splitext(entry)[1].lower()
                if ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
                    # Validate PNGs to avoid fake CSVs misnamed as .PNG (like in OCT5k)
                    if ext == '.png':
                        try:
                            with open(full_path, 'rb') as f:
                                if not f.read(8).startswith(b'\x89PNG\r\n\x1a\n'):
                                    continue
                        except Exception:
                            continue
                    if 'mask' in entry.lower() or 'seg' in entry.lower():
                        if len(node['masks']) < 20: node['masks'].append(rel_path)
                    else:
                        if len(node['images']) < 20: node['images'].append(rel_path)
                    has_content = True
                elif ext in ['.npy', '.mat']:
                    if ext == '.mat':
                        try:
                            from scipy.io import whosmat
                            mat_info = whosmat(full_path)
                            for array_name, shape, dtype in mat_info:
                                if len(node['arrays']) < 20: 
                                    node['arrays'].append(f"{rel_path}?array={array_name}")
                                has_content = True
                        except Exception:
                            if len(node['arrays']) < 20: node['arrays'].append(rel_path)
                            has_content = True
                    else:
                        if len(node['arrays']) < 20: node['arrays'].append(rel_path)
                        has_content = True
                elif ext in ['.csv', '.json', '.xml', '.txt'] or entry.lower() in ['readme.md', 'readme.txt', 'readme']:
                    if name == 'OCT5k' and ext == '.csv':
                        continue
                    if len(node['metadata']) < 20: node['metadata'].append(rel_path)
                    has_content = True
                elif ext == '.zip':
                    import zipfile
                    try:
                        with zipfile.ZipFile(full_path, 'r') as z:
                            for member in z.namelist():
                                m_ext = os.path.splitext(member)[1].lower()
                                if m_ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
                                    if 'mask' in member.lower() or 'seg' in member.lower():
                                        if len(node['masks']) < 20: node['masks'].append(f"{rel_path}?member={member}")
                                    else:
                                        if len(node['images']) < 20: node['images'].append(f"{rel_path}?member={member}")
                                    has_content = True
                    except Exception:
                        pass
                    if len(node['metadata']) < 20: node['metadata'].append(rel_path)
                    has_content = True
        
        if has_content:
            return node
        return None
    root_node = scan_dir_tree(dataset_path, '')
    if root_node:
        tree = root_node
    else:
        tree = {'images': [], 'masks': [], 'arrays': [], 'metadata': [], 'subdirs': {}}
        
    return jsonify(tree)

@app.route('/api/readme/<name>')
def api_readme(name):
    dataset_path = os.path.join(DATA_DIR, name)
    readme_path = request.args.get('path', '')
    if not readme_path:
        return jsonify({'error': 'No path provided'})
    full_path = os.path.abspath(os.path.join(dataset_path, readme_path))
    if not full_path.startswith(dataset_path) or not os.path.exists(full_path):
        return abort(404)
    
    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    if full_path.endswith('.md'):
        html = markdown.markdown(content)
        return jsonify({'html': html})
    else:
        return jsonify({'text': content})

@app.route('/api/csv/<name>')
def api_csv(name):
    dataset_path = os.path.join(DATA_DIR, name)
    csv_path = request.args.get('path', '')
    if not csv_path:
        return jsonify({'error': 'No path provided'})
    full_path = os.path.abspath(os.path.join(dataset_path, csv_path))
    if not full_path.startswith(dataset_path) or not os.path.exists(full_path):
        return abort(404)
    
    try:
        df = pd.read_csv(full_path, nrows=100)
        return jsonify({'columns': df.columns.tolist(), 'data': df.to_dict(orient='records')})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/mat/<name>')
def api_mat(name):
    dataset_path = os.path.join(DATA_DIR, name)
    mat_path = request.args.get('path', '')
    if not mat_path:
        return jsonify({'error': 'No path provided'})
    full_path = os.path.abspath(os.path.join(dataset_path, mat_path))
    if not full_path.startswith(dataset_path) or not os.path.exists(full_path):
        return abort(404)
    
    try:
        from scipy.io import loadmat
        mat = loadmat(full_path)
        info = {}
        for k, v in mat.items():
            if not k.startswith('__'):
                if isinstance(v, np.ndarray):
                    info[k] = f"Array of shape {v.shape} and type {v.dtype}"
                else:
                    info[k] = str(type(v))
        return jsonify({'info': info})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/data/<name>/<path:filepath>')
def serve_data(name, filepath):
    dataset_path = os.path.join(DATA_DIR, name)
    full_path = os.path.abspath(os.path.join(dataset_path, filepath))
    
    if not full_path.startswith(dataset_path) or not os.path.exists(full_path):
        return abort(404)
        
    ext = os.path.splitext(filepath)[1].lower()
    
    try:
        is_zip_member = (ext == '.zip' and request.args.get('member'))
        is_mask = ('mask' in filepath.lower() or 'seg' in filepath.lower())
        if ext in ['.tif', '.tiff'] or is_zip_member or (ext in ['.png', '.jpg', '.jpeg', '.bmp'] and is_mask):
            member = request.args.get('member')
            if ext == '.zip':
                import zipfile
                with zipfile.ZipFile(full_path, 'r') as z:
                    with z.open(member) as f:
                        img = Image.open(io.BytesIO(f.read()))
                filepath_for_logic = member
            else:
                img = Image.open(full_path)
                filepath_for_logic = filepath
            
            # If it's a mask with very small integer classes, normalize it
            if 'mask' in filepath_for_logic.lower() or 'seg' in filepath_for_logic.lower():
                arr = np.array(img)
                ptp = float(np.max(arr)) - float(np.min(arr))
                if ptp > 0 and arr.max() < 10:
                    arr = (arr - arr.min()) * (255.0 / ptp)
                    arr = arr.astype(np.uint8)
                    img = Image.fromarray(arr)
            
            if img.mode not in ('L', 'RGB', 'RGBA'):
                img = img.convert('RGBA')
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            buf.seek(0)
            return send_file(buf, mimetype='image/png')
            
        elif ext == '.npy':
            arr = np.load(full_path)
            arr = np.squeeze(arr)
            
            # Support requesting a specific slice index
            idx_str = request.args.get('index', '')
            if idx_str.isdigit():
                idx = int(idx_str)
                # Determine which axis is the batch/slice axis
                # Typically if shape is (N, H, W) or (N, H, W, C), the first axis is N.
                if arr.ndim > 2 and arr.shape[0] > 4:
                    if idx < arr.shape[0]:
                        arr = arr[idx]
                elif arr.ndim > 2 and arr.shape[-1] > 4:
                    # e.g. (H, W, N)
                    if idx < arr.shape[-1]:
                        arr = arr[..., idx]
            
            while arr.ndim > 2 and arr.shape[-1] not in [1, 3, 4]:
                if arr.shape[-1] > arr.shape[0]:
                    arr = arr[..., arr.shape[-1]//2]
                else:
                    arr = arr[arr.shape[0]//2]
            arr = np.nan_to_num(arr)
            ptp = float(np.max(arr)) - float(np.min(arr))
            if ptp != 0:
                arr = (arr - arr.min()) / ptp * 255.0
            arr = arr.astype(np.uint8)
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = np.squeeze(arr, axis=-1)
            img = Image.fromarray(arr)
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            buf.seek(0)
            return send_file(buf, mimetype='image/png')
            
        elif ext == '.mat':
            from scipy.io import loadmat
            mat = loadmat(full_path)
            
            array_name = request.args.get('array')
            if array_name and array_name in mat and isinstance(mat[array_name], np.ndarray) and mat[array_name].size > 0:
                arrays_to_check = [(array_name, mat[array_name])]
            else:
                arrays_to_check = mat.items()

            for k, v in arrays_to_check:
                if not k.startswith('__') and isinstance(v, np.ndarray) and v.size > 0:
                    arr = np.squeeze(v)
                    
                    idx_str = request.args.get('index', '')
                    if idx_str.isdigit():
                        idx = int(idx_str)
                        if arr.ndim > 2 and arr.shape[-1] > 4:
                            if idx < arr.shape[-1]:
                                arr = arr[..., idx]
                        elif arr.ndim > 2 and arr.shape[0] > 4:
                            if idx < arr.shape[0]:
                                arr = arr[idx]
                                
                    while arr.ndim > 2 and arr.shape[-1] not in [1, 3, 4]:
                        if arr.shape[-1] > arr.shape[0]:
                            arr = arr[..., arr.shape[-1]//2]
                        else:
                            arr = arr[arr.shape[0]//2]
                    if arr.ndim >= 2:
                        arr = np.nan_to_num(arr)
                        ptp = float(np.max(arr)) - float(np.min(arr))
                        if ptp != 0:
                            arr = (arr - arr.min()) / ptp * 255.0
                        arr = arr.astype(np.uint8)
                        if arr.ndim == 3 and arr.shape[-1] == 1:
                            arr = np.squeeze(arr, axis=-1)
                        img = Image.fromarray(arr)
                        buf = io.BytesIO()
                        img.save(buf, format='PNG')
                        buf.seek(0)
                        return send_file(buf, mimetype='image/png')
            
            # If no suitable array found, return generic mat info
            return send_from_directory(dataset_path, filepath)
            
    except Exception as e:
        print(f"Error serving {filepath}: {e}")
        # fallback
        pass
        
    return send_from_directory(dataset_path, filepath)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
