import os
import sys
import socket
from flask import Flask, jsonify, send_from_directory

app = Flask(__name__)

WORKSPACE_ROOT = "/home/mmk/Codes/oct_data_synthesis"

# Excluded directories for scanning to keep it fast
EXCLUDED_DIRS = {'.git', 'mlruns', '.gemini', '.agents', '__pycache__', 'node_modules', '.venv', 'venv', 'env'}
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}

def get_image_list():
    image_files = []
    for root, dirs, files in os.walk(WORKSPACE_ROOT):
        # Modify dirs in-place to skip excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in ALLOWED_EXTENSIONS:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, WORKSPACE_ROOT)
                image_files.append(rel_path)
    # Sort files naturally
    image_files.sort()
    return image_files

@app.route('/')
def index():
    # Read templates/index.html directly from the templates subdirectory
    html_path = os.path.join(os.path.dirname(__file__), 'templates', 'index.html')
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "templates/index.html not found. Please create the frontend first.", 404

@app.route('/api/images')
def api_images():
    try:
        images = get_image_list()
        return jsonify({"success": True, "images": images})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/image/<path:rel_path>')
def api_image(rel_path):
    try:
        # Prevent path traversal attacks
        abs_path = os.path.abspath(os.path.join(WORKSPACE_ROOT, rel_path))
        if not abs_path.startswith(WORKSPACE_ROOT):
            return "Access denied", 403
        
        if not os.path.exists(abs_path):
            return "File not found", 404
            
        directory = os.path.dirname(abs_path)
        filename = os.path.basename(abs_path)
        return send_from_directory(directory, filename)
    except Exception as e:
        return str(e), 500

def find_free_port(start_port=5000):
    port = start_port
    while port < 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                port += 1
    return start_port

if __name__ == '__main__':
    port = find_free_port(5000)
    print("\n" + "="*60)
    print(f"OCT Intensity Profiler Server starting...")
    print(f"Local URL: http://localhost:{port}")
    print(f"To access this app from your local machine, use SSH Port Forwarding:")
    print(f"  ssh -L {port}:localhost:{port} <username>@<server_ip>")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=port, debug=False)
