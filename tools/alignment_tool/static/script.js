const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const select = document.getElementById('image-select');

const inputs = {
    x: document.getElementById('offset-x'),
    y: document.getElementById('offset-y'),
    sx: document.getElementById('scale-x'),
    sy: document.getElementById('scale-y'),
    r: document.getElementById('rotate'),
    fx: document.getElementById('flip-x'),
    fy: document.getElementById('flip-y')
};

const vals = {
    x: document.getElementById('val-x'),
    y: document.getElementById('val-y'),
    sx: document.getElementById('val-sx'),
    sy: document.getElementById('val-sy'),
    r: document.getElementById('val-r')
};

let currentImgObj = null;
let currentData = null;
let currentDirName = null;

const colors = ["red", "lime", "cyan", "yellow", "magenta", "orange", "blue", "purple"];

// Init
fetch('/api/images')
    .then(r => r.json())
    .then(images => {
        images.forEach(img => {
            const opt = document.createElement('option');
            opt.value = img.dir_name;
            opt.textContent = img.img_name;
            select.appendChild(opt);
        });
        if(images.length > 0) {
            loadSelectedImage();
        }
    });

select.addEventListener('change', loadSelectedImage);

Object.keys(inputs).forEach(k => {
    inputs[k].addEventListener('input', () => {
        if (vals[k]) vals[k].textContent = inputs[k].value;
        draw();
    });
});

document.getElementById('save-btn').addEventListener('click', () => {
    if(!currentDirName) return;
    const payload = {
        offset_x: parseFloat(inputs.x.value),
        offset_y: parseFloat(inputs.y.value),
        scale_x: parseFloat(inputs.sx.value),
        scale_y: parseFloat(inputs.sy.value),
        rotate: parseFloat(inputs.r.value),
        flip_x: inputs.fx.checked,
        flip_y: inputs.fy.checked
    };
    
    fetch(`/api/save/${currentDirName}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    }).then(r => r.json()).then(res => {
        const msg = document.getElementById('status-msg');
        msg.textContent = "Saved successfully!";
        setTimeout(() => msg.textContent = "", 2000);
    });
});

function loadSelectedImage() {
    currentDirName = select.value;
    const imgName = select.options[select.selectedIndex].text;
    
    // Load data
    fetch(`/api/data/${currentDirName}`)
        .then(r => r.json())
        .then(data => {
            currentData = data;
            
            // Set UI to saved alignment
            inputs.x.value = data.alignment.offset_x;
            inputs.y.value = data.alignment.offset_y;
            inputs.sx.value = data.alignment.scale_x;
            inputs.sy.value = data.alignment.scale_y;
            inputs.r.value = data.alignment.rotate || 0;
            inputs.fx.checked = data.alignment.flip_x || false;
            inputs.fy.checked = data.alignment.flip_y || false;
            
            Object.keys(inputs).forEach(k => { if(vals[k]) vals[k].textContent = inputs[k].value; });
            
            // Load image
            const img = new Image();
            img.src = `/data/${currentDirName}/${imgName}`;
            img.onload = () => {
                currentImgObj = img;
                canvas.width = img.width;
                canvas.height = img.height;
                draw();
            };
        });
}

function draw() {
    if(!currentImgObj || !currentData) return;
    
    // Clear & draw image
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(currentImgObj, 0, 0);
    
    // Get adjustments
    const ox = parseFloat(inputs.x.value);
    const oy = parseFloat(inputs.y.value);
    const sx = parseFloat(inputs.sx.value);
    const sy = parseFloat(inputs.sy.value);
    const r = parseFloat(inputs.r.value) * Math.PI / 180;
    const fx = inputs.fx.checked ? -1 : 1;
    const fy = inputs.fy.checked ? -1 : 1;
    
    const max_y_raw = currentData.global_max_y;
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    
    // Draw paths
    ctx.lineWidth = 2;
    currentData.layers.forEach(layer => {
        ctx.strokeStyle = colors[layer.layer_idx % colors.length];
        ctx.beginPath();
        let started = false;
        
        for(let i=0; i<layer.x.length; i++) {
            if(layer.x[i] !== null && layer.y[i] !== null && !isNaN(layer.x[i]) && !isNaN(layer.y[i])) {
                // Exact alignment from MATLAB parameters
                let bx = (layer.y[i] / currentData.resize_scale) + currentData.min_xrange - 1;
                let by = (layer.x[i] / currentData.resize_scale) + currentData.min_yrange - 1;
                
                // User scaling & flipping (around center)
                let tx = (bx - cx) * sx * fx;
                let ty = (by - cy) * sy * fy;
                
                // User rotation
                let rx = tx * Math.cos(r) - ty * Math.sin(r);
                let ry = tx * Math.sin(r) + ty * Math.cos(r);
                
                // Translate back & add offset
                let final_x = rx + cx + ox;
                let final_y = ry + cy + oy;
                
                if(!started) {
                    ctx.moveTo(final_x, final_y);
                    started = true;
                } else {
                    ctx.lineTo(final_x, final_y);
                }
            }
        }
        ctx.stroke();
    });
}
