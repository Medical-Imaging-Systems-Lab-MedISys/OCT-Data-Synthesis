document.addEventListener('DOMContentLoaded', () => {
    const datasetList = document.getElementById('dataset-list');
    const currentTitle = document.getElementById('current-dataset-title');
    const emptyState = document.getElementById('empty-state');
    const loadingState = document.getElementById('loading-state');
    const datasetView = document.getElementById('dataset-view');
    
    let currentDataset = null;
    let datasetData = null;

    fetch('/api/datasets')
        .then(res => res.json())
        .then(datasets => {
            datasets.forEach(ds => {
                const li = document.createElement('li');
                li.className = 'dataset-item';
                li.textContent = ds;
                li.onclick = () => selectDataset(ds, li);
                datasetList.appendChild(li);
            });
        });

    function hideAll() {
        emptyState.classList.add('hidden');
        loadingState.classList.add('hidden');
        datasetView.classList.add('hidden');
        datasetView.innerHTML = ''; // clear dynamic content
    }

    function selectDataset(name, element) {
        document.querySelectorAll('.dataset-item').forEach(el => el.classList.remove('active'));
        if(element) element.classList.add('active');
        
        currentDataset = name;
        currentTitle.textContent = name;
        
        hideAll();
        loadingState.classList.remove('hidden');

        fetch(`/api/dataset/${name}`)
            .then(res => res.json())
            .then(data => {
                datasetData = data;
                renderDataset();
            })
            .catch(err => {
                console.error(err);
                loadingState.classList.add('hidden');
                emptyState.classList.remove('hidden');
            });
    }

    function renderDataset() {
        loadingState.classList.add('hidden');
        datasetView.classList.remove('hidden');
        
        if (datasetData.is_oct5k) {
            let html = `<div class="section-card glass-panel tree-dir">
                <h3 class="section-title">OCT5k Dataset</h3>
                <p>CSV Files Count: ${datasetData.csv_files.length}</p>
                <details style="margin-top: 15px;">
                    <summary style="cursor: pointer; font-weight: bold;">Show CSV Files</summary>
                    <ul style="margin-top: 10px; max-height: 400px; overflow-y: auto;">
                        ${datasetData.csv_files.map(f => `<li>${f}</li>`).join('')}
                    </ul>
                </details>
            </div>`;
            html += renderTree('/', datasetData);
            datasetView.innerHTML = html;
            return;
        }

        // Render tree recursively
        const treeHtml = renderTree('/', datasetData);
        datasetView.innerHTML = treeHtml;
    }

    function renderTree(dirName, node) {
        if (!node) return '';
        
        let html = `<div class="section-card glass-panel tree-dir">`;
        html += `<h3 class="section-title">📁 ${dirName}</h3>`;
        
        if (node.metadata && node.metadata.length > 0) {
            html += `<div class="sub-section"><h4>Metadata (${node.metadata.length})</h4><ul class="meta-list">`;
            node.metadata.forEach(m => {
                html += `<li>${m}</li>`;
            });
            html += `</ul></div>`;
        }

        if (node.images && node.images.length > 0) {
            html += `<div class="sub-section"><h4>Images</h4><div class="gallery-grid">`;
            node.images.slice(0, 10).forEach(img => {
                html += `<div class="gallery-item"><img src="/data/${currentDataset}/${img}" loading="lazy"><div class="overlay">${img.split('/').pop()}</div></div>`;
            });
            html += `</div></div>`;
        }

        if (node.masks && node.masks.length > 0) {
            html += `<div class="sub-section"><h4>Masks</h4><div class="gallery-grid">`;
            node.masks.slice(0, 10).forEach(img => {
                html += `<div class="gallery-item"><img src="/data/${currentDataset}/${img}" loading="lazy"><div class="overlay">${img.split('/').pop()}</div></div>`;
            });
            html += `</div></div>`;
        }
        
        if (node.arrays && node.arrays.length > 0) {
            html += `<div class="sub-section"><h4>Arrays (MAT/NPY)</h4><div class="gallery-grid">`;
            node.arrays.slice(0, 10).forEach(arr => {
                html += `<div class="gallery-item"><img src="/data/${currentDataset}/${arr}" loading="lazy"><div class="overlay">${arr.split('/').pop()}</div></div>`;
            });
            html += `</div></div>`;
        }
        
        if (node.subdirs && Object.keys(node.subdirs).length > 0) {
            html += `<div class="subdirs">`;
            Object.keys(node.subdirs).sort().forEach(subDirName => {
                html += renderTree(subDirName, node.subdirs[subDirName]);
            });
            html += `</div>`;
        }
        
        html += `</div>`;
        return html;
    }

    const themeBtn = document.getElementById('theme-toggle');
    if (themeBtn) {
        themeBtn.addEventListener('click', () => {
            document.body.style.filter = document.body.style.filter === 'hue-rotate(90deg)' ? 'none' : 'hue-rotate(90deg)';
        });
    }
});
