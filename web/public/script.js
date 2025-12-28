document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('gifInput');
    const browseBtn = document.getElementById('browseButton');
    const fileInfo = document.getElementById('fileInfo');
    const fileNameSpan = document.getElementById('fileName');
    const removeFileBtn = document.getElementById('removeFile');
    const convertBtn = document.getElementById('convertButton');
    const statusDiv = document.getElementById('status');
    const previewArea = document.getElementById('previewArea');
    const gifPreviewImg = document.getElementById('gifPreviewImg');
    const gifMeta = document.getElementById('gifMeta');
    const svgWrapper = document.getElementById('svgWrapper');
    const svgMeta = document.getElementById('svgMeta');
    const downloadLink = document.getElementById('downloadLink');

    let selectedFile = null;

    // Drag & Drop Events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('drag-over'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('drag-over'), false);
    });

    dropZone.addEventListener('drop', handleDrop, false);
    dropZone.addEventListener('click', () => fileInput.click());
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // Prevent triggering dropZone click
        fileInput.click();
    });

    fileInput.addEventListener('change', handleFiles);
    removeFileBtn.addEventListener('click', clearFile);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) {
            handleFileSelection(files[0]);
        }
    }

    function handleFiles() {
        if (fileInput.files.length) {
            handleFileSelection(fileInput.files[0]);
        }
    }

    function handleFileSelection(file) {
        if (file.type !== 'image/gif') {
            statusDiv.textContent = 'Error: Please select a valid GIF file.';
            statusDiv.style.color = 'var(--error-color)';
            return;
        }

        selectedFile = file;
        fileNameSpan.textContent = file.name;
        
        // UI Updates
        dropZone.classList.add('hidden');
        fileInfo.classList.remove('hidden');
        convertBtn.disabled = false;
        statusDiv.textContent = '';
        
        // Preview GIF
        const reader = new FileReader();
        reader.onload = (e) => {
            gifPreviewImg.src = e.target.result;
            previewArea.classList.remove('hidden');
            // Reset SVG part
            svgWrapper.innerHTML = '';
            svgMeta.textContent = '';
            downloadLink.classList.add('hidden');
            
            // Calculate size
            const sizeKB = (file.size / 1024).toFixed(1);
            gifMeta.textContent = `Size: ${sizeKB} KB`;
        };
        reader.readAsDataURL(file);
    }

    function clearFile() {
        selectedFile = null;
        fileInput.value = '';
        dropZone.classList.remove('hidden');
        fileInfo.classList.add('hidden');
        convertBtn.disabled = true;
        previewArea.classList.add('hidden');
        statusDiv.textContent = '';
        downloadLink.classList.add('hidden');
    }

    // Conversion Logic
    convertBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        statusDiv.textContent = 'Converting... This may take a moment.';
        statusDiv.style.color = 'var(--text-color)';
        convertBtn.disabled = true;
        svgWrapper.innerHTML = '<div style="padding:20px;">Processing...</div>';

        const reader = new FileReader();
        reader.readAsDataURL(selectedFile);
        reader.onload = async () => {
            const base64Data = reader.result;

            // Collect params
            const params = {};
            const fpsVal = document.getElementById('fps').value;
            if (fpsVal) params.fps = fpsVal;

            const ids = [
                'colormode', 'hierarchical', 'mode', 'filter_speckle',
                'color_precision', 'layer_difference', 'corner_threshold',
                'length_threshold', 'max_iterations', 'splice_threshold',
                'path_precision'
            ];

            ids.forEach(id => {
                const el = document.getElementById(id);
                if (el.value !== '') params[id] = el.value;
            });

            const requestBody = JSON.stringify({ file: base64Data, params });
            
            // Check size limit (approximate)
            if (requestBody.length > 4.5 * 1024 * 1024) {
                statusDiv.textContent = 'Error: File too large for web demo (Limit 4.5MB). Use CLI tool.';
                statusDiv.style.color = 'var(--error-color)';
                convertBtn.disabled = false;
                svgWrapper.innerHTML = '';
                return;
            }

            try {
                const response = await fetch('/api/convert', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: requestBody,
                });

                const contentType = response.headers.get("content-type");
                
                if (response.ok && contentType && contentType.includes("application/json")) {
                    const data = await response.json();
                    
                    // Display SVG
                    svgWrapper.innerHTML = data.svg;
                    
                    // Create Blob for download
                    const blob = new Blob([data.svg], { type: 'image/svg+xml' });
                    const blobURL = URL.createObjectURL(blob);
                    const sizeKB = (blob.size / 1024).toFixed(1);
                    
                    svgMeta.textContent = `Size: ${sizeKB} KB`;
                    downloadLink.href = blobURL;
                    downloadLink.download = selectedFile.name.replace(/\.gif$/i, '.svg');
                    downloadLink.classList.remove('hidden');
                    
                    statusDiv.textContent = 'Conversion Successful!';
                    statusDiv.style.color = 'var(--success-color)';
                } else {
                    let errorMsg = 'Unknown error occurred.';
                    if (contentType && contentType.includes("application/json")) {
                        const errData = await response.json();
                        errorMsg = errData.error || errorMsg;
                    } else {
                        errorMsg = await response.text();
                    }
                    throw new Error(errorMsg);
                }
            } catch (error) {
                console.error(error);
                statusDiv.textContent = `Error: ${error.message}`;
                statusDiv.style.color = 'var(--error-color)';
                svgWrapper.innerHTML = '';
            } finally {
                convertBtn.disabled = false;
            }
        };
    });
});
