document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('gifInput');
    const browseBtn = document.getElementById('browseButton');
    const fileInfo = document.getElementById('fileInfo');
    const fileNameSpan = document.getElementById('fileName');
    const removeFileBtn = document.getElementById('removeFile');
    const convertBtn = document.getElementById('convertButton');
    const statusText = document.getElementById('statusText');
    const loadingSpinner = document.getElementById('loadingSpinner');
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
    removeFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        clearFile();
    });

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
        // Check mime type or extension
        if (file.type !== 'image/gif' && !file.name.toLowerCase().endsWith('.gif')) {
            showStatus('Error: Please select a valid GIF file.', 'error');
            return;
        }

        selectedFile = file;
        fileNameSpan.textContent = file.name;

        // UI Updates
        dropZone.classList.add('hidden');
        fileInfo.classList.remove('hidden');
        convertBtn.disabled = false;
        showStatus('');

        // Preview GIF
        const reader = new FileReader();
        reader.onload = (e) => {
            gifPreviewImg.src = e.target.result;
            previewArea.classList.remove('hidden');
            // Reset SVG part
            svgWrapper.innerHTML = '<div style="color: var(--text-secondary); font-style: italic;">Waiting for conversion...</div>';
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
        showStatus('');
        downloadLink.classList.add('hidden');
    }

    function showStatus(message, type = 'normal') {
        statusText.textContent = message;
        if (type === 'error') {
            statusText.style.color = 'var(--error-color)';
            loadingSpinner.classList.add('hidden');
        } else if (type === 'success') {
            statusText.style.color = 'var(--success-color)';
            loadingSpinner.classList.add('hidden');
        } else if (type === 'loading') {
            statusText.style.color = 'var(--text-color)';
            loadingSpinner.classList.remove('hidden');
        } else {
            statusText.style.color = 'var(--text-color)';
            loadingSpinner.classList.add('hidden');
        }
    }

    // Conversion Logic
    convertBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        showStatus('Converting... This may take a moment.', 'loading');
        convertBtn.disabled = true;
        svgWrapper.innerHTML = '<div style="padding:20px; color: var(--text-secondary);">Processing...</div>';
        downloadLink.classList.add('hidden');

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
                showStatus('Error: File too large for web demo (Limit 4.5MB). Use CLI tool.', 'error');
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

                    showStatus('Conversion Successful!', 'success');
                } else {
                    let errorMsg = 'Unknown error occurred.';
                    if (contentType && contentType.includes("application/json")) {
                        const errData = await response.json();
                        errorMsg = errData.error || errorMsg;
                    } else {
                        const text = await response.text();
                        if (text.trim().startsWith('<')) {
                            console.error("Server returned HTML:", text);
                            errorMsg = `Server Error (${response.status})`;
                        } else {
                            errorMsg = text;
                        }
                    }
                    throw new Error(errorMsg);
                }
            } catch (error) {
                console.error(error);
                showStatus(`Error: ${error.message}`, 'error');
                svgWrapper.innerHTML = '<div style="color: var(--error-color);">Conversion failed</div>';
            } finally {
                convertBtn.disabled = false;
            }
        };
    });
});
