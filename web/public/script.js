document.getElementById('convertButton').addEventListener('click', async () => {
    const fileInput = document.getElementById('gifInput');
    const statusDiv = document.getElementById('status');
    const previewDiv = document.getElementById('preview');
    const downloadLink = document.getElementById('downloadLink');

    // Clear previous results
    statusDiv.textContent = '';
    previewDiv.innerHTML = '';
    downloadLink.style.display = 'none';


    if (!fileInput.files || fileInput.files.length === 0) {
        statusDiv.textContent = 'Please select a GIF file.';
        return;
    }

    const file = fileInput.files[0];
    if (file.type !== 'image/gif') {
        statusDiv.textContent = 'Please select a valid GIF file.';
        return;
    }

    statusDiv.textContent = 'Converting... Please wait.';

    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = async () => {
        const base64Data = reader.result;

        const params = {
            fps: document.getElementById('fps').value,
            colormode: document.getElementById('colormode').value,
            hierarchical: document.getElementById('hierarchical').value,
            mode: document.getElementById('mode').value,
            filter_speckle: document.getElementById('filter_speckle').value,
            color_precision: document.getElementById('color_precision').value,
            layer_difference: document.getElementById('layer_difference').value,
            corner_threshold: document.getElementById('corner_threshold').value,
            length_threshold: document.getElementById('length_threshold').value,
            max_iterations: document.getElementById('max_iterations').value,
            splice_threshold: document.getElementById('splice_threshold').value,
            path_precision: document.getElementById('path_precision').value,
        };

        try {
            const response = await fetch('/api/convert', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ file: base64Data, params }),
            });

            const data = await response.json();

            if (response.ok) {
                statusDiv.textContent = 'Conversion successful!';
                const blob = new Blob([data.svg], { type: 'image/svg+xml' });
                const blobURL = URL.createObjectURL(blob);

                // Preview
                const img = document.createElement('img');
                img.src = blobURL;
                previewDiv.appendChild(img);

                // Download link
                downloadLink.href = blobURL;
                downloadLink.download = 'animation.svg'; // Set filename
                downloadLink.style.display = 'block';

            } else {
                statusDiv.textContent = `Error: ${data.error}`;
            }
        } catch (error) {
            statusDiv.textContent = `An error occurred: ${error}`;
        }
    };

    reader.onerror = () => {
        statusDiv.textContent = 'Error reading file.';
    }
});
