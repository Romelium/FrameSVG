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

            const contentType = response.headers.get("content-type");
            if (response.ok) {
                if (contentType && contentType.includes("application/json")) {
                    const data = await response.json();
                    statusDiv.textContent = 'Conversion successful!';
                    const blob = new Blob([data.svg], { type: 'image/svg+xml' });
                    const blobURL = URL.createObjectURL(blob);

                    const img = document.createElement('img');
                    img.src = blobURL;
                    previewDiv.appendChild(img);

                    downloadLink.href = blobURL;
                    downloadLink.download = 'animation.svg';
                    downloadLink.style.display = 'block';
                } else {
                    statusDiv.textContent = "Unexpected response type from server.";
                }
            } else {
                if (contentType && contentType.includes("application/json")) {
                    const data = await response.json();
                    statusDiv.textContent = `Error: ${data.error}`;
                } else if (contentType && contentType.includes("text/plain")) {
                    const errorText = await response.text();
                    statusDiv.textContent = `Error: ${errorText}`;
                } else if (contentType && contentType.includes("text/html")) {
                    const errorText = await response.text();
                    let errorMessage = "";

                    const explanationMatch = errorText.match(/Error code explanation: \d+ - (.*)<\/p>/);
                    if (explanationMatch) {
                        errorMessage = explanationMatch[1];
                    } else {
                        const messageMatch = errorText.match(/Message: (.*?)<\/p>/);
                        if (messageMatch) {
                            errorMessage = messageMatch[1];
                        } else {
                            errorMessage = "An HTML error occurred, but could not extract the message.";
                        }
                    }
                    statusDiv.textContent = `Error: ${errorMessage}`;
                }
                else {
                    statusDiv.textContent = `An error occurred: ${response.status} ${response.statusText}`;
                }
            }
        } catch (error) {
            statusDiv.textContent = `An error occurred: ${error}`;
        }
    };

    reader.onerror = () => {
        statusDiv.textContent = 'Error reading file.';
    }
});
