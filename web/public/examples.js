async function getUncompressedSize(url, element) {
    const fullResponse = await fetch(url);
    if (!fullResponse.ok) return -1;
    const blob = await fullResponse.blob();
    const uncompressedSizeInKB = (blob.size / 1024).toFixed(1);
    element.textContent += ` | ${uncompressedSizeInKB} KB (raw)`;
}

async function getFileSize(url, element, already_uncompressed = false) {
    try {
        const response = await fetch(url, { method: 'HEAD' });
        if (!response.ok) {
            element.textContent = 'Error';
            return;
        }

        const contentLength = response.headers.get('Content-Length');
        if (contentLength) {
            const sizeInKB = (parseInt(contentLength) / 1024).toFixed(1);
            element.textContent = already_uncompressed ? `${sizeInKB} KB` : `${sizeInKB} KB (comp)`;
        }

        if (!already_uncompressed) getUncompressedSize(url, element);
    } catch {
        element.textContent = 'Error';
    }
}

document.querySelectorAll('.example-group').forEach(async group => {
    const gifExample = group.querySelector('.example:first-child');
    const svgExample = group.querySelector('.example:last-child');

    if (gifExample) {
        const gifImg = gifExample.querySelector('img');
        const gifSizeSpan = gifExample.querySelector('p > span');
        if (gifImg && gifSizeSpan) getFileSize(gifImg.src, gifSizeSpan, true);
    }

    if (svgExample) {
        const svgImg = svgExample.querySelector('img');
        const svgSizeSpan = svgExample.querySelector('p > span');
        if (svgImg && svgSizeSpan) getFileSize(svgImg.src, svgSizeSpan, false);
    }
});
