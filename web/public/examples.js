async function getFileSize(url, element) {
    try {
        const response = await fetch(url, { method: 'HEAD' });
        if (response.ok) {
            const contentLength = response.headers.get('Content-Length');
            if (contentLength) {
                const sizeInKB = (parseInt(contentLength) / 1024).toFixed(1); // Convert to KB, one decimal place
                element.textContent = `${sizeInKB} KB`;
            } else {
                element.textContent = 'Size N/A';
            }
        } else {
            element.textContent = 'Error';
        }
    } catch (error) {
        console.error('Error fetching file size:', error);
        element.textContent = 'Error';
    }
}

// Select all example groups
const exampleGroups = document.querySelectorAll('.example-group');

exampleGroups.forEach(async group => {
    // Within each group, find the GIF and SVG elements
    const gifExample = group.querySelector('.example:first-child');
    const svgExample = group.querySelector('.example:last-child');

    // Get the image URL and the span element for the GIF
    if (gifExample) {
        const gifImg = gifExample.querySelector('img');
        const gifSizeSpan = gifExample.querySelector('p > span');
        if (gifImg && gifSizeSpan) {
            getFileSize(gifImg.src, gifSizeSpan);
        }
    }


    // Get the image URL and the span element for the SVG
    if (svgExample) {
        const svgImg = svgExample.querySelector('img');
        const svgSizeSpan = svgExample.querySelector('p > span');
        if (svgImg && svgSizeSpan) {
            getFileSize(svgImg.src, svgSizeSpan);
        }
    }
});
