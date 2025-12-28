<h1 align="center">
  <img height="250" src="images/framesvg.svg" alt="FrameSVG Logo" />
</h1>

<p align="center">
  <strong>Convert animated GIFs to animated SVGs.</strong>
</p>

<p align="center">
  <a href="https://framesvg.romelium.cc"><strong>🌐 Try the Web App</strong></a>
  <br/>
  <br/>
  <a href="https://github.com/romelium/framesvg/actions/workflows/tests.yml"><img alt="Build Status" src="https://img.shields.io/github/actions/workflow/status/romelium/framesvg/tests.yml?style=flat-square"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square"></a>
  <a href="https://pypi.org/project/framesvg/"><img alt="PyPI Version" src="https://img.shields.io/pypi/v/framesvg?style=flat-square"></a>
  <a href="https://pypi.org/project/framesvg/"><img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/framesvg?style=flat-square"></a>
</p>

---

<!-- Kyubey Animation Block -->
<img align="left" hspace="10" src="images/kyubey.svg" alt="Kyubey SVG Image"/>
<img align="right" hspace="10" src="images/kyubey.svg" alt="Kyubey SVG Image"/>

`framesvg` is a [web app](https://framesvg.romelium.cc), command-line tool, and Python library that converts animated GIFs into animated SVGs. It leverages the power of [VTracer](https://www.visioncortex.org/vtracer/) for raster-to-vector conversion, producing smooth, scalable, and *true vector* animations.

This is a significant improvement over embedding raster images (like GIFs) directly within SVGs, as `framesvg` generates genuine vector output that plays automatically and scales beautifully. Ideal for readmes, documentation, and web graphics.

<br clear="both"/>

## 📑 Table of Contents

- [Features](#-features)
- [Examples](#-examples)
  - [Coding Examples](#coding-examples)
  - [General Examples](#general-examples)
  - [Transparent/Binary Examples](#transparentbinary-examples)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Command Line Interface](#command-line-interface)
  - [Python Library](#python-library)
- [Configuration & Options](#-configuration--options)
- [API Reference](#-api-reference)
- [Optimization Guide](#-optimization-guide)
- [Development](#-development)

## ✨ Features

*   **True Vector Output:** Creates scalable vector graphics, not embedded rasters.
*   **Automatic Playback:** Generated SVGs play automatically in browsers and image viewers.
*   **Network Efficiency:** SVGs are text-based. When served with Gzip or Brotli compression, they can load significantly faster than GIFs.
*   **Flexible Tooling:** Available as a [Web App](https://framesvg.romelium.cc), CLI tool, and Python library.
*   **Highly Configurable:** Fine-tune the vectorization process (speckle filtering, smoothing, color precision) to balance quality vs. file size.
*   **Variable Frame Rate Support:** Preserves the specific duration of each frame from the original GIF.

## 🎨 Examples

> **Note:** You can view a live comparison gallery on the [Examples Page](https://framesvg.romelium.cc/examples.html).

### Coding Examples

| Original GIF | Converted SVG |
| :---: | :---: |
| <img src="images/Code Coding GIF by EscuelaDevRock Git.gif" height="200" alt="Git GIF"> | <img src="images/Code Coding GIF by EscuelaDevRock Git.svg" height="200" alt="Git SVG"> |
| <img src="images/Code Coding GIF by EscuelaDevRock GitHub.gif" height="200" alt="GitHub GIF"> | <img src="images/Code Coding GIF by EscuelaDevRock GitHub.svg" height="200" alt="GitHub SVG"> |
| <img src="images/Code Coding GIF by EscuelaDevRock VSCode.gif" height="200" alt="VSCode GIF"> | <img src="images/Code Coding GIF by EscuelaDevRock VSCode.svg" height="200" alt="VSCode SVG"> |
| <img src="images/Code Coding GIF by EscuelaDevRock Sublime.gif" height="200" alt="Sublime GIF"> | <img src="images/Code Coding GIF by EscuelaDevRock Sublime.svg" height="200" alt="Sublime SVG"> |

### General Examples

| Original GIF | Converted SVG |
| :---: | :---: |
| <img src="images/Good_Morning_GIF_by_Hello_All.gif" height="200" alt="Good Morning GIF"> | <img src="images/Good_Morning_GIF_by_Hello_All.svg" height="200" alt="Good Morning SVG"> |
| <img src="images/icon_loading_GIF.gif" height="200" alt="Loading GIF"> | <img src="images/icon_loading_GIF.svg" height="200" alt="Loading SVG"> |
| <img src="images/voila hands GIF by brontron.gif" height="200" alt="Voila GIF"> | <img src="images/voila hands GIF by brontron.svg" height="200" alt="Voila SVG"> |

### Transparent/Binary Examples

These examples use the `binary` color mode. All bright colors turn transparent. (If they appear dark here, it is due to the transparency on a dark background; they look correct on light backgrounds).

| Original GIF | Converted SVG |
| :---: | :---: |
| <img src="images/black and white loop GIF by Sculpture.gif" height="200" alt="Loop GIF"> | <img src="images/black and white loop GIF by Sculpture.svg" height="200" alt="Loop SVG"> |
| <img src="images/Black And White Loop GIF by Pi-Slices.gif" height="200" alt="Pi-Slices GIF"> | <img src="images/Black And White Loop GIF by Pi-Slices.svg" height="200" alt="Pi-Slices SVG"> |

## 📦 Installation

### Option 1: Using pip (Recommended for Library use)

```bash
pip install framesvg
```

### Option 2: Using [pipx](https://pipx.pypa.io/latest/installation/) (Recommended for CLI use)

If you only need the command-line tool, `pipx` installs it in an isolated environment.

```bash
pipx install framesvg
```

## 🚀 Usage

### Command-Line Interface

Basic conversion:
```bash
framesvg input.gif
```

Specify output filename and force 24 FPS:
```bash
framesvg input.gif output.svg --fps 24
```

Optimize for file size (High filtering, low precision):
```bash
framesvg input.gif --filter-speckle 10 --color-precision 4 --mode polygon
```

### Python Library

```python
from framesvg import gif_to_animated_svg_write, gif_to_animated_svg

# 1. Convert and save directly to a file
gif_to_animated_svg_write("input.gif", "output.svg", fps=30)

# 2. Get SVG content as a string (useful for web apps)
svg_content = gif_to_animated_svg("input.gif")
print(f"Generated SVG size: {len(svg_content)} bytes")

# 3. Advanced: Custom VTracer options
options = {
    "mode": "spline",       # Smoother curves
    "filter_speckle": 2,    # Keep more detail
    "colormode": "binary"   # Black and white
}
gif_to_animated_svg_write("input.gif", "artistic.svg", vtracer_options=options)
```

## ⚙️ Configuration & Options

These options control the VTracer engine. Tweak these to balance **Visual Fidelity** vs. **File Size**.

| Flag | Option | Default | Description |
| :--- | :--- | :--- | :--- |
| `-f` | `--fps` | *None* | Animation speed. Defaults to GIF frame delays; falls back to 10. |
| `-c` | `--colormode` | `color` | `color` (standard) or `binary` (threshold-based transparency). |
| `-m` | `--mode` | `polygon` | `polygon` (smaller files), `spline` (smoother curves), or `none`. |
| `-s` | `--filter-speckle` | `4` | **Crucial.** Cleans up noise. Higher = smaller file, less detail. |
| `-p` | `--color-precision` | `8` | Bit-depth of colors. Lower = fewer colors, smaller file. |
| `-d` | `--layer-difference` | `16` | Gradient threshold. Higher = fewer layers, smaller file. |
| `-i` | `--hierarchical` | `stacked` | `stacked` (shapes on top of others) or `cutout`. |
| | `--corner-threshold` | `60` | Minimum angle to be considered a corner. |
| | `--length-threshold` | `4.0` | Minimum path length to render. |
| | `--splice-threshold` | `45` | Angle threshold for splitting splines. |

## 📚 API Reference

### `gif_to_animated_svg`

```python
def gif_to_animated_svg(
    gif_path: str,
    vtracer_options: dict | None = None,
    fps: float | None = None
) -> str
```

Reads a GIF file and returns the animated SVG content as a string.

*   **`gif_path`**: Path to the source GIF.
*   **`vtracer_options`**: A dictionary of VTracer parameters. Keys correspond to the CLI flags above, but use underscores instead of hyphens (e.g., `filter_speckle`, `color_precision`).
*   **`fps`**: Force a specific frame rate. If `None`, uses the GIF's original frame delays.

### `gif_to_animated_svg_write`

```python
def gif_to_animated_svg_write(
    gif_path: str,
    output_svg_path: str,
    vtracer_options: dict | None = None,
    fps: float | None = None
) -> None
```

Reads a GIF file, converts it, and saves the result directly to `output_svg_path`.

## 📉 Optimization Guide

Vectorizing complex raster images can result in large files. Here is how to keep file size under control:

1.  **Increase Speckle Filter (`-s`):** This is the most effective setting. Try increasing it to `10` or higher to remove small "dust" artifacts.
2.  **Use Polygon Mode (`-m polygon`):** Splines look smoother but require more data. Polygons are often sufficient for animation and significantly smaller.
3.  **Reduce Color Precision (`-p`):** Dropping this to `6` or `4` reduces the palette size, merging similar colors.
4.  **Simplify the Source:** If possible, reduce the noise or dither in the source GIF before converting.

> **Tip:** Use the [VTracer Online Demo](https://www.visioncortex.org/vtracer/) to visualize how these parameters affect a single frame before converting the whole animation.

## 🛠 Development

### Setup

We use [Hatch](https://hatch.pypa.io/latest/install) for dependency management.

```bash
# Install Hatch
pip install hatch

# Run tests
hatch test

# Format and Lint
hatch fmt
```

### Web App Development

The web interface is built with vanilla HTML/JS and Python serverless functions (Vercel).

1.  **Install Vercel CLI:** `npm install -g vercel`
2.  **Run Locally:**
    ```bash
    vercel dev
    ```
    *Note: Set the root directory to `./web` when prompted.*

## 🤝 Contributing

Contributions are welcome! Please submit pull requests or open issues on the [GitHub repository](https://github.com/romelium/framesvg).

## 📄 License

MIT License © 2025 Romelium
