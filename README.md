<h1 align="center">
  <img height="250" src="images/framesvg.svg" alt="FrameSVG Logo" />
</h1>

<p align="center">
  <strong>Convert animated GIFs to animated SVGs.</strong>
</p>

<p align="center">
  <a href="https://github.com/romelium/framesvg/actions/workflows/tests.yml"><img alt="Build Status: Passing" src="https://img.shields.io/github/actions/workflow/status/romelium/framesvg/tests.yml"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <a href="https://pypi.org/project/framesvg/"><img alt="PyPI Version" src="https://img.shields.io/pypi/v/framesvg"></a>
  <a href="https://pypi.org/project/framesvg/"><img alt="Python Versions Supported" src="https://img.shields.io/pypi/pyversions/framesvg"></a>
</p>

<!-- Be on the left and right of next paragraph -->
<img align="left" hspace=10 src="images/kyubey.svg" alt="Kyubey SVG Image"/>
<img align="right" hspace=10 src="images/kyubey.svg" alt="Kyubey SVG Image"/>

`framesvg` is a [web app](https://framesvg.romelium.cc), command-line tool, and Python library that converts animated GIFs into animated SVGs. It leverages the power of [VTracer](https://www.visioncortex.org/vtracer/) for raster-to-vector conversion, producing smooth, scalable, and *true vector* animations.  This is a significant improvement over embedding raster images (like GIFs) directly within SVGs, as `framesvg` generates genuine vector output that plays automatically and scales beautifully.  Ideal for readmes, documentation, and web graphics.

You can try it now at [framesvg.romelium.cc](https://framesvg.romelium.cc)

<br clear="both"/>

## Why Use framesvg?

* **True Vector Output:**  Unlike simply embedding a GIF within an SVG, `framesvg` creates a true vector animation.  This means:
  * **Scalability:**  The SVG can be resized to any dimensions without losing quality.
  * **Smaller File Size (Potentially):**  For many GIFs, the resulting SVG will be smaller, especially for graphics with large areas of solid color or simple shapes.  Complex, photographic GIFs may be larger, however.
* **Automatic Playback:** The generated SVGs are designed to play automatically in any environment that supports SVG animations (web browsers, GitHub, many image viewers, etc.).
* **Easy to Use:**  Simple command-line interface and a clean Python API.
* **Customizable:**  Control the frame rate and fine-tune the VTracer conversion process for optimal results.
* **Network Compression:**  SVGs are text-based and highly compressible. Web servers typically use gzip or Brotli compression, *significantly* reducing the actual transfer size of SVG files compared to GIFs (which are already compressed and don't utilize this). This leads to much faster loading times than GIFs. You can see it [here](https://framesvg.romelium.cc/examples.html).

## Examples

There is a dedicated and more better example page [here](https://framesvg.romelium.cc/examples.html)

The following examples demonstrate the conversion of GIFs (left) to SVGs (right) using `framesvg`.

<p align="center">
<img height=200 src="images/Code Coding GIF by EscuelaDevRock Git.gif" alt="Animated GIF showing code being written on Git"/>
<img height=200 src="images/Code Coding GIF by EscuelaDevRock Git.svg" alt="Animated SVG showing code being written on Git"/>
<img height=200 src="images/Code Coding GIF by EscuelaDevRock GitHub.gif" alt="Animated GIF showing code being written on Github"/>
<img height=200 src="images/Code Coding GIF by EscuelaDevRock GitHub.svg" alt="Animated SVG showing code being written on Github"/>
<img height=200 src="images/Code Coding GIF by EscuelaDevRock VSCode.gif" alt="Animated GIF showing code being written on VSCode"/>
<img height=200 src="images/Code Coding GIF by EscuelaDevRock VSCode.svg" alt="Animated SVG showing code being written on VSCode"/>
<img height=200 src="images/Code Coding GIF by EscuelaDevRock Sublime.gif" alt="Animated GIF showing code being written on Sublime Text"/>
<img height=200 src="images/Code Coding GIF by EscuelaDevRock Sublime.svg" alt="Animated SVG showing code being written on Sublime Text"/>
</p>

### More Examples

<p align="center">
<img height=200 src="images/Good_Morning_GIF_by_Hello_All.gif" alt="Animated GIF of Good Morning greeting"/>
<img height=200 src="images/Good_Morning_GIF_by_Hello_All.svg" alt="Animated SVG of Good Morning greeting"/>
<img height=200 src="images/icon_loading_GIF.gif" alt="Animated GIF of a loading icon"/>
<img height=200 src="images/icon_loading_GIF.svg" alt="Animated SVG of a loading icon"/>
<img height=200 src="images/voila hands GIF by brontron.gif" alt="Animated GIF of hands doing a voila gesture"/>
<img height=200 src="images/voila hands GIF by brontron.svg" alt="Animated SVG of hands doing a voila gesture"/>
</p>

### Complex Examples (Transparent Backgrounds)

These examples demonstrate `binary` color mode. All bright colors in `binary` color mode turns transparent. (If they appear dark, it is due to the transparency. They will look correct on light backgrounds)

<p align="center">
<img height=200 src="images/black and white loop GIF by Sculpture.gif" alt="Animated GIF of a black and white loop pattern"/>
<img height=200 src="images/black and white loop GIF by Sculpture.svg" alt="Animated SVG of a black and white loop pattern"/>
<img height=200 src="images/Black And White Loop GIF by Pi-Slices.gif" alt="Animated GIF of another black and white loop pattern"/>
<img height=200 src="images/Black And White Loop GIF by Pi-Slices.svg" alt="Animated SVG of another black and white loop pattern"/>
</p>

## Installation

### Using pipx (Recommended for CLI-only use)

If you primarily intend to use `framesvg` as a command-line tool (and don't need the Python library for development), `pipx` is the recommended installation method.  `pipx` installs Python applications in isolated environments, preventing dependency conflicts with other projects.

```bash
pipx install framesvg
```

To install `pipx` if you don't already have it:

```bash
python3 -m pip install --user pipx
python3 -m pipx ensurepath
```

(You may need to restart your shell after installing `pipx`.)

### Using pip

The easiest way to install `framesvg` is via pip:

```bash
pip install framesvg
```

This installs both the command-line tool and the Python library.

### From Source

1. **Clone the repository:**

    ```bash
    git clone https://github.com/romelium/framesvg
    cd framesvg
    ```

2. **Install:**

    ```bash
    pip install .
    ```

## Usage

### Command-Line Interface

```bash
framesvg input.gif [output.svg] [options]
```

* **`input.gif`:**  (Required) Path to the input GIF file.
* **`output.svg`:** (Optional) Path to save the output SVG file.  If omitted, the output file will have the same name as the input, but with a `.svg` extension.

**Options:**

* **`-f`, `--fps <value>`:**  Sets the frames per second (FPS) for the animation. (Default: Uses the average FPS calculated from the input GIF's frame durations. Falls back to 10 FPS if durations are missing or invalid).
* **`-l`, `--log-level <level>`:**  Sets the logging level.  (Default: INFO).  Choices: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`, `NONE`.  `DEBUG` provides detailed output for troubleshooting.

* **VTracer Options:**  These options control the raster-to-vector conversion process performed by VTracer.  Refer to the [VTracer Documentation](https://www.visioncortex.org/vtracer-docs/) and [Online Demo](https://www.visioncortex.org/vtracer/) for detailed explanations.

  * `-c`, `--colormode <mode>`:  Color mode. (Default: `color`).  Choices: `color`, `binary`.
  * `-i`, `--hierarchical <mode>`:  Hierarchy mode. (Default: `stacked`). Choices: `stacked`, `cutout`.
  * `-m`, `--mode <mode>`:  Conversion mode. (Default: `polygon`). Choices: `spline`, `polygon`, `none`.  `spline` creates smoother curves, but `polygon` often results in smaller files.
  * `-s`, `--filter-speckle <value>`:  Reduces noise and small details. (Default: 4).  *This is a key parameter for controlling file size.* Higher values = smaller files, but less detail.
  * `-p`, `--color-precision <value>`:  Number of significant bits for color quantization. (Default: 8). Lower values = smaller files, but fewer colors.
  * `-d`, `--layer-difference <value>`:  Controls the number of layers. (Default: 16).  Higher values can reduce file size.
  * `--corner-threshold <value>`:  Angle threshold for corner detection. (Default: 60).
  * `--length-threshold <value>`:  Minimum path length. (Default: 4.0).
  * `--max-iterations <value>`:  Maximum number of optimization iterations. (Default: 10).
  * `--splice-threshold <value>`:  Angle threshold for splitting splines. (Default: 45).
  * `--path-precision <value>`:  Number of decimal places for path coordinates. (Default: 8).

**Command-Line Examples:**

```bash
# Basic conversion with default settings
framesvg input.gif

# Specify output file and set FPS to 24
framesvg input.gif output.svg -f 24

# Optimize for smaller file size (less detail)
framesvg input.gif -s 8 -p 3 -d 128

# Enable debug logging
framesvg input.gif -l DEBUG
```

### Python API

```python
from framesvg import gif_to_animated_svg_write, gif_to_animated_svg

# Example 1: Convert and save to a file (using GIF's average FPS)
gif_to_animated_svg_write("input.gif", "output.svg", fps=30)

# Example 2: Get the SVG as a string
animated_svg_string = gif_to_animated_svg("input.gif", fps=12)
print(f"Generated SVG length: {len(animated_svg_string)}")
# ... do something with the string (e.g., save to file, display in a web app)

# Example 3: Customize VTracer options
custom_options = {
    "mode": "spline",
    "filter_speckle": 2,
}
gif_to_animated_svg_write("input.gif", "output_custom.svg", vtracer_options=custom_options)
```

### API Reference

* **`gif_to_animated_svg_write(gif_path, output_svg_path, vtracer_options=None, fps=10.0, image_loader=None, vtracer_instance=None)`:**
  * `gif_path` (str): Path to the input GIF file.
  * `output_svg_path` (str): Path to save the output SVG file.
  * `vtracer_options` (dict, optional):  A dictionary of VTracer options.  If `None`, uses `DEFAULT_VTRACER_OPTIONS`.
  * `fps` (float | None, optional): Frames per second. If `None` (default), calculates the average FPS from the input GIF. Falls back to 10.0 if calculation fails.
  * `image_loader` (ImageLoader, optional): Custom image loader.
  * `vtracer_instance` (VTracer, optional):  Custom VTracer instance.
  * Raises: `FileNotFoundError`, `NotAnimatedGifError`, `NoValidFramesError`, `DimensionError`, `ExtractionError`, `FramesvgError`, `IsADirectoryError`.

* **`gif_to_animated_svg(gif_path, vtracer_options=None, fps=10.0, image_loader=None, vtracer_instance=None)`:**
  * `gif_path` (str): Path to the input GIF file.
  * `vtracer_options` (dict, optional):  A dictionary of VTracer options.  If `None`, uses `DEFAULT_VTRACER_OPTIONS`.
  * `fps` (float | None, optional): Frames per second. If `None` (default), calculates the average FPS from the input GIF. Falls back to 10.0 if calculation fails.
  * `image_loader` (ImageLoader, optional): Custom image loader.
  * `vtracer_instance` (VTracer, optional):  Custom VTracer instance.
  * Returns: The animated SVG as a string.
  * Raises: `FileNotFoundError`, `NotAnimatedGifError`, `NoValidFramesError`, `DimensionError`, `ExtractionError`, `FramesvgError`.

## Tips for Optimizing Large File Size (> 1MB)

* **[Online Demo](https://www.visioncortex.org/vtracer/):**  Use this to visualize tweaking values. Experiment to find the best balance between size and quality.
* **`filter-speckle`:**  *This is the most impactful setting for reducing file size, especially on complex images.*  Increasing it removes small details.
* **`--mode polygon`:**  Use the default polygon mode unless smooth curves (spline mode) are absolutely necessary.  Polygon mode can significantly reduce file size by a factor of 5 or more.
* **`layer-difference`:**  Increase this to reduce the number of layers.
* **`color-precision`:** Reduce the number of colors by lowering this value.

## Dev

### Install Hatch (Recommended)

follow [this](https://hatch.pypa.io/latest/install)

or just

```bash
pip install hatch
```

### Format and lint

```bash
hatch fmt
```

### Testing

```bash
hatch test
```

### Other Hatch Commands

```bash
hatch -h
```

## Dev Web app

### Install Vercel

Install the Vercel CLI globally:

```bash
npm install -g vercel
```

### Running Locally

```bash
vercel dev
```

**Setup (First Time Only):**  When running for the *first* time, you'll be prompted to configure settings.  Ensure you set the "In which directory is your code located?" option to `./web`.

**Note:**  The first conversion may take a significant amount of time. This is because the serverless functions need to be built. Subsequent conversions will be faster.

### Deploy to Vercel

```bash
vercel deploy
```

## Contributing

Contributions are welcome!  Please submit pull requests or open issues on the [GitHub repository](https://github.com/romelium/framesvg).
