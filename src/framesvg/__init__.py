from __future__ import annotations

import argparse
import io
import logging
import os
import re
import sys
from typing import Literal, Protocol, TypedDict

from PIL import Image


class VTracerOptions(TypedDict, total=False):
    colormode: Literal["color", "binary"] | None
    hierarchical: Literal["stacked", "cutout"] | None
    mode: Literal["spline", "polygon", "none"] | None
    filter_speckle: int | None
    color_precision: int | None
    layer_difference: int | None
    corner_threshold: int | None
    length_threshold: float | None
    max_iterations: int | None
    splice_threshold: int | None
    path_precision: int | None


FALLBACK_FPS = 10.0
"""Fallback FPS if GIF duration cannot be determined."""

DEFAULT_VTRACER_OPTIONS: VTracerOptions = {
    "colormode": "color",
    "hierarchical": "stacked",
    "mode": "polygon",
    "filter_speckle": 4,
    "color_precision": 8,
    "layer_difference": 16,
    "corner_threshold": 60,
    "length_threshold": 4.0,
    "max_iterations": 10,
    "splice_threshold": 45,
    "path_precision": 8,
}


class FramesvgError(Exception):
    """Base class for exceptions."""


class NotAnimatedGifError(FramesvgError):
    """Input GIF is not animated."""

    def __init__(self, gif_path: str):
        super().__init__(f"{gif_path} is not an animated GIF.")


class NoValidFramesError(FramesvgError):
    """No valid SVG frames generated."""

    def __init__(self):
        super().__init__("No valid SVG frames were generated.")


class DimensionError(FramesvgError):
    """SVG dimensions could not be determined."""

    def __init__(self):
        super().__init__("Could not determine SVG dimensions.")


class ExtractionError(FramesvgError):
    """SVG content could not be extracted."""

    def __init__(self):
        super().__init__("Could not extract SVG content.")


class FrameOutOfRangeError(FramesvgError):
    """Frame out of Range."""

    def __init__(self, frame_number, max_frames):
        super().__init__(f"Frame number {frame_number} is out of range.  Must be between 0 and {max_frames -1}")


class ImageWrapper(Protocol):
    is_animated: bool
    n_frames: int
    format: str | None
    info: dict

    def seek(self, frame: int) -> None: ...
    def save(self, fp, img_format) -> None: ...
    def close(self) -> None: ...


class ImageLoader(Protocol):
    def open(self, filepath: str) -> ImageWrapper: ...


class VTracer(Protocol):
    def convert_raw_image_to_svg(self, image_bytes: bytes, img_format: str, options: VTracerOptions) -> str: ...


class PILImageLoader:
    def open(self, filepath: str) -> ImageWrapper:
        return Image.open(filepath)


class DefaultVTracer:
    def convert_raw_image_to_svg(self, image_bytes: bytes, img_format: str, options: VTracerOptions) -> str:
        import vtracer

        return vtracer.convert_raw_image_to_svg(image_bytes, img_format=img_format, **options)


_DEFAULT_IMAGE_LOADER = PILImageLoader()
_DEFAULT_VTRACER = DefaultVTracer()


def load_image_wrapper(filepath: str, image_loader: ImageLoader) -> ImageWrapper:
    """Loads an image and returns an ImageWrapper."""
    try:
        return image_loader.open(filepath)
    except FileNotFoundError:
        logging.exception("File not found: %s", filepath)
        raise
    except Exception:
        logging.exception("Error loading image: %s", filepath)
        raise


def is_animated_gif(img: ImageWrapper, filepath: str) -> None:
    """Checks if the image is an animated GIF."""
    if not img.is_animated:
        raise NotAnimatedGifError(filepath)


def _calculate_gif_average_fps(img: ImageWrapper) -> float | None:
    """Calculates the average FPS from GIF frame durations."""
    if not img.is_animated or img.n_frames <= 1:
        return None  # Not animated or single frame

    total_duration_ms = 0
    valid_frames_with_duration = 0
    try:
        for i in range(img.n_frames):
            img.seek(i)
            # PIL uses 100ms default if duration is missing or 0.
            # Use this default directly in the sum.
            duration = img.info.get("duration", 100)
            # Ensure duration is at least 1ms if it was 0, consistent with some viewers/browsers
            # treating 0 as a very small delay rather than 100ms.
            # However, for FPS calculation, using the 100ms default seems more robust.
            total_duration_ms += duration if duration > 0 else 100
            valid_frames_with_duration += 1
    except EOFError:
        logging.warning("EOFError encountered while reading GIF durations. FPS calculation might be inaccurate.")

    # Avoid division by zero if somehow total_duration_ms is 0 after processing
    if total_duration_ms <= 0:
        return None

    return valid_frames_with_duration / (total_duration_ms / 1000.0)


def extract_svg_dimensions_from_content(svg_content: str) -> dict[str, int] | None:
    """Extracts width and height from SVG."""
    dims: dict[str, int] = {"width": 0, "height": 0}
    view_box_pattern = re.compile(r'viewBox=["\'](\d+)\s+(\d+)\s+(\d+)\s+(\d+)["\']')
    width_pattern = re.compile(r"<svg[^>]*width=[\"'](\d+)")
    height_pattern = re.compile(r"<svg[^>]*height=[\"'](\d+)")

    match = view_box_pattern.search(svg_content)
    if match:
        dims["width"], dims["height"] = int(match.group(3)), int(match.group(4))
    else:
        match_width = width_pattern.search(svg_content)
        if match_width:
            dims["width"] = int(match_width.group(1))
        match_height = height_pattern.search(svg_content)
        if match_height:
            dims["height"] = int(match_height.group(1))

    if dims["width"] <= 0 or dims["height"] <= 0:
        return None
    return dims


def extract_inner_svg_content_from_full_svg(full_svg_content: str) -> str:
    """Extracts content within <svg> tags."""
    start_pos = full_svg_content.find("<svg")
    if start_pos == -1:
        return ""

    start_pos = full_svg_content.find(">", start_pos) + 1
    end_pos = full_svg_content.rfind("</svg>")
    if start_pos == -1 or end_pos == -1:
        return ""

    return full_svg_content[start_pos:end_pos]


def process_gif_frame(
    img: ImageWrapper,
    frame_number: int,
    vtracer_instance: VTracer,
    vtracer_options: VTracerOptions,
) -> tuple[str, dict[str, int] | None]:
    """Processes single GIF frame, converting to SVG."""
    if not 0 <= frame_number < img.n_frames:
        raise FrameOutOfRangeError(frame_number, img.n_frames)

    img.seek(frame_number)
    with io.BytesIO() as img_byte_arr:
        img_byte_arr.name = "temp.gif"
        img.save(img_byte_arr, img_format="GIF")
        img_bytes = img_byte_arr.getvalue()

    svg_content = vtracer_instance.convert_raw_image_to_svg(img_bytes, img_format="GIF", options=vtracer_options)
    dims = extract_svg_dimensions_from_content(svg_content)
    inner_svg = extract_inner_svg_content_from_full_svg(svg_content) if dims else ""

    return inner_svg, dims


def process_gif_frames(
    img: ImageWrapper,
    vtracer_instance: VTracer,
    vtracer_options: VTracerOptions,
) -> tuple[list[str], dict[str, int]]:
    """Processes all GIF frames."""
    frames: list[str] = []
    max_dims = {"width": 0, "height": 0}

    for i in range(img.n_frames):
        inner_svg_content, dims = process_gif_frame(img, i, vtracer_instance, vtracer_options)

        if dims:
            max_dims["width"] = max(max_dims["width"], dims["width"])
            max_dims["height"] = max(max_dims["height"], dims["height"])
            if inner_svg_content:
                frames.append(inner_svg_content)

    if not frames:
        raise NoValidFramesError

    return frames, max_dims


def create_animated_svg_string(frames: list[str], max_dims: dict[str, int], fps: float) -> str:
    """Generates animated SVG string."""
    if not frames:
        msg = "No frames to generate SVG."
        raise ValueError(msg)
    if fps <= 0:
        logging.warning("FPS is non-positive (%.2f), defaulting to fallback FPS %.1f", fps, FALLBACK_FPS)
        fps = FALLBACK_FPS
    frame_duration = 1.0 / fps
    total_duration = frame_duration * len(frames)

    svg_str = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{max_dims["width"]}" height="{max_dims["height"]}" '
        f'viewBox="0 0 {max_dims["width"]} {max_dims["height"]}">'
        f"<title>Generated Animation</title>\n"
        f"<desc>{len(frames)} frames at {fps:.6f} FPS</desc>\n"
        '<g id="animationGroup">\n'
    )

    for i, frame_content in enumerate(frames):
        start_fraction = i / len(frames)
        end_fraction = (i + 1) / len(frames)
        svg_str += (
            f'<g id="frame{i}" opacity="0">\n'
            f"{frame_content}\n"
            f'<animate attributeName="opacity" dur="{total_duration:.6f}s" '
            f'values="0;1;0" keyTimes="0;{start_fraction:.6f};{end_fraction:.6f}" '
            'calcMode="discrete" repeatCount="indefinite" begin="0s"/>\n'
            "</g>\n"
        )

    svg_str += "</g>\n</svg>\n"
    return svg_str


def save_svg_to_file(svg_string: str, output_path: str) -> None:
    """Writes SVG string to file."""
    if os.path.isdir(output_path):
        msg = f"'{output_path}' is a directory, not a file."
        logging.error(msg)
        raise IsADirectoryError(msg)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(svg_string)
    except Exception:
        logging.exception("Error writing SVG to file: %s", output_path)
        raise


def gif_to_animated_svg(
    gif_path: str,
    vtracer_options: VTracerOptions | None = None,
    fps: float | None = None,
    image_loader: ImageLoader | None = None,
    vtracer_instance: VTracer | None = None,
) -> str:
    """Main function to convert GIF to animated SVG."""
    image_loader = image_loader or _DEFAULT_IMAGE_LOADER
    vtracer_instance = vtracer_instance or _DEFAULT_VTRACER

    options = DEFAULT_VTRACER_OPTIONS.copy()
    if vtracer_options:
        options.update(vtracer_options)

    img = load_image_wrapper(gif_path, image_loader)
    try:
        is_animated_gif(img, gif_path)

        effective_fps = fps
        if effective_fps is None:
            calculated_fps = _calculate_gif_average_fps(img)
            effective_fps = calculated_fps if calculated_fps is not None else FALLBACK_FPS
            logging.info("Using calculated average FPS: %.2f (Fallback: %.1f)", effective_fps, FALLBACK_FPS)
        elif effective_fps <= 0:
            logging.warning("Provided FPS is non-positive (%.2f), using fallback FPS %.1f", effective_fps, FALLBACK_FPS)
            effective_fps = FALLBACK_FPS

        frames, max_dims = process_gif_frames(img, vtracer_instance, options)
        return create_animated_svg_string(frames, max_dims, effective_fps)
    finally:
        img.close()


def gif_to_animated_svg_write(
    gif_path: str,
    output_svg_path: str,
    vtracer_options: VTracerOptions | None = None,
    fps: float | None = None,
    image_loader: ImageLoader | None = None,
    vtracer_instance: VTracer | None = None,
) -> None:
    """Converts and writes to file."""
    svg = gif_to_animated_svg(gif_path, vtracer_options, fps, image_loader, vtracer_instance)
    save_svg_to_file(svg, output_svg_path)


def validate_positive_int(value: str) -> int:
    """Validates positive integer input."""
    try:
        int_value = int(value)
        if int_value <= 0:
            msg = f"{value} is not a positive integer."
            raise argparse.ArgumentTypeError(msg)
    except ValueError as e:
        msg = f"{value} is not a valid integer."
        raise argparse.ArgumentTypeError(msg) from e
    else:
        return int_value


def validate_positive_float(value: str) -> float:
    """Validates positive float input."""
    try:
        float_value = float(value)
        if float_value <= 0:
            msg = f"{value} is not a positive float."
            raise argparse.ArgumentTypeError(msg)
    except ValueError as e:
        msg = f"{value} is not a valid float."
        raise argparse.ArgumentTypeError(msg) from e
    else:
        return float_value


def parse_cli_arguments(args: list[str]) -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Convert an animated GIF to an animated SVG.")
    parser.add_argument("gif_path", help="Path to the input GIF file.")
    parser.add_argument(
        "output_svg_path",
        nargs="?",
        help="Output. Defaults to input filename with .svg.",
    )
    parser.add_argument(
        "-f",
        "--fps",
        type=validate_positive_float,
        default=None,  # Default is now None, handled later
        help=f"Frames per second. (Default: Use GIF average FPS, fallback: {FALLBACK_FPS}).",
    )
    parser.add_argument(
        "-l",
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NONE"],
        default="INFO",
        help="Set the logging level (default: INFO).",
    )

    # VTracer options
    parser.add_argument("-c", "--colormode", choices=["color", "binary"], help="Color mode.")
    parser.add_argument("-i", "--hierarchical", choices=["stacked", "cutout"], help="Hierarchical mode.")
    parser.add_argument("-m", "--mode", choices=["spline", "polygon", "none"], help="Mode.")
    parser.add_argument("-s", "--filter-speckle", type=validate_positive_int, help="Filter speckle.")
    parser.add_argument("-p", "--color-precision", type=validate_positive_int, help="Color precision.")
    parser.add_argument("-d", "--layer-difference", type=validate_positive_int, help="Layer difference.")
    parser.add_argument("--corner-threshold", type=validate_positive_int, help="Corner threshold.")
    parser.add_argument("--length-threshold", type=validate_positive_float, help="Length threshold.")
    parser.add_argument("--max-iterations", type=validate_positive_int, help="Max iterations.")
    parser.add_argument("--splice-threshold", type=validate_positive_int, help="Splice threshold.")
    parser.add_argument("--path-precision", type=validate_positive_int, help="Path precision.")

    return parser.parse_args(args)


def main() -> None:
    """Main entry point."""
    try:
        args = parse_cli_arguments(sys.argv[1:])

        if args.log_level != "NONE":
            logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")

        output_path = args.output_svg_path
        if output_path is None:
            base, _ = os.path.splitext(args.gif_path)
            output_path = base + ".svg"

        vtracer_options = {
            k: v
            for k, v in vars(args).items()
            if k
            in [
                "colormode",
                "hierarchical",
                "mode",
                "filter_speckle",
                "color_precision",
                "layer_difference",
                "corner_threshold",
                "length_threshold",
                "max_iterations",
                "splice_threshold",
                "path_precision",
            ]
            and v is not None
        }

        gif_to_animated_svg_write(args.gif_path, output_path, vtracer_options=vtracer_options, fps=args.fps)
        logging.info("Converted %s to %s", args.gif_path, output_path)
    except SystemExit as e:
        sys.exit(e.code)
    except FramesvgError:
        # Specific, expected errors are logged within the functions
        # Log general message here for unexpected FramesvgError subclasses
        logging.exception("FrameSVG processing failed. Check previous logs for details.")
        sys.exit(1)
    except Exception:
        logging.exception("An unexpected error occurred.")
        sys.exit(1)
