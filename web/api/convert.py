from http.server import BaseHTTPRequestHandler
import json
import base64
import logging
import os
from urllib.parse import parse_qs

from framesvg import (
    gif_to_animated_svg,
    DEFAULT_VTRACER_OPTIONS,
    VTracerOptions,
    FramesvgError,
    NotAnimatedGifError,
    NoValidFramesError,
    DimensionError,
    ExtractionError,
)


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        """Handles the GIF to SVG conversion request."""
        logging.basicConfig(level=logging.INFO)

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self.send_error(400, "No content provided")
                return

            content_type = self.headers.get("Content-Type", "")
            if "application/json" not in content_type:
                self.send_error(400, "Invalid content type")
                return

            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode("utf-8"))
            except json.JSONDecodeError:
                self.send_error(400, "Invalid JSON data")
                return

            if "file" not in data:
                self.send_error(400, "No file provided")
                return

            try:
                gif_data = base64.b64decode(data["file"].split(",")[1])
            except Exception:
                self.send_error(400, "Invalid base64 data")
                return

            params = data.get("params", {})
            vtracer_options: VTracerOptions = DEFAULT_VTRACER_OPTIONS.copy()

            if params:
                for key, value in params.items():
                    if key in DEFAULT_VTRACER_OPTIONS and value is not None:
                        if isinstance(DEFAULT_VTRACER_OPTIONS[key], int):
                            try:
                                vtracer_options[key] = int(value)  # type: ignore
                            except ValueError:
                                self.send_error(400, f"Invalid integer value for {key}")
                                return
                        elif isinstance(DEFAULT_VTRACER_OPTIONS[key], float):
                            try:
                                vtracer_options[key] = float(value)  # type: ignore
                            except ValueError:
                                self.send_error(400, f"Invalid float value for {key}")
                                return
                        elif isinstance(DEFAULT_VTRACER_OPTIONS[key], str):
                            if key == "colormode" and value not in ["color", "binary"]:
                                self.send_error(400, "Invalid value for colormode")
                                return
                            if key == "hierarchical" and value not in ["stacked", "cutout"]:
                                self.send_error(400, "Invalid value for hierarchical")
                                return
                            if key == "mode" and value not in ["spline", "polygon", "none"]:
                                self.send_error(400, "Invalid value for mode")
                                return
                            vtracer_options[key] = value  # type: ignore

            fps = float(params.get("fps", DEFAULT_VTRACER_OPTIONS))
            if "fps" in params:
                try:
                    fps = float(params["fps"])
                except ValueError:
                    self.send_error(400, "Invalid fps value")
                    return

            temp_gif_path = "/tmp/input.gif"
            with open(temp_gif_path, "wb") as f:
                f.write(gif_data)

            try:
                svg_content = gif_to_animated_svg(temp_gif_path, vtracer_options=vtracer_options, fps=fps)
            except NotAnimatedGifError:
                self.send_error(400, "The provided GIF is not animated.")
                return
            except NoValidFramesError:
                self.send_error(500, "No valid frames were generated.")
                return
            except (DimensionError, ExtractionError):
                self.send_error(500, "Error processing the GIF.")
                return
            except FramesvgError as e:
                self.send_error(500, f"FrameSVG Error: {e}")
                return
            except Exception as e:
                logging.exception("Unexpected error: %s", e)
                self.send_error(500, "An unexpected error occurred.")
                return
            finally:
                try:
                    os.remove(temp_gif_path)
                except OSError:
                    pass

            svg_size_bytes = len(svg_content.encode("utf-8"))
            if svg_size_bytes > 4.5 * 1024 * 1024:
                svg_size_mb = svg_size_bytes / (1024 * 1024)
                msg = f"Generated SVG is too large (exceeds 4.5MB). Size: {svg_size_mb:.2f}MB."
                self.send_error(413, msg, msg + " Use CLI or Python library for large input and output")
                return

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"svg": svg_content}).encode("utf-8"))

        except Exception as e:
            logging.exception("Unexpected error in handler: %s", e)
            self.send_error(500, "An unexpected error occurred.")
            return
