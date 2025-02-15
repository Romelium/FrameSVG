from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
from PIL import Image, ImageCms, ImageDraw

from framesvg import (
    DEFAULT_FPS,
    DEFAULT_VTRACER_OPTIONS,
    FrameOutOfRangeError,
    NotAnimatedGifError,
    NoValidFramesError,
    VTracerOptions,
    create_animated_svg_string,
    extract_inner_svg_content_from_full_svg,
    extract_svg_dimensions_from_content,
    gif_to_animated_svg,
    gif_to_animated_svg_write,
    is_animated_gif,
    load_image_wrapper,
    parse_cli_arguments,
    process_gif_frame,
    process_gif_frames,
    save_svg_to_file,
)

if TYPE_CHECKING:
    from pathlib import Path


def create_mock_image(
    *,
    is_animated: bool = True,
    n_frames: int = 2,
    width: int = 100,
    height: int = 100,
    img_format: str = "GIF",
    close_raises: bool = False,
):
    """Creates a mock Image object."""
    mock_img = Mock()
    mock_img.is_animated = is_animated
    mock_img.n_frames = n_frames
    mock_img.width = width
    mock_img.height = height
    mock_img.format = img_format
    mock_img.info = {"duration": 100}

    if close_raises:
        mock_img.close.side_effect = Exception("Mock close error")
    else:
        mock_img.close = Mock()  # Ensure close is callable
    return mock_img


def create_mock_vtracer(
    return_svg: str | list[str] = '<svg width="100" height="100"><path/></svg>',
    raise_error: bool = False,  # noqa: FBT001 FBT002
):
    """Creates a mock VTracer."""
    mock_vtracer = Mock()
    if raise_error:
        mock_vtracer.convert_raw_image_to_svg.side_effect = Exception("VTracer error")
    elif isinstance(return_svg, list):
        mock_vtracer.convert_raw_image_to_svg.side_effect = return_svg
    else:
        mock_vtracer.convert_raw_image_to_svg.return_value = return_svg
    return mock_vtracer


def create_temp_gif(
    tmp_path: Path,
    *,
    is_animated: bool = True,
    num_frames: int = 2,
    widths: tuple[int, ...] = (100,),
    heights: tuple[int, ...] = (100,),
    durations: tuple[int, ...] = (100,),
    corrupt: bool = False,
    use_palette: bool = False,
    add_transparency: bool = False,
    color_profile: bool = False,
) -> str:
    """Creates a temp GIF."""
    gif_path = tmp_path / "test.gif"
    images = []

    for i in range(max(1, num_frames if is_animated else 1)):
        width = widths[i % len(widths)]
        height = heights[i % len(heights)]
        if use_palette:
            img = Image.new("P", (width, height), color=0)
            img.putpalette(
                [
                    0,
                    0,
                    0,  # Black
                    255,
                    255,
                    255,  # White
                    255,
                    0,
                    0,  # Red
                    0,
                    255,
                    0,  # Green
                    0,
                    0,
                    255,  # Blue
                ]
                * 51
            )  # Repeat the palette to fill 256 entries

        else:
            img = Image.new(
                "RGB",
                (width, height),
                color=(i * 50 % 256, i * 100 % 256, i * 150 % 256),
            )

        if add_transparency and i % 2 == 0:  # Make every other frame have some transparency
            alpha = Image.new("L", (width, height), color=128)  # semi-transparent
            if use_palette:
                img.putalpha(alpha)  # P mode needs special handling for alpha
            else:
                img = img.convert("RGBA")
                img.putalpha(alpha)

        if color_profile:
            # Create a simple sRGB color profile
            profile = ImageCms.createProfile("sRGB")
            img.info["icc_profile"] = ImageCms.ImageCmsProfile(profile).tobytes()

        draw = ImageDraw.Draw(img)
        draw.text((10, 10), text=f"frame {i}", fill=(0, 0, 0))
        images.append(img)

    if is_animated and num_frames > 1:
        images[0].save(
            gif_path,
            "GIF",
            save_all=True,
            append_images=images[1:],
            duration=[durations[i % len(durations)] for i in range(num_frames)],
            loop=0,
            transparency=0 if add_transparency else None,  # Specify transparency color index
        )
    else:
        images[0].save(gif_path, "GIF")

    if corrupt:
        with open(gif_path, "wb") as f:
            f.write(b"CORRUPT")

    return str(gif_path)


@pytest.fixture
def sample_svg_content():
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="200" viewBox="0 0 100 200">\n'
        '<path d="M10 10 L90 90"/>\n'
        "</svg>"
    )


@pytest.fixture
def sample_frames():
    return [
        '<path d="M10 10 L90 90"/>',
        '<path d="M20 20 L80 80"/>',
        '<path d="M30 30 L70 70"/>',
    ]


@pytest.fixture(params=[True, False])
def animated_gif_state(request):
    return request.param


@pytest.fixture(params=[1, 2, 5, 10])
def frame_number_count(request):
    return request.param


@pytest.fixture(params=[(100, 100), (200, 150), (150, 200), (300, 300)])
def image_dimensions(request):
    return request.param


@pytest.fixture(params=[(50,), (100,), (200,)])
def duration_values(request):
    return request.param


@pytest.fixture
def mock_image_instance(animated_gif_state, frame_number_count, image_dimensions):
    width, height = image_dimensions
    return create_mock_image(is_animated=animated_gif_state, n_frames=frame_number_count, width=width, height=height)


@pytest.fixture
def mock_vtracer_instance_for_tests():
    return create_mock_vtracer()


@pytest.fixture
def mock_image_loader_instance():
    return Mock()


def test_load_image_wrapper_success(tmp_path):
    gif_path = create_temp_gif(tmp_path)

    # Create mock objects locally
    mock_loader = Mock()
    mock_image = create_mock_image(is_animated=True, img_format="GIF")
    mock_loader.open.return_value = mock_image

    img_wrapper = load_image_wrapper(gif_path, mock_loader)

    assert img_wrapper.is_animated
    assert img_wrapper.format == "GIF"  # check the format too.
    mock_loader.open.assert_called_once_with(gif_path)


def test_load_image_wrapper_file_not_found():
    mock_loader = Mock()
    mock_loader.open.side_effect = FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_image_wrapper("nonexistent.gif", mock_loader)


def test_load_image_wrapper_general_exception(tmp_path):
    mock_loader = Mock()
    gif_path = create_temp_gif(tmp_path)
    mock_loader.open.side_effect = Exception("Some error")
    with pytest.raises(Exception, match="Some error"):  # Keep generic, as PIL can raise many
        load_image_wrapper(gif_path, mock_loader)


def test_load_image_wrapper_corrupt_file(tmp_path):
    mock_loader = Mock()
    gif_path = create_temp_gif(tmp_path, corrupt=True)
    mock_loader.open.side_effect = Image.UnidentifiedImageError
    with pytest.raises(Image.UnidentifiedImageError):
        load_image_wrapper(gif_path, mock_loader)


def test_load_image_wrapper_unsupported_format(tmp_path):
    mock_loader = Mock()
    # Create a text file with .gif extension
    path = tmp_path / "fake.gif"
    path.write_text("Not a GIF")
    mock_loader.open.side_effect = Image.UnidentifiedImageError
    with pytest.raises(Image.UnidentifiedImageError):
        load_image_wrapper(str(path), mock_loader)


@pytest.mark.parametrize("system_error", [PermissionError, OSError])
def test_load_image_wrapper_system_errors(system_error):
    mock_loader = Mock()
    mock_loader.open.side_effect = system_error
    with pytest.raises(system_error):
        load_image_wrapper("test.gif", mock_loader)


@pytest.mark.parametrize(
    ("svg_str_input", "expected_output"),
    [
        ('<svg width="100" height="200">', {"width": 100, "height": 200}),
        ('<svg viewBox="0 0 300 400">', {"width": 300, "height": 400}),
        (
            '<svg width="100" viewBox="0 0 200 300">',
            {"width": 200, "height": 300},
        ),
        (
            '<svg height="100" viewBox="0 0 200 300">',
            {"width": 200, "height": 300},
        ),
        ("<svg>", None),
        ('<svg width="0" height="100">', None),
        ('<svg width="100" height="0">', None),
        ('<svg width="-100" height="100">', None),
        ('<svg width="invalid" height="100">', None),
    ],
)
def test_extract_svg_dimensions_from_content_variations(svg_str_input, expected_output):
    if expected_output is None:
        assert extract_svg_dimensions_from_content(svg_str_input) is None
    else:
        assert extract_svg_dimensions_from_content(svg_str_input) == expected_output


@pytest.mark.parametrize(
    ("full_svg_input", "expected_inner_content"),
    [
        ("<svg>content</svg>", "content"),
        ("<svg></svg>", ""),
        ("<svg>nested<g>content</g></svg>", "nested<g>content</g>"),
        ("not svg content", ""),
        ("<svg>unclosed", ""),
        ("", ""),
    ],
)
def test_extract_inner_svg_content_from_full_svg_variations(full_svg_input, expected_inner_content):
    assert extract_inner_svg_content_from_full_svg(full_svg_input) == expected_inner_content


def test_process_gif_frame_success(mock_image_instance, mock_vtracer_instance_for_tests):
    inner_svg, dims = process_gif_frame(
        mock_image_instance, 0, mock_vtracer_instance_for_tests, DEFAULT_VTRACER_OPTIONS
    )
    assert isinstance(inner_svg, str)
    assert isinstance(dims, dict)
    assert "width" in dims
    assert "height" in dims


def test_process_gif_frame_vtracer_error(mock_image_instance, mock_vtracer_instance_for_tests):
    mock_vtracer_instance_for_tests.convert_raw_image_to_svg.side_effect = Exception("Simulated VTracer error")
    with pytest.raises(Exception, match="Simulated VTracer error"):
        process_gif_frame(mock_image_instance, 0, mock_vtracer_instance_for_tests, DEFAULT_VTRACER_OPTIONS)


@pytest.mark.parametrize("selected_frame", [-1, 0, 1, 2])
def test_process_gif_frame_number_variations(mock_image_instance, mock_vtracer_instance_for_tests, selected_frame):
    if 0 <= selected_frame < mock_image_instance.n_frames:
        inner_svg, dims = process_gif_frame(
            mock_image_instance,
            selected_frame,
            mock_vtracer_instance_for_tests,
            DEFAULT_VTRACER_OPTIONS,
        )
        assert isinstance(inner_svg, str)
        assert isinstance(dims, dict)
    else:
        with pytest.raises(FrameOutOfRangeError):
            process_gif_frame(
                mock_image_instance,
                selected_frame,
                mock_vtracer_instance_for_tests,
                DEFAULT_VTRACER_OPTIONS,
            )


def test_check_if_animated_gif_positive(mock_image_instance):
    mock_image_instance.is_animated = True  # Ensure it's set to True
    # Test should pass without raising an exception
    is_animated_gif(mock_image_instance, "test.gif")


def test_check_if_animated_gif_negative(mock_image_instance):
    mock_image_instance.is_animated = False  # Set to False
    with pytest.raises(NotAnimatedGifError, match="test.gif is not an animated GIF."):
        is_animated_gif(mock_image_instance, "test.gif")


def test_create_animated_svg_string_comprehensive_tests(sample_frames):
    fps_list = [1.0, 10.0, 24.0, 60.0]
    dimension_list = [
        {"width": 100, "height": 100},
        {"width": 200, "height": 150},
        {"width": 1920, "height": 1080},
    ]

    for test_fps in fps_list:
        for test_dims in dimension_list:
            svg_string_result = create_animated_svg_string(sample_frames, test_dims, test_fps)

            assert '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' in svg_string_result

            assert f'width="{test_dims["width"]}"' in svg_string_result
            assert f'height="{test_dims["height"]}"' in svg_string_result
            assert f'viewBox="0 0 {test_dims["width"]} {test_dims["height"]}"' in svg_string_result

            expected_duration = len(sample_frames) / test_fps
            assert f'dur="{expected_duration:.6f}s"' in svg_string_result

            for frame_content in sample_frames:
                assert frame_content in svg_string_result


def test_create_animated_svg_string_edge_cases_tests():
    test_dims = {"width": 100, "height": 100}

    with pytest.raises(ValueError, match="No frames to generate SVG."):
        create_animated_svg_string([], test_dims, DEFAULT_FPS)

    single_frame_svg = create_animated_svg_string(["<path/>"], test_dims, DEFAULT_FPS)
    assert "<path/>" in single_frame_svg
    assert 'repeatCount="indefinite"' in single_frame_svg

    high_fps_svg = create_animated_svg_string(["<path/>"], test_dims, 1000.0)
    assert 'dur="0.001000s"' in high_fps_svg

    low_fps_svg = create_animated_svg_string(["<path/>"], test_dims, 0.1)
    assert 'dur="10.000000s"' in low_fps_svg


def test_process_gif_frames_integration_tests(mock_image_instance, mock_vtracer_instance_for_tests):
    frames, max_dims = process_gif_frames(mock_image_instance, mock_vtracer_instance_for_tests, DEFAULT_VTRACER_OPTIONS)
    assert isinstance(frames, list)
    assert isinstance(max_dims, dict)
    assert len(frames) > 0
    assert all(isinstance(frame, str) for frame in frames)
    assert max_dims["width"] > 0
    assert max_dims["height"] > 0

    mock_image_instance.n_frames = 0
    with pytest.raises(NoValidFramesError):
        process_gif_frames(mock_image_instance, mock_vtracer_instance_for_tests, DEFAULT_VTRACER_OPTIONS)

    mock_image_instance.n_frames = 2  # Restore

    mock_vtracer = create_mock_vtracer(return_svg="<svg></svg>")
    with pytest.raises(NoValidFramesError):
        process_gif_frames(mock_image_instance, mock_vtracer, DEFAULT_VTRACER_OPTIONS)

    mock_vtracer = create_mock_vtracer(return_svg=['<svg width="10" height="10"><path/></svg>', "<svg></svg>"])
    frames, max_dims = process_gif_frames(mock_image_instance, mock_vtracer, DEFAULT_VTRACER_OPTIONS)
    assert len(frames) == 1


def test_gif_to_animated_svg_comprehensive(tmp_path, mock_image_loader_instance, mock_vtracer_instance_for_tests):
    frame_counts = [2, 5, 10]
    dimension_sets = [(100, 100), (200, 150), (150, 200)]
    fps_values = [10.0, 24.0, 60.0]
    vtracer_options_set: VTracerOptions = {
        "colormode": "color",
        "filter_speckle": 4,
        "corner_threshold": 60,
    }

    for num_frames in frame_counts:
        for width, height in dimension_sets:
            for fps in fps_values:
                gif_path = create_temp_gif(tmp_path, num_frames=num_frames, widths=(width,), heights=(height,))

                # Set up the mock_image_loader correctly.  Important!
                mock_image = create_mock_image(is_animated=True, n_frames=num_frames, width=width, height=height)
                mock_image_loader_instance.open.return_value = mock_image

                svg_string = gif_to_animated_svg(
                    gif_path,
                    vtracer_options=vtracer_options_set,
                    fps=fps,
                    image_loader=mock_image_loader_instance,
                    vtracer_instance=mock_vtracer_instance_for_tests,
                )

                assert svg_string.startswith("<?xml")
                assert "<svg" in svg_string
                assert "</svg>" in svg_string
                assert f"<desc>{num_frames} frames at {fps:.6f} FPS</desc>" in svg_string
                assert 'repeatCount="indefinite"' in svg_string
                mock_image_loader_instance.open.assert_called_with(gif_path)


def test_gif_to_animated_svg_error_handling(tmp_path):
    gif_path = create_temp_gif(tmp_path, is_animated=False)
    with pytest.raises(NotAnimatedGifError):
        gif_to_animated_svg(gif_path)

    gif_path = create_temp_gif(tmp_path, corrupt=True)
    with pytest.raises(Image.UnidentifiedImageError):
        gif_to_animated_svg(gif_path)

    with pytest.raises(FileNotFoundError):
        gif_to_animated_svg("nonexistent.gif")


@pytest.mark.parametrize(
    ("cli_args", "expected_parsed_args"),
    [
        (
            ["input.gif"],
            {
                "gif_path": "input.gif",
                "output_svg_path": None,
                "fps": DEFAULT_FPS,
                "log_level": "INFO",
            },
        ),
        (
            ["input.gif", "output.svg"],
            {
                "gif_path": "input.gif",
                "output_svg_path": "output.svg",
                "fps": DEFAULT_FPS,
                "log_level": "INFO",
            },
        ),
        (
            ["input.gif", "--fps", "30"],
            {
                "gif_path": "input.gif",
                "output_svg_path": None,
                "fps": 30.0,
                "log_level": "INFO",
            },
        ),
        (
            ["input.gif", "--log-level", "DEBUG"],
            {
                "gif_path": "input.gif",
                "output_svg_path": None,
                "fps": DEFAULT_FPS,
                "log_level": "DEBUG",
            },
        ),
        (
            ["input.gif", "--colormode", "binary", "--filter-speckle", "2"],
            {
                "gif_path": "input.gif",
                "output_svg_path": None,
                "fps": DEFAULT_FPS,
                "log_level": "INFO",
                "colormode": "binary",
                "filter_speckle": 2,
            },
        ),
        (
            [
                "input.gif",
                "output.svg",
                "--fps",
                "60",
                "--log-level",
                "ERROR",
                "--mode",
                "polygon",
            ],
            {
                "gif_path": "input.gif",
                "output_svg_path": "output.svg",
                "fps": 60.0,
                "log_level": "ERROR",
                "mode": "polygon",
            },
        ),
        (
            ["input.gif", "-f", "24"],
            {
                "gif_path": "input.gif",
                "output_svg_path": None,
                "fps": 24.0,
                "log_level": "INFO",
            },
        ),
        (
            [
                "input.gif",
                "--colormode",
                "color",
                "--hierarchical",
                "stacked",
                "--mode",
                "spline",
                "--filter-speckle",
                "4",
                "--color-precision",
                "6",
                "--layer-difference",
                "16",
                "--corner-threshold",
                "100",
                "--length-threshold",
                "4.0",
                "--max-iterations",
                "10",
                "--splice-threshold",
                "45",
                "--path-precision",
                "8",
            ],
            {
                "gif_path": "input.gif",
                "output_svg_path": None,
                "fps": DEFAULT_FPS,
                "log_level": "INFO",
                "colormode": "color",
                "hierarchical": "stacked",
                "mode": "spline",
                "filter_speckle": 4,
                "color_precision": 6,
                "layer_difference": 16,
                "corner_threshold": 100,
                "length_threshold": 4.0,
                "max_iterations": 10,
                "splice_threshold": 45,
                "path_precision": 8,
            },
        ),
        (
            ["input.gif", "output file.svg"],
            {
                "gif_path": "input.gif",
                "output_svg_path": "output file.svg",
                "fps": DEFAULT_FPS,
                "log_level": "INFO",
            },
        ),
        (
            ["input.gif", "--fps", "abc"],
            {
                "gif_path": "input.gif",
                "fps": "abc",
                "output_svg_path": None,
                "log_level": "INFO",
            },
        ),  # Should raise error.
    ],
)
def test_parse_cli_arguments_comprehensive(cli_args, expected_parsed_args):
    if cli_args == []:
        with pytest.raises(SystemExit):
            parse_cli_arguments(cli_args)
        return

    if (
        "fps" in expected_parsed_args
        and isinstance(expected_parsed_args["fps"], str)
        and expected_parsed_args["fps"] == "abc"
    ):
        with pytest.raises(SystemExit):
            parse_cli_arguments(cli_args)
        return

    parsed_args = parse_cli_arguments(cli_args)
    for key, expected_value in expected_parsed_args.items():
        assert (
            getattr(parsed_args, key) == expected_value
        ), f"Mismatch: key '{key}': expected {expected_value}, got {getattr(parsed_args, key)}"


@pytest.mark.parametrize(
    ("cli_arguments", "expected_error_message"),
    [
        ([], "the following arguments are required: gif_path"),
        (["in.gif", "--fps", "-1"], "argument -f/--fps: -1 is not a positive float."),
        (["in.gif", "--fps", "0"], "argument -f/--fps: 0 is not a positive float."),
        (
            ["in.gif", "--fps", "abc"],
            "argument -f/--fps: abc is not a valid float.",
        ),
        (
            ["in.gif", "--log-level", "INVALID"],
            "argument -l/--log-level: invalid choice: 'INVALID'",
        ),
        (
            ["in.gif", "--colormode", "invalid"],
            "argument -c/--colormode: invalid choice: 'invalid'",
        ),
        (
            ["in.gif", "--filter-speckle", "-1"],
            "argument -s/--filter-speckle: -1 is not a positive integer.",
        ),
        (
            ["in.gif", "--filter-speckle", "0"],
            "argument -s/--filter-speckle: 0 is not a positive integer.",
        ),
        (
            ["in.gif", "--filter-speckle", "abc"],
            "argument -s/--filter-speckle: abc is not a valid integer.",
        ),
    ],
)
def test_parse_cli_arguments_validation_invalid(cli_arguments, expected_error_message, capsys):
    with pytest.raises(SystemExit) as excinfo:
        parse_cli_arguments(cli_arguments)
    assert excinfo.value.code == 2
    captured_output = capsys.readouterr()
    assert expected_error_message in captured_output.err


def test_save_svg_to_file_success(tmp_path, sample_svg_content):
    output_file_path = str(tmp_path / "output.svg")
    save_svg_to_file(sample_svg_content, output_file_path)

    assert os.path.exists(output_file_path)
    with open(output_file_path, encoding="utf-8") as f:
        written_file_content = f.read()
    assert written_file_content == sample_svg_content


def test_save_svg_to_file_errors(tmp_path, sample_svg_content):
    nonexistent_directory = str(tmp_path / "nonexistent_dir")
    with pytest.raises(FileNotFoundError):
        save_svg_to_file(sample_svg_content, nonexistent_directory + "/output.svg")

    read_only_directory = tmp_path / "readonly"
    read_only_directory.mkdir()
    output_svg_file_ro = read_only_directory / "output.svg"
    output_svg_file_ro.touch()

    os.chmod(output_svg_file_ro, 0o444)

    with pytest.raises(PermissionError):
        save_svg_to_file(sample_svg_content, str(output_svg_file_ro))

    output_directory = tmp_path / "output_dir"
    output_directory.mkdir()
    with pytest.raises(IsADirectoryError):
        save_svg_to_file(sample_svg_content, str(output_directory))


def test_gif_to_animated_svg_write_integration(tmp_path):
    gif_input_path = create_temp_gif(tmp_path, num_frames=3)
    svg_output_path = tmp_path / "output.svg"
    gif_to_animated_svg_write(gif_input_path, str(svg_output_path))
    assert svg_output_path.exists()
    assert svg_output_path.stat().st_size > 0

    output_directory = tmp_path / "output_dir"
    output_directory.mkdir()
    with pytest.raises(IsADirectoryError):
        gif_to_animated_svg_write(gif_input_path, str(output_directory))


def test_gif_to_animated_svg_closes_image(tmp_path, mock_vtracer_instance_for_tests):
    """Test image gets closed even with processing or vtracer errors"""
    gif_path = create_temp_gif(tmp_path)
    mock_image = create_mock_image()
    mock_image_loader = Mock(return_value=mock_image)  # Mock Image Loader
    mock_image_loader.open.return_value = mock_image  #

    # Test case 1: Successful conversion.
    gif_to_animated_svg(gif_path, image_loader=mock_image_loader, vtracer_instance=mock_vtracer_instance_for_tests)
    mock_image.close.assert_called()  # Image Close should have been called.

    # reset call count.
    mock_image.close.reset_mock()

    # Test case 2:  vtracer failure.
    mock_vtracer_fail = create_mock_vtracer(raise_error=True)
    with pytest.raises(Exception, match="VTracer error"):  # Keep as generic exception.
        gif_to_animated_svg(gif_path, image_loader=mock_image_loader, vtracer_instance=mock_vtracer_fail)
    mock_image.close.assert_called()

    # Test Case 3: GIF not animated.
    mock_image.close.reset_mock()  # reset.
    mock_image.is_animated = False
    with pytest.raises(NotAnimatedGifError):
        gif_to_animated_svg(gif_path, image_loader=mock_image_loader)
    mock_image.close.assert_called()

    # Test Case 4: No valid frames
    mock_image.close.reset_mock()  # reset.
    mock_image.is_animated = True
    mock_vtracer_no_valid = create_mock_vtracer(return_svg="<svg></svg>")
    with pytest.raises(NoValidFramesError):
        gif_to_animated_svg(gif_path, image_loader=mock_image_loader, vtracer_instance=mock_vtracer_no_valid)
    mock_image.close.assert_called()

    # Test Case 5: Image closing raises error
    mock_image.close.reset_mock()  # reset.
    mock_image.is_animated = True
    mock_image.close.side_effect = Exception("Simulated close error")  # Set close to raise error.
    with pytest.raises(Exception, match="Simulated close error"):
        gif_to_animated_svg(gif_path, image_loader=mock_image_loader, vtracer_instance=mock_vtracer_instance_for_tests)
    mock_image.close.assert_called()  # should still be called.
