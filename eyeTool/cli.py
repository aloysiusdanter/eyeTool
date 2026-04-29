"""CLI argument parsing for eyeTool."""

from __future__ import annotations

import argparse


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments for eyeTool."""
    parser = argparse.ArgumentParser(
        description="eyeTool - camera capture utility for the NanoPi M6.",
    )
    parser.add_argument(
        "--device",
        "-d",
        default=None,
        help=(
            "Camera device: integer index (e.g. '0') or path "
            "(e.g. '/dev/video-camera0'). Defaults to '/dev/video-camera0' "
            "if present, else 0."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default="captured_image.jpg",
        help="Output filename for capture mode (default: captured_image.jpg).",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=("menu", "feed", "capture", "probe", "record-multi"),
        default="menu",
        help="Run mode. 'menu' (default) launches the interactive menu. 'record-multi' auto-starts multi-camera recording.",
    )
    parser.add_argument(
        "--display",
        "-D",
        default=None,
        metavar="DISPLAY",
        help=(
            "X display to use for GUI windows, e.g. ':0' (built-in LCD) or "
            "':1' (HDMI). Overrides auto-detection and $DISPLAY."
        ),
    )
    return parser.parse_args(argv)
