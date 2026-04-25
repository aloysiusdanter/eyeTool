"""
camera.py - Camera device discovery, resolution selection, and OpenCV capture
initialisation for eyeTool.

Target platform: FriendlyElec NanoPi M6 (Rockchip RK3588S, aarch64).
"""

import os
import re
import subprocess

import cv2

DEFAULT_CAMERA_SYMLINK = "/dev/video-camera0"

_MJPEG_BONUS: float = 3.0


def get_best_resolution(device_path: str) -> tuple[int, int, float, str] | None:
    """Query ``v4l2-ctl`` for all discrete resolutions/frame-rates on
    *device_path* and return ``(width, height, fps, fmt)`` for the entry with
    the highest ``width × height × fps × format_bonus`` score.

    MJPEG receives a bonus multiplier (``_MJPEG_BONUS``) so it is preferred
    over raw formats (e.g. YUYV) at the same resolution when both are listed.

    Returns ``None`` if ``v4l2-ctl`` is unavailable, the device is not a
    character device, or no discrete sizes are reported.
    """
    if not os.path.exists(device_path):
        return None
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", device_path, "--list-formats-ext"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    best: tuple[int, int, float, str] | None = None
    best_score: float = -1.0
    current_w: int | None = None
    current_h: int | None = None
    current_fmt: str = "YUYV"
    current_fmt_bonus: float = 1.0

    size_re = re.compile(r"Size:\s+Discrete\s+(\d+)x(\d+)")
    fps_re = re.compile(r"\(([\d.]+)\s+fps\)")
    fmt_re = re.compile(r"'([A-Z0-9]+)'")

    for line in result.stdout.splitlines():
        fmt_m = fmt_re.search(line)
        if fmt_m and "Size" not in line:
            current_fmt = fmt_m.group(1)
            current_fmt_bonus = _MJPEG_BONUS if current_fmt == "MJPG" else 1.0
            current_w = None
            current_h = None
            continue
        m = size_re.search(line)
        if m:
            current_w, current_h = int(m.group(1)), int(m.group(2))
            continue
        if current_w is not None:
            m = fps_re.search(line)
            if m:
                fps = float(m.group(1))
                score = current_w * current_h * fps * current_fmt_bonus
                if score > best_score:
                    best_score = score
                    best = (current_w, current_h, fps, current_fmt)

    return best


def find_usb_camera() -> str | None:
    """Return the first ``/dev/videoN`` node that belongs to a USB UVC
    device, as reported by ``v4l2-ctl --list-devices``.

    Returns ``None`` if no USB device is found or ``v4l2-ctl`` is unavailable.
    """
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    usb_block = False
    dev_re = re.compile(r"^\s+(/dev/video\d+)")
    for line in result.stdout.splitlines():
        if re.search(r"usb", line, re.IGNORECASE):
            usb_block = True
            continue
        if usb_block:
            m = dev_re.match(line)
            if m:
                return m.group(1)
            if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                usb_block = False
    return None


def _has_discrete_formats(device_path: str) -> bool:
    """Return True if *device_path* reports at least one discrete capture size.

    MIPI-CSI ISP nodes always report stepwise size ranges even with no sensor
    attached. Only a real connected sensor (USB UVC or attached MIPI-CSI)
    reports discrete ``WxH`` sizes, so this is used to confirm the MIPI-CSI
    symlink has an actual sensor before preferring it over USB auto-discovery.
    """
    return get_best_resolution(device_path) is not None


def resolve_camera_source(device: str | None) -> int | str:
    """Return the argument that should be passed to ``cv2.VideoCapture``.

    - If ``device`` is ``None``, prefer ``/dev/video-camera0`` (NanoPi M6
      MIPI-CSI symlink) when it exists, then try to auto-discover the first
      USB UVC camera via ``v4l2-ctl``, and finally fall back to index ``0``.
    - If ``device`` is a pure integer string, return it as ``int``.
    - Otherwise return it verbatim (e.g. ``"/dev/video1"``).
    """
    if device is None:
        if os.path.exists(DEFAULT_CAMERA_SYMLINK) and _has_discrete_formats(DEFAULT_CAMERA_SYMLINK):
            return DEFAULT_CAMERA_SYMLINK
        usb = find_usb_camera()
        if usb is not None:
            print(f"No MIPI-CSI camera found; auto-detected USB camera: {usb}")
            return usb
        return 0
    if device.isdigit():
        return int(device)
    return device


def open_camera(source: int | str) -> cv2.VideoCapture | None:
    """Open the camera, auto-apply the best resolution for USB devices,
    and print useful diagnostics on failure.

    For USB UVC cameras (any ``/dev/videoN`` path that is not the MIPI-CSI
    symlink), ``v4l2-ctl`` is queried to find the ``(width, height)`` that
    maximises ``width × height × fps`` across all reported formats.
    MIPI-CSI (``/dev/video-camera0``) is left untouched.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open camera '{source}'.")
        print("Hints:")
        print("  - Run 'v4l2-ctl --list-devices' to see available cameras.")
        print("  - Make sure your user is in the 'video' group.")
        print("  - Try '--device /dev/video-camera0' or another index.")
        return None

    device_path: str | None = None
    if isinstance(source, str) and source != DEFAULT_CAMERA_SYMLINK:
        device_path = source
    elif isinstance(source, int):
        device_path = f"/dev/video{source}"

    if device_path is not None:
        best = get_best_resolution(device_path)
        if best is not None:
            w, h, fps, fmt = best
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS, fps)
            if fmt == "MJPG":
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            print(f"Auto-selected: {w}x{h} @ {fps:.0f} fps ({fmt}) for '{device_path}'")

    return cap
