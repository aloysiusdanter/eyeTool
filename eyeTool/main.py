#!/usr/bin/env python3
"""
eyeTool - Camera image loading application using OpenCV.

Target platform: FriendlyElec NanoPi M6 (Rockchip RK3588S, aarch64)
running FriendlyElec Ubuntu 24.04 Desktop. See README.md for the full
list of prerequisites.
"""

import argparse
import glob
import os
import signal
import sys
import time

import cv2

from camera import open_camera, resolve_camera_source

_quit_requested = False


def detect_x_displays() -> list[str]:
    """Return a sorted list of available X display strings (e.g. [':0', ':1'])
    by scanning the XWayland/X11 sockets in /tmp/.X11-unix/."""
    sockets = glob.glob("/tmp/.X11-unix/X*")
    displays = []
    for s in sorted(sockets):
        name = os.path.basename(s)
        if name.startswith("X") and name[1:].isdigit():
            displays.append(f":{name[1:]}")
    return displays


def set_display(display: str) -> None:
    """Set the DISPLAY environment variable and attempt to locate XAUTHORITY."""
    os.environ["DISPLAY"] = display
    if "XAUTHORITY" not in os.environ:
        candidates = [
            os.path.expanduser("~/.Xauthority"),
            f"/run/user/{os.getuid()}/.Xauthority",
        ]
        for path in candidates:
            if os.path.exists(path):
                os.environ["XAUTHORITY"] = path
                break


def auto_set_display() -> str | None:
    """If DISPLAY is not set, auto-detect XWayland sockets and pick ':0'.

    Returns the display string that was set, or None if already set or no
    display was found.
    """
    if os.environ.get("DISPLAY"):
        return None
    displays = detect_x_displays()
    if not displays:
        return None
    chosen = displays[0]
    set_display(chosen)
    print(f"Auto-detected display: {chosen} (use menu option 5 or --display to change)")
    return chosen


def _signal_handler(sig, frame):  # noqa: ARG001
    global _quit_requested
    _quit_requested = True


def _check_display() -> bool:
    """Return True if DISPLAY is set; print an error and return False otherwise."""
    if not os.environ.get("DISPLAY"):
        print("Error: No X display available.")
        print("Hints:")
        print("  - Run menu option 5 to select a display target.")
        print("  - Or pass --display :0 on the command line.")
        print("  - On the NanoPi display, run: xhost +local:")
        return False
    return True


def load_camera_feed(source: int | str) -> None:
    """Initialize camera and display a live feed. Press 'q' to quit."""
    if not _check_display():
        return
    cap = open_camera(source)
    if cap is None:
        return

    global _quit_requested
    _quit_requested = False
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    print(f"Camera feed started on '{source}'. Press 'q'/ESC on the window, or Ctrl+C here to quit.")
    cv2.namedWindow("eyeTool - Camera Feed", cv2.WINDOW_NORMAL)
    try:
        while not _quit_requested:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            cv2.imshow("eyeTool - Camera Feed", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # 27 = ESC
                break
            if cv2.getWindowProperty("eyeTool - Camera Feed", cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera feed closed.")


def capture_single_image(source: int | str, output: str) -> None:
    """Capture a single frame on SPACE, save to ``output``; 'q' to quit."""
    if not _check_display():
        return
    cap = open_camera(source)
    if cap is None:
        return

    print(f"Press SPACE to capture image to '{output}', 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            cv2.imshow("eyeTool - Capture Mode", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                if cv2.imwrite(output, frame):
                    print(f"Image saved as {output}")
                else:
                    print(f"Error: Failed to write image to '{output}'.")
                break
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def probe_camera(source: int | str, warmup_frames: int = 30,
                 warmup_timeout_s: float = 3.0) -> None:
    """Open the camera briefly and print basic info, without a GUI window.

    Useful for smoke-testing the camera wiring over SSH before touching
    the display. Retries a few times because MIPI-CSI sensors on the
    NanoPi M6 often drop the first several frames while AE/AWB settle.
    """
    cap = open_camera(source)
    if cap is None:
        return
    try:
        backend = cap.getBackendName()
        w_prop = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_prop = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Opened camera: source='{source}', backend='{backend}', "
              f"reported={w_prop}x{h_prop}@{fps:.1f}fps")

        deadline = time.monotonic() + warmup_timeout_s
        attempts = 0
        while time.monotonic() < deadline and attempts < warmup_frames:
            ret, frame = cap.read()
            attempts += 1
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"Probe OK after {attempts} read(s): frame={w}x{h}")
                return
            time.sleep(0.05)

        print(f"Probe failed: no frame after {attempts} read(s) within "
              f"{warmup_timeout_s:.1f}s.")
        print("Hints:")
        print("  - For MIPI-CSI on RK3588S, OpenCV's V4L2 backend may not "
              "configure the ISP pipeline. Try a GStreamer-based capture, "
              "or test with a USB UVC camera on --device /dev/video0.")
        print("  - Confirm the sensor is actually connected and detected: "
              "`v4l2-ctl --list-devices` and `dmesg | grep -i camera`.")
    finally:
        cap.release()


def select_display_menu() -> None:
    """Interactive sub-menu to choose which X display target to use."""
    displays = detect_x_displays()
    current = os.environ.get("DISPLAY", "(not set)")
    if not displays:
        print("No X displays found in /tmp/.X11-unix/.")
        print("Make sure the NanoPi desktop is running and XWayland is active.")
        return
    print("\n--- Select Display Target ---")
    for i, d in enumerate(displays, 1):
        marker = " (current)" if d == current else ""
        label = " [built-in LCD]" if d == ":0" else " [HDMI / secondary]" if d == ":1" else ""
        print(f"  {i}. {d}{label}{marker}")
    raw = input(f"Enter number (1-{len(displays)}) or display string (e.g. :0): ").strip()
    if not raw:
        return
    if raw.startswith(":"):
        chosen = raw
    elif raw.isdigit() and 1 <= int(raw) <= len(displays):
        chosen = displays[int(raw) - 1]
    else:
        print(f"Invalid input '{raw}', display not changed.")
        return
    set_display(chosen)
    print(f"Display target set to {chosen}.")
    print("Note: if cv2.imshow fails with 'Authorization required', run on the")
    print("NanoPi display: xhost +local:")


def interactive_menu(source: int | str, output: str) -> None:
    print("=== eyeTool - Camera Application ===")
    print(f"Camera source: {source}")
    print(f"Display target: {os.environ.get('DISPLAY', '(not set)')}")
    print("1. Live camera feed")
    print("2. Capture single image")
    print("3. Probe camera (no GUI)")
    print("4. Exit")
    print("5. Select display target")

    while True:
        choice = input("\nEnter your choice (1-5): ").strip()
        if choice == "1":
            load_camera_feed(source)
        elif choice == "2":
            capture_single_image(source, output)
        elif choice == "3":
            probe_camera(source)
        elif choice == "4":
            print("Goodbye!")
            break
        elif choice == "5":
            select_display_menu()
        else:
            print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")


def parse_args(argv: list[str]) -> argparse.Namespace:
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
        choices=("menu", "feed", "capture", "probe"),
        default="menu",
        help="Run mode. 'menu' (default) launches the interactive menu.",
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.display is not None:
        set_display(args.display)
    else:
        auto_set_display()

    source = resolve_camera_source(args.device)

    if args.mode == "menu":
        interactive_menu(source, args.output)
    elif args.mode == "feed":
        load_camera_feed(source)
    elif args.mode == "capture":
        capture_single_image(source, args.output)
    elif args.mode == "probe":
        probe_camera(source)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}")
        sys.exit(1)
