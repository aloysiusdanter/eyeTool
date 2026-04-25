#!/usr/bin/env python3
"""
eyeTool - Camera image loading application using OpenCV.

Target platform: FriendlyElec NanoPi M6 (Rockchip RK3588S, aarch64)
running FriendlyElec Ubuntu 24.04 Desktop. See README.md for the full
list of prerequisites.
"""

import argparse
import signal
import sys
import time

import cv2

from camera import open_camera, resolve_camera_source

_quit_requested = False


def _signal_handler(sig, frame):  # noqa: ARG001
    global _quit_requested
    _quit_requested = True


def load_camera_feed(source: int | str) -> None:
    """Initialize camera and display a live feed. Press 'q' to quit."""
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


def interactive_menu(source: int | str, output: str) -> None:
    print("=== eyeTool - Camera Application ===")
    print(f"Camera source: {source}")
    print("1. Live camera feed")
    print("2. Capture single image")
    print("3. Probe camera (no GUI)")
    print("4. Exit")

    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        if choice == "1":
            load_camera_feed(source)
        elif choice == "2":
            capture_single_image(source, output)
        elif choice == "3":
            probe_camera(source)
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
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
