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
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

from camera import open_camera, resolve_camera_source
from rknn_yolov8 import infer as rknn_infer, warmup as rknn_warmup
from pipeline import (Detector, FrameSource, MultiDetector,
                       overlay_detections, overlay_tile_detections)
from config import get_config
from stream import StreamManager
from compositor import GridCompositor
from hotplug import HotplugMonitor
from zones import load_zones
from preprocess import Preprocess

_quit_requested = False

_detection_enabled = True
_detection_confidence = 0.5
_target_fps = 30
_detect_every_n = 1  # run NPU on every Nth captured frame
_use_multi_core = False  # use 3 NPU cores (round-robin)

PERSON_CLASS_ID = 0  # COCO class index for "person"


def draw_detections(frame: np.ndarray, confidence: float = 0.5) -> int:
    """Run YOLOv8n person detection on *frame* using the RK3588 NPU.

    Draws green bounding boxes with confidence in-place on *frame* and
    returns the number of persons detected.
    """
    boxes, classes, scores, scale, (pad_w, pad_h) = rknn_infer(frame, conf_thres=confidence)
    count = 0
    for (x1, y1, x2, y2), score, cls in zip(boxes, scores, classes):
        if int(cls) != PERSON_CLASS_ID:
            continue
        # Map back from 640-input letterboxed space to original frame coords
        x1 = int((x1 - pad_w) / scale)
        x2 = int((x2 - pad_w) / scale)
        y1 = int((y1 - pad_h) / scale)
        y2 = int((y2 - pad_h) / scale)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"person {score:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        count += 1
    return count


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


def _merge_mutter_xauth(display: str) -> bool:
    """Merge GNOME/mutter's XWayland auth cookie into ~/.Xauthority.

    GNOME creates a private Xauthority file for XWayland at a path like
    /run/user/<uid>/.mutter-Xwaylandauth.XXXXXX. When connecting over SSH
    the session lacks that cookie, causing "Authorization required" errors.
    This function extracts the cookie for *display* and merges it into
    ~/.Xauthority so cv2.imshow works from SSH without manual xauth steps.

    Returns True if the merge succeeded, False otherwise.
    """
    mutter_files = glob.glob(f"/run/user/{os.getuid()}/.mutter-Xwaylandauth.*")
    if not mutter_files:
        return False
    mutter_auth = mutter_files[0]
    xauth_home = os.path.expanduser("~/.Xauthority")
    try:
        extract = subprocess.run(
            ["xauth", "-f", mutter_auth, "extract", "-", display],
            capture_output=True,
            timeout=5,
        )
        if extract.returncode != 0:
            return False
        merge = subprocess.run(
            ["xauth", "-f", xauth_home, "merge", "-"],
            input=extract.stdout,
            capture_output=True,
            timeout=5,
        )
        return merge.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def set_display(display: str) -> None:
    """Set DISPLAY and auto-configure XAUTHORITY for GNOME/XWayland sessions.

    Merges the mutter XWayland auth cookie into ~/.Xauthority so that
    cv2.imshow works from SSH without requiring manual xhost or xauth steps.
    """
    os.environ["DISPLAY"] = display
    _merge_mutter_xauth(display)
    xauth_home = os.path.expanduser("~/.Xauthority")
    if os.path.exists(xauth_home):
        os.environ["XAUTHORITY"] = xauth_home
    elif "XAUTHORITY" not in os.environ:
        mutter_files = glob.glob(f"/run/user/{os.getuid()}/.mutter-Xwaylandauth.*")
        if mutter_files:
            os.environ["XAUTHORITY"] = mutter_files[0]


def auto_set_display() -> str | None:
    """Choose a DISPLAY for this run.

    Precedence (highest first):
      1. ``DISPLAY`` already exported by the parent shell -- never override.
      2. ``display.target`` in ``user_settings.json`` / manufacturer
         default -- the user's persistent preference.
      3. First X11 socket found in ``/tmp/.X11-unix/`` -- typically ``:0``.

    Returns the display string actually set, or ``None`` if nothing
    suitable was found.
    """
    if os.environ.get("DISPLAY"):
        return None

    # 2. saved preference
    try:
        saved = (get_config().get("display.target", "") or "").strip()
    except Exception:  # noqa: BLE001 -- config init should never fail us here
        saved = ""
    if saved:
        available = detect_x_displays()
        if not available or saved in available:
            set_display(saved)
            print(f"Display target: {saved} (saved preference)")
            return saved
        print(f"[main] saved display preference {saved!r} not present "
              f"on this system; available={available}; falling back")

    # 3. first detected socket
    displays = detect_x_displays()
    if not displays:
        return None
    chosen = displays[0]
    set_display(chosen)
    print(f"Auto-detected display: {chosen} "
          f"(use menu option 9 or --display to change)")
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


def letterbox_frame(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Scale frame to fit target dimensions while preserving aspect ratio.

    Returns a canvas of size (target_h, target_w, 3) with the scaled frame
    centered and black bars filling the remaining space.
    """
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas


def load_camera_feed(source: int | str) -> None:
    """Async pipeline: capture thread + detector thread + display loop.

    Press 'q'/ESC on the window or Ctrl+C on the console to quit.
    """
    if not _check_display():
        return
    cap = open_camera(source)
    if cap is None:
        return

    global _quit_requested
    _quit_requested = False
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Scale to display size (built-in LCD is 800x480)
    display_w, display_h = 800, 480
    det_status = "ON" if _detection_enabled else "OFF"
    print(f"Feed info: {w}x{h} @ {fps:.1f} fps (letterboxed to {display_w}x{display_h})  |  source: {source}  |  display: {os.environ.get('DISPLAY', '?')}")
    multi_status = "3-CORE" if _use_multi_core else "1-CORE"
    print(f"Detection: {det_status}  |  confidence: {_detection_confidence}  |  target FPS: {_target_fps}  |  detect every N: {_detect_every_n}  |  NPU: {multi_status}")
    print("Press 'q'/ESC on the window, or Ctrl+C here to quit.")

    frame_source = FrameSource(cap)
    detector: Detector | None = None
    frame_source.start()
    if _detection_enabled:
        detector = Detector(frame_source,
                            conf_thres=_detection_confidence,
                            detect_every_n=_detect_every_n,
                            use_multi_core=_use_multi_core)
        detector.start()

    cv2.namedWindow("eyeTool - Camera Feed", cv2.WINDOW_NORMAL)
    first_frame = True
    frame_interval = 1.0 / _target_fps if _target_fps > 0 else 0
    prev_time = time.monotonic()
    actual_fps = 0.0
    last_stats_ts = time.monotonic()
    try:
        while not _quit_requested:
            if not frame_source.wait_new(timeout=1.0):
                # Camera stalled; loop back and re-check _quit_requested.
                continue
            snap = frame_source.get_latest()
            if snap is None:
                continue
            frame, frame_ts = snap
            # Work on a shallow copy so the capture thread can safely
            # overwrite the underlying buffer on its next read.
            frame = frame.copy()

            now = time.monotonic()
            if detector is not None:
                det = detector.get_latest()
                overlay_detections(frame, det, now)

            frame_letterboxed = letterbox_frame(frame, display_w, display_h)

            elapsed = now - prev_time
            if elapsed > 0:
                actual_fps = 1.0 / elapsed
            prev_time = now

            cv2.putText(frame_letterboxed, f"FPS: {actual_fps:.0f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)

            # Periodic pipeline stats on the console.
            if now - last_stats_ts > 2.0:
                det_fps = detector.fps() if detector else 0.0
                print(f"Pipeline: capture={frame_source.fps():5.1f} Hz  "
                      f"detect={det_fps:5.1f} Hz  display={actual_fps:5.1f} Hz")
                last_stats_ts = now

            cv2.imshow("eyeTool - Camera Feed", frame_letterboxed)
            if first_frame:
                cv2.setWindowProperty("eyeTool - Camera Feed",
                                      cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
                first_frame = False
            else:
                # Keep fullscreen property set (XWayland sometimes loses it)
                cv2.setWindowProperty("eyeTool - Camera Feed",
                                      cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)

            wait_ms = max(1, int(frame_interval * 1000 - elapsed * 1000))
            key = cv2.waitKey(wait_ms) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # 27 = ESC
                break
            if cv2.getWindowProperty("eyeTool - Camera Feed",
                                     cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if detector is not None:
            detector.stop()
        frame_source.stop()
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
    """Interactive sub-menu to choose which X display target to use.

    The choice is applied to *this* process immediately (so cv2.imshow
    works straight away) and the user is then prompted whether to
    persist the choice as the default for **future launches** by
    writing ``display.target`` into ``user_settings.json``.
    """
    cfg = get_config()
    displays = detect_x_displays()
    current = os.environ.get("DISPLAY", "(not set)")
    saved = (cfg.get("display.target", "") or "").strip() or "(none -- auto)"
    if not displays:
        print("No X displays found in /tmp/.X11-unix/.")
        print("Make sure the NanoPi desktop is running and XWayland is active.")
        return
    print("\n--- Select Display Target ---")
    print(f"  active this session : {current}")
    print(f"  saved as default    : {saved}")
    print()
    for i, d in enumerate(displays, 1):
        marker = " (current)" if d == current else ""
        label = (" [built-in LCD]" if d == ":0"
                 else " [HDMI / secondary]" if d == ":1" else "")
        print(f"  {i}. {d}{label}{marker}")
    print(f"  C. Clear saved default (revert to auto-detect)")
    raw = input(f"Enter number (1-{len(displays)}), display string "
                f"(e.g. :0), or 'C' to clear: ").strip()
    if not raw:
        return

    # --- clear saved default --------------------------------------------
    if raw.lower() == "c":
        cfg.set("display.target", "")
        cfg.save_user()
        print("Saved display preference cleared. Future launches will "
              "auto-detect.")
        return

    if raw.startswith(":"):
        chosen = raw
    elif raw.isdigit() and 1 <= int(raw) <= len(displays):
        chosen = displays[int(raw) - 1]
    else:
        print(f"Invalid input '{raw}', display not changed.")
        return

    set_display(chosen)
    print(f"Display target set to {chosen} (this session).")
    print("Note: if cv2.imshow fails with 'Authorization required', run on")
    print("the NanoPi display: xhost +local:")

    save = input(f"Also save {chosen} as the default for future launches? "
                 f"[y/N]: ").strip().lower()
    if save == "y":
        cfg.set("display.target", chosen)
        cfg.save_user()
        print(f"Saved: future launches will use {chosen} unless overridden "
              f"by --display.")


def detection_settings_menu() -> None:
    """Interactive sub-menu for YOLO detection settings."""
    global _detection_enabled, _detection_confidence, _target_fps, _detect_every_n, _use_multi_core
    while True:
        det_status = "ON" if _detection_enabled else "OFF"
        multi_status = "3-CORE" if _use_multi_core else "1-CORE"
        print(f"\n--- Detection Settings ---")
        print(f"  1. Toggle detection [{det_status}]")
        print(f"  2. Set confidence threshold [{_detection_confidence:.2f}]")
        print(f"  3. Set target display FPS [{_target_fps}]")
        print(f"  4. Detect every N captured frames [{_detect_every_n}]")
        print(f"  5. Use 3 NPU cores (round-robin) [{multi_status}]")
        print(f"  6. Back to main menu")
        raw = input("Enter choice (1-6): ").strip()
        if raw == "1":
            _detection_enabled = not _detection_enabled
            print(f"Detection {'enabled' if _detection_enabled else 'disabled'}.")
        elif raw == "2":
            val = input(f"Confidence threshold (0.0-1.0) [{_detection_confidence:.2f}]: ").strip()
            if val:
                try:
                    v = float(val)
                    if 0.0 <= v <= 1.0:
                        _detection_confidence = v
                        print(f"Confidence threshold set to {_detection_confidence:.2f}.")
                    else:
                        print("Value must be between 0.0 and 1.0.")
                except ValueError:
                    print("Invalid number.")
        elif raw == "3":
            val = input(f"Target FPS (1-120) [{_target_fps}]: ").strip()
            if val:
                try:
                    v = int(val)
                    if 1 <= v <= 120:
                        _target_fps = v
                        print(f"Target FPS set to {_target_fps}.")
                    else:
                        print("Value must be between 1 and 120.")
                except ValueError:
                    print("Invalid number.")
        elif raw == "4":
            val = input(f"Detect every N frames (1-10) [{_detect_every_n}]: ").strip()
            if val:
                try:
                    v = int(val)
                    if 1 <= v <= 10:
                        _detect_every_n = v
                        print(f"Detect-every-N set to {_detect_every_n}.")
                    else:
                        print("Value must be between 1 and 10.")
                except ValueError:
                    print("Invalid number.")
        elif raw == "5":
            _use_multi_core = not _use_multi_core
            print(f"Multi-core NPU {'enabled' if _use_multi_core else 'disabled'}.")
            print("  3-CORE: round-robin across NPU_CORE_0/1/2 (~3× detection rate)")
            print("  1-CORE: single NPU_CORE_AUTO (lower power, ~10 Hz detect)")
        elif raw == "6":
            break
        else:
            print("Invalid choice.")


def load_multi_camera_feed() -> None:
    """Multi-stream 2x2 grid display (Phase A foundation; no detection yet).

    Opens every currently-connected USB UVC camera into a slot determined
    by its kernel-stable USB port path. Press 'q'/ESC on the window or
    Ctrl+C on the console to quit.
    """
    if not _check_display():
        return

    cfg = get_config()
    max_streams = int(cfg.get("streams.max_streams", 4))
    display_w = int(cfg.get("display.width", 800))
    display_h = int(cfg.get("display.height", 480))
    target_fps = int(cfg.get("display.target_fps", 30))
    watchdog = float(cfg.get("streams.watchdog_stall_s", 2.0))

    # Restore saved port_path -> slot_id bindings from zones.json
    saved_bindings: dict[str, int] = {}
    for sid, slot_cfg in cfg.all_slots().items():
        pp = slot_cfg.get("port_path")
        if pp:
            saved_bindings[pp] = sid

    # Live binding-change persistence: a fresh hot-plug into an empty slot
    # also triggers this, so zones.json always reflects "what plugged in
    # where" without waiting until shutdown.
    _persist_lock = threading.Lock()
    def _persist_binding(port_path: str, slot_id: int) -> None:
        with _persist_lock:
            slot_cfg = cfg.slot(slot_id) or {}
            slot_cfg["port_path"] = port_path
            cfg.update_slot(slot_id, slot_cfg)
            cfg.save_zones()
        print(f"[main] saved binding: slot {slot_id} <- {port_path}")

    # Load any persisted preprocessing config per slot.
    saved_preprocess: dict[int, Preprocess] = {}
    for sid, slot_cfg in cfg.all_slots().items():
        pre_cfg = slot_cfg.get("preprocessing")
        if pre_cfg:
            try:
                pp = Preprocess.from_dict(pre_cfg)
                if not pp.is_identity():
                    saved_preprocess[sid] = pp
            except (TypeError, ValueError) as e:
                print(f"[main] slot {sid}: bad preprocessing config: {e}")

    manager = StreamManager(max_streams=max_streams,
                            watchdog_stall_s=watchdog,
                            saved_bindings=saved_bindings,
                            on_binding_change=_persist_binding,
                            saved_preprocess=saved_preprocess)
    manager.open_all_present()

    # Start hot-plug monitor (re-uses StreamManager.on_hotplug for add/remove)
    monitor = HotplugMonitor(manager.on_hotplug)
    if monitor.start():
        print("[main] hot-plug monitor running (pyudev)")
    else:
        print("[main] hot-plug monitor disabled")

    compositor = GridCompositor(display_w=display_w, display_h=display_h)

    # Detector across all active streams (single shared NPU, round-robin).
    detector: MultiDetector | None = None
    if _detection_enabled:
        detector = MultiDetector(manager,
                                 conf_thres=_detection_confidence,
                                 detect_every_n=_detect_every_n,
                                 use_multi_core=_use_multi_core)
        detector.start()

    # Load alarm-zone polygons (per-slot, frame-space coordinates)
    zones = load_zones(cfg)
    if zones:
        for sid, z in zones.items():
            print(f"[main] slot {sid}: alarm zone with {len(z.polygon)} vertices")
    else:
        print("[main] no alarm zones defined (red/green ring not active)")

    # Per-tile overlay: map detection boxes from inference-space onto the
    # tile's letterboxed crop. Render order: zone polygon -> boxes (so
    # boxes always sit on top of the translucent fill).
    _now_ts_ref = [time.monotonic()]
    def _tile_overlay(slot_id, tile, scale, off_x, off_y, snap):
        zone = zones.get(slot_id)
        if zone is not None:
            zone.draw_on_tile(tile, scale, off_x, off_y)
        if detector is None:
            return
        det = detector.get_result(slot_id)
        overlay_tile_detections(tile, det, scale, off_x, off_y,
                                 _now_ts_ref[0], zone=zone)
    compositor.set_overlay(_tile_overlay)

    global _quit_requested
    _quit_requested = False
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    det_status = "ON" if _detection_enabled else "OFF"
    multi_status = "3-CORE" if _use_multi_core else "1-CORE"
    print(f"Multi-camera feed: {display_w}x{display_h}, target {target_fps} FPS, {max_streams} slots")
    print(f"Detection: {det_status}  |  conf: {_detection_confidence}  |  every-N: {_detect_every_n}  |  NPU: {multi_status}")
    print("Press 'q'/ESC on the window, or Ctrl+C here to quit.")

    cv2.namedWindow("eyeTool - Multi Feed", cv2.WINDOW_NORMAL)
    first_frame = True
    last_stats_ts = time.monotonic()
    frame_interval = 1.0 / target_fps if target_fps > 0 else 0.0
    prev_time = time.monotonic()
    actual_fps = 0.0

    try:
        while not _quit_requested:
            now = time.monotonic()
            _now_ts_ref[0] = now  # tile overlay reads this for staleness colour
            snapshots = manager.snapshot()
            canvas = compositor.render(snapshots)

            elapsed = now - prev_time
            if elapsed > 0:
                actual_fps = 1.0 / elapsed
            prev_time = now

            cv2.putText(canvas, f"{actual_fps:4.1f} FPS",
                        (display_w - 90, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
                        cv2.LINE_AA)

            if now - last_stats_ts > 2.0:
                parts = []
                for snap in snapshots:
                    parts.append(f"slot{snap.slot_id}={snap.state.value}/{snap.capture_fps:4.1f}Hz")
                det_fps = detector.fps() if detector else 0.0
                print("Pipeline: " + "  ".join(parts)
                      + f"  detect={det_fps:5.1f} Hz  display={actual_fps:5.1f} Hz")
                last_stats_ts = now

            cv2.imshow("eyeTool - Multi Feed", canvas)
            cv2.setWindowProperty("eyeTool - Multi Feed",
                                  cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
            first_frame = False

            wait_ms = max(1, int(frame_interval * 1000 - elapsed * 1000)) if frame_interval > 0 else 1
            key = cv2.waitKey(wait_ms) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty("eyeTool - Multi Feed",
                                     cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        monitor.stop()
        if detector is not None:
            detector.stop()
        manager.stop_all()
        cv2.destroyAllWindows()
        print("Multi-camera feed closed.")


def _grab_fresh_frame(devnode: str, warmup_frames: int = 5):
    """Open *devnode*, throw away the first few (often stale/black) frames,
    return the next valid frame plus the open VideoCapture released."""
    cap = open_camera(devnode)
    if cap is None:
        return None
    try:
        frame = None
        for _ in range(warmup_frames + 1):
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
        return frame
    finally:
        try:
            cap.release()
        except Exception:  # noqa: BLE001
            pass


def setup_zone_for_slot(slot_id: int) -> bool:
    """Run the polygon editor against a single fresh frame from *slot_id*.

    Returns True if the user saved a polygon (or deleted the existing
    one), False if they cancelled or no camera is available.
    """
    if not _check_display():
        return False

    # Lazy imports so this menu item works even when running headless tests
    from polygon_editor import run as run_polygon_editor
    from hotplug import list_cameras

    cfg = get_config()
    slot_cfg = cfg.slot(slot_id) or {}
    port_path = slot_cfg.get("port_path")
    if not port_path:
        print(f"Slot {slot_id}: no camera bound. Use the multi-camera feed "
              f"first to bind a USB port to this slot.")
        return False

    cams = {c.port_path: c for c in list_cameras()}
    info = cams.get(port_path)
    if info is None:
        print(f"Slot {slot_id}: camera at {port_path} is not connected. "
              f"Plug it in and try again.")
        return False

    print(f"Slot {slot_id}: opening live feed on {info.devnode} ...")
    cap = open_camera(info.devnode)
    if cap is None:
        print(f"Slot {slot_id}: failed to open {info.devnode}.")
        return False

    # Read a few warmup frames -- some V4L2 drivers (notably the SPCA
    # USB cameras here) deliver black/garbled buffers right after
    # opening, and the first valid frame fixes the canvas size used by
    # the editor for the rest of the session.
    initial_frame = None
    for _ in range(8):
        ok, f = cap.read()
        if ok and f is not None and f.size > 0:
            initial_frame = f
    if initial_frame is None:
        cap.release()
        print(f"Slot {slot_id}: no frames received from {info.devnode}.")
        return False

    label = f"slot {slot_id} ({info.model or info.port_path})"
    existing = slot_cfg.get("polygon") or []
    existing_tuples = [(int(p[0]), int(p[1])) for p in existing]

    # Editor auto-detects screen resolution via xrandr and runs against
    # the *live* capture handed in via ``cap=``.
    try:
        new_poly = run_polygon_editor(
            initial_frame, existing_polygon=existing_tuples,
            slot_label=label, cap=cap,
        )
    finally:
        try:
            cap.release()
        except Exception:  # noqa: BLE001
            pass

    if new_poly is None:
        print(f"Slot {slot_id}: edit cancelled, no changes saved.")
        return False
    if new_poly == []:
        slot_cfg.pop("polygon", None)
        cfg.update_slot(slot_id, slot_cfg)
        cfg.save_zones()
        print(f"Slot {slot_id}: polygon removed, zones.json updated.")
        return True

    slot_cfg["polygon"] = [list(p) for p in new_poly]
    cfg.update_slot(slot_id, slot_cfg)
    cfg.save_zones()
    print(f"Slot {slot_id}: saved polygon with {len(new_poly)} vertices.")
    return True


def setup_preprocess_for_slot(slot_id: int) -> bool:
    """Live brightness/contrast/saturation/gamma editor for *slot_id*.

    Returns True if changes were saved to ``zones.json``.
    """
    if not _check_display():
        return False
    from preprocess_editor import run as run_pp_editor
    from hotplug import list_cameras

    cfg = get_config()
    slot_cfg = cfg.slot(slot_id) or {}
    port_path = slot_cfg.get("port_path")
    if not port_path:
        print(f"Slot {slot_id}: no camera bound; run multi-camera feed "
              f"first to bind a USB port to this slot.")
        return False
    cams = {c.port_path: c for c in list_cameras()}
    info = cams.get(port_path)
    if info is None:
        print(f"Slot {slot_id}: camera at {port_path} not connected.")
        return False

    print(f"Slot {slot_id}: opening live feed on {info.devnode} ...")
    cap = open_camera(info.devnode)
    if cap is None:
        print(f"Slot {slot_id}: failed to open {info.devnode}.")
        return False

    initial = Preprocess.from_dict(slot_cfg.get("preprocessing"))
    label = f"slot {slot_id} ({info.model or info.port_path})"

    try:
        new_pp = run_pp_editor(cap, initial=initial, slot_label=label)
    finally:
        try:
            cap.release()
        except Exception:  # noqa: BLE001
            pass

    if new_pp is None:
        print(f"Slot {slot_id}: preprocessing edit cancelled.")
        return False

    if new_pp.is_identity():
        slot_cfg.pop("preprocessing", None)
    else:
        slot_cfg["preprocessing"] = new_pp.to_dict()
    cfg.update_slot(slot_id, slot_cfg)
    cfg.save_zones()
    print(f"Slot {slot_id}: saved preprocessing -> {new_pp.to_dict()}")
    return True


def preprocess_settings_menu() -> None:
    cfg = get_config()
    max_streams = int(cfg.get("streams.max_streams", 4))
    while True:
        from hotplug import list_cameras
        connected = {c.port_path: c for c in list_cameras()}

        print("\n=== Image preprocessing ===")
        any_bound = False
        for sid in range(max_streams):
            slot_cfg = cfg.slot(sid) or {}
            pp = slot_cfg.get("port_path", "")
            cam = connected.get(pp)
            state = "(no camera bound)" if not pp else (
                f"{cam.model or 'connected'} -> {cam.devnode}"
                if cam else "BOUND but UNAVAILABLE")
            pre = slot_cfg.get("preprocessing") or {}
            pre_state = ("default" if not pre else
                         f"B={pre.get('brightness', 0):+.2f} "
                         f"C={pre.get('contrast', 1):.2f} "
                         f"S={pre.get('saturation', 1):.2f} "
                         f"G={pre.get('gamma', 1):.2f}")
            print(f"  {sid}.  {state:42}  [{pre_state}]")
            if pp:
                any_bound = True
        if not any_bound:
            print("  -- no slots bound yet. Run option 2 first. --")
        print(f"  {max_streams}.  Back")

        choice = input(f"\nSelect slot to tune (0-{max_streams - 1}, "
                       f"{max_streams}=back): ").strip()
        if not choice.isdigit():
            print("Invalid choice.")
            continue
        n = int(choice)
        if n == max_streams:
            return
        if 0 <= n < max_streams:
            setup_preprocess_for_slot(n)
        else:
            print("Invalid choice.")


def configuration_menu() -> None:
    """Save current state as manufacturer default / restore / clear."""
    cfg = get_config()
    while True:
        has_zones = cfg.has_manufacturer_zones()
        print("\n=== Configuration ===")
        print("  1. Save current settings + zones as manufacturer default")
        print(f"  2. Restore manufacturer default {'(zones available)' if has_zones else '(no zones archive yet)'}")
        print("  3. Clear user setting overrides only (keep zones)")
        print("  4. Show config file paths")
        print("  5. Back")
        choice = input("\nEnter your choice (1-5): ").strip()
        if choice == "1":
            confirm = input("Promote current state to manufacturer default? "
                            "[y/N]: ").strip().lower()
            if confirm == "y":
                cfg.save_as_manufacturer_default(include_zones=True)
                print(f"Saved manufacturer default to {cfg.manufacturer_path}")
                if cfg.has_manufacturer_zones():
                    print(f"Saved manufacturer zones to {cfg.manufacturer_zones_path}")
            else:
                print("Cancelled.")
        elif choice == "2":
            confirm = input("Drop user overrides and restore manufacturer "
                            "default? [y/N]: ").strip().lower()
            if confirm == "y":
                cfg.restore_manufacturer_default(include_zones=True)
                print("Restored. Restart the multi-camera feed for changes "
                      "to take effect.")
            else:
                print("Cancelled.")
        elif choice == "3":
            confirm = input("Clear user settings overrides? [y/N]: "
                            ).strip().lower()
            if confirm == "y":
                cfg.clear_user_overrides()
                print("User overrides cleared.")
            else:
                print("Cancelled.")
        elif choice == "4":
            print(f"  manufacturer settings: {cfg.manufacturer_path}")
            print(f"  manufacturer zones   : {cfg.manufacturer_zones_path} "
                  f"({'present' if cfg.has_manufacturer_zones() else 'absent'})")
            print(f"  user settings (delta): {cfg.user_path}")
            print(f"  zones (slots/poly)   : {cfg.zones_path}")
        elif choice == "5":
            return
        else:
            print("Invalid choice.")


def setup_zones_menu() -> None:
    """Top-level slot picker for zone setup."""
    cfg = get_config()
    max_streams = int(cfg.get("streams.max_streams", 4))

    while True:
        from hotplug import list_cameras
        connected = {c.port_path: c for c in list_cameras()}

        print("\n=== Setup alarm zones ===")
        any_bound = False
        for sid in range(max_streams):
            slot_cfg = cfg.slot(sid) or {}
            pp = slot_cfg.get("port_path", "")
            poly = slot_cfg.get("polygon") or []
            cam = connected.get(pp)
            state = "(no camera bound)" if not pp else (
                f"{cam.model or 'connected'} -> {cam.devnode}"
                if cam else "BOUND but UNAVAILABLE (plug in)")
            poly_state = f"{len(poly)} verts" if poly else "no polygon"
            print(f"  {sid}.  {state:42}  [{poly_state}]")
            if pp:
                any_bound = True
        if not any_bound:
            print("  -- no slots bound yet. Run option 2 (Multi-camera "
                  "feed) once so cameras get assigned to slots. --")
        print(f"  {max_streams}.  Back")

        choice = input(f"\nSelect slot to edit (0-{max_streams - 1}, "
                       f"{max_streams}=back): ").strip()
        if not choice.isdigit():
            print("Invalid choice.")
            continue
        n = int(choice)
        if n == max_streams:
            return
        if 0 <= n < max_streams:
            setup_zone_for_slot(n)
        else:
            print("Invalid choice.")


def interactive_menu(source: int | str, output: str) -> None:
    print("=== eyeTool - Camera Application ===")
    print(f"Camera source: {source}")
    print(f"Display target: {os.environ.get('DISPLAY', '(not set)')}")
    det_status = "ON" if _detection_enabled else "OFF"
    print(" 1. Live camera feed (single)")
    print(" 2. Multi-camera feed (2x2 grid)")
    print(" 3. Setup alarm zones")
    print(" 4. Image preprocessing")
    print(" 5. Monitoring TUI")
    print(" 6. Configuration (save/restore default)")
    print(" 7. Capture single image")
    print(" 8. Probe camera (no GUI)")
    print(" 9. Select display target")
    print(f"10. Detection settings [{det_status}]")
    print("11. Exit")

    while True:
        choice = input("\nEnter your choice (1-11): ").strip()
        if choice == "1":
            load_camera_feed(source)
        elif choice == "2":
            load_multi_camera_feed()
        elif choice == "3":
            setup_zones_menu()
        elif choice == "4":
            preprocess_settings_menu()
        elif choice == "5":
            from monitor import run as run_monitor
            run_monitor()
        elif choice == "6":
            configuration_menu()
        elif choice == "7":
            capture_single_image(source, output)
        elif choice == "8":
            probe_camera(source)
        elif choice == "9":
            select_display_menu()
        elif choice == "10":
            detection_settings_menu()
        elif choice == "11":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1-11.")


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
