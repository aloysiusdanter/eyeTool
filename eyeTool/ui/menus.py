"""Feature functions and interactive menus for eyeTool.

Contains all camera feed, zone setup, preprocessing, detection settings,
configuration, and interactive menu functions extracted from main.py.
"""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import threading
import time

import cv2
import numpy as np

from core.camera import open_camera
from core.config import get_config
from core.display import (check_display, create_fullscreen_window,
                          select_display_menu, set_fullscreen)
from detection.rknn_yolov8 import infer as rknn_infer
from detection.pipeline import (Detector, FrameSource, MultiDetector,
                       overlay_detections, overlay_tile_detections)
from streams.stream import StreamManager
from streams.compositor import GridCompositor
from core.hotplug import HotplugMonitor
from core.zones import load_zones
from preprocessing.preprocess import Preprocess
from utils.external_logging import open_external_log_for_subprocess
import shutil

# ── Shared mutable state ──────────────────────────────────────────────
# These are module-level so that all menu/feature functions share the
# same runtime settings and the TUI can read them for display.

_quit_requested = False

detection_enabled = True
detection_confidence = 0.5
target_fps = 30
detect_every_n = 1        # run NPU on every Nth captured frame
use_multi_core = False    # use 3 NPU cores (round-robin)
low_latency_profile = False
max_box_age_ms = 120
zone_fill = True

PERSON_CLASS_ID = 0       # COCO class index for "person"


def _load_detection_settings() -> None:
    """Load detection settings from config into module-level globals."""
    global detection_enabled, detection_confidence, target_fps
    global detect_every_n, use_multi_core
    global low_latency_profile, max_box_age_ms, zone_fill
    cfg = get_config()
    detection_enabled = bool(cfg.get("detection.enabled", True))
    detection_confidence = float(cfg.get("detection.confidence", 0.5))
    target_fps = int(cfg.get("detection.target_fps", 30))
    detect_every_n = int(cfg.get("detection.detect_every_n", 1))
    use_multi_core = bool(cfg.get("detection.use_multi_core", False))
    low_latency_profile = bool(cfg.get("detection.low_latency_profile", False))
    max_box_age_ms = int(cfg.get("detection.max_box_age_ms", 120))
    max_box_age_ms = max(0, min(max_box_age_ms, 2000))
    zone_fill = bool(cfg.get("detection.zone_fill", True))


def _save_detection_settings() -> None:
    """Save current detection settings to user_settings.json."""
    global detection_enabled, detection_confidence, target_fps
    global detect_every_n, use_multi_core
    global low_latency_profile, max_box_age_ms, zone_fill
    cfg = get_config()
    cfg.set("detection.enabled", detection_enabled)
    cfg.set("detection.confidence", detection_confidence)
    cfg.set("detection.target_fps", target_fps)
    cfg.set("detection.detect_every_n", detect_every_n)
    cfg.set("detection.use_multi_core", use_multi_core)
    cfg.set("detection.low_latency_profile", low_latency_profile)
    cfg.set("detection.max_box_age_ms", max_box_age_ms)
    cfg.set("detection.zone_fill", zone_fill)
    cfg.save_user()


def _signal_handler(sig, frame):  # noqa: ARG001
    global _quit_requested
    _quit_requested = True


# ── Helper ────────────────────────────────────────────────────────────

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


# ── Camera feed functions ─────────────────────────────────────────────

def load_camera_feed(source: int | str) -> None:
    """Async pipeline: capture thread + detector thread + display loop.

    Press 'q'/ESC on the window or Ctrl+C on the console to quit.
    """
    if not check_display():
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

    # Get display resolution and scale accordingly
    from core.camera import get_display_resolution
    monitor_w, monitor_h = get_display_resolution()
    # Scale to display size (use monitor resolution for best quality)
    display_w, display_h = monitor_w, monitor_h
    det_status = "ON" if detection_enabled else "OFF"
    print(f"Feed info: {w}x{h} @ {fps:.1f} fps (letterboxed to {display_w}x{display_h})  |  source: {source}  |  display: {os.environ.get('DISPLAY', '?')}")
    multi_status = "3-CORE" if use_multi_core else "1-CORE"
    print(f"Detection: {det_status}  |  confidence: {detection_confidence}  |  target FPS: {target_fps}  |  detect every N: {detect_every_n}  |  NPU: {multi_status}")
    if low_latency_profile:
        print(f"Low-latency profile: ON  |  stale box cutoff: {max_box_age_ms} ms")
    print("Press 'q'/ESC on the window, or Ctrl+C here to quit.")

    frame_source = FrameSource(cap)
    detector: Detector | None = None
    frame_source.start()
    if detection_enabled:
        detector = Detector(frame_source,
                            conf_thres=detection_confidence,
                            detect_every_n=detect_every_n,
                            use_multi_core=use_multi_core)
        detector.start()

    create_fullscreen_window("eyeTool - Camera Feed")
    first_frame = True
    frame_interval = 1.0 / target_fps if target_fps > 0 else 0
    last_present_ts = time.monotonic()
    actual_fps = 0.0
    last_stats_ts = time.monotonic()
    max_box_age_s: float | None = None
    if low_latency_profile and detection_enabled:
        max_box_age_s = max_box_age_ms / 1000.0
    try:
        while not _quit_requested:
            loop_start = time.monotonic()
            # Aggressive: poll instead of wait to avoid blocking
            snap = frame_source.get_latest()
            if snap is None:
                # No frame yet, minimal sleep to avoid busy-spin
                time.sleep(0.001)
                continue
            frame, frame_ts = snap
            # Aggressive: skip frame copy for maximum speed (risky but fast)
            # frame = frame.copy()

            now = loop_start
            if detector is not None:
                det = detector.get_latest()
                overlay_detections(frame, det, now, max_age_s=max_box_age_s)

            # Aggressive: skip letterboxing if frame already at target size
            if frame.shape[1] == display_w and frame.shape[0] == display_h:
                frame_letterboxed = frame
            else:
                frame_letterboxed = letterbox_frame(frame, display_w, display_h)

            present_elapsed = now - last_present_ts
            if present_elapsed > 0:
                actual_fps = 1.0 / present_elapsed
            last_present_ts = now

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
            try:
                set_fullscreen("eyeTool - Camera Feed")
            except cv2.error:
                pass
            first_frame = False

            work_time = time.monotonic() - loop_start
            wait_s = frame_interval - work_time
            wait_ms = 1 if frame_interval <= 0 else max(1, int(wait_s * 1000))
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
    if not check_display():
        return
    cap = open_camera(source)
    if cap is None:
        return

    global _quit_requested
    _quit_requested = False
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    # Unblock signals in case curses blocked them
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})

    print(f"Press SPACE on the OpenCV window to capture image to '{output}', 'q' or ESC to quit, or Ctrl+C in this console.")
    try:
        while not _quit_requested:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            from core.camera import get_display_resolution
            display_w, display_h = get_display_resolution()
            frame_display = letterbox_frame(frame, display_w, display_h)
            cv2.imshow("eyeTool - Capture Mode", frame_display)
            set_fullscreen("eyeTool - Capture Mode")
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                if cv2.imwrite(output, frame):
                    print(f"Image saved as {output}")
                else:
                    print(f"Error: Failed to write image to '{output}'.")
                break
            if key in (ord("q"), ord("Q"), 27):  # 27 = ESC
                break
            if cv2.getWindowProperty("eyeTool - Capture Mode",
                                     cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ── Recording storage management ────────────────────────────────────────

def _get_disk_usage(path: str) -> float:
    """Return disk usage percentage for the filesystem containing *path*.
    
    Returns 0.0 if unable to determine usage.
    """
    try:
        stat = shutil.disk_usage(path)
        return (stat.used / stat.total) * 100.0
    except (OSError, ZeroDivisionError):
        return 0.0


def _cleanup_old_files(save_dir: str, threshold_percent: float) -> int:
    """Delete oldest recording files in *save_dir* until disk usage is below threshold.
    
    Returns number of files deleted.
    """
    deleted_count = 0
    try:
        files = []
        for f in os.listdir(save_dir):
            if f.endswith((".ts", ".mp4")):
                full_path = os.path.join(save_dir, f)
                try:
                    mtime = os.path.getmtime(full_path)
                    files.append((full_path, mtime))
                except OSError:
                    continue
        
        # Sort by modification time (oldest first)
        files.sort(key=lambda x: x[1])
        
        # Delete oldest files until below threshold
        for file_path, _ in files:
            usage = _get_disk_usage(save_dir)
            if usage < threshold_percent:
                break
            try:
                os.remove(file_path)
                deleted_count += 1
                print(f"[storage] deleted old file: {file_path}")
            except OSError as e:
                print(f"[storage] failed to delete {file_path}: {e}")
        
        if deleted_count > 0:
            usage_after = _get_disk_usage(save_dir)
            print(f"[storage] deleted {deleted_count} file(s), disk usage now {usage_after:.1f}%")
    
    except OSError as e:
        print(f"[storage] cleanup failed: {e}")
    
    return deleted_count


# ── Hardware-accelerated video writer (Rockchip MPP) ────────────────────

class _MppVideoWriter:
    """Write video frames using Rockchip MPP H.264 hardware encoder via GStreamer.

    Accepts BGR numpy frames (same as cv2.VideoWriter) and pipes raw video
    into a gst-launch-1.0 subprocess that uses mpph264enc for encoding.
    A background writer thread prevents blocking the display loop.
    """

    def __init__(self, output_path: str, fps: float, width: int, height: int):
        self._path = output_path
        self._fps = fps
        self._w = width
        self._h = height
        self._proc: subprocess.Popen | None = None
        self._opened = False
        self._frame_size = width * height * 3  # BGR
        self._frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._writer_thread: threading.Thread | None = None

        pipeline = (
            f"fdsrc fd=0 ! "
            f"rawvideoparse width={width} height={height} "
            f"format=bgr framerate={int(fps)}/1 ! "
            f"videoconvert ! video/x-raw,format=NV12 ! "
            f"mpph264enc ! h264parse config-interval=1 ! "
            f"mpegtsmux ! filesink location={output_path} sync=false"
        )
        try:
            self._stderr_log = open_external_log_for_subprocess("gstreamer")
            stderr_target = self._stderr_log.__enter__()
            self._proc = subprocess.Popen(
                ["gst-launch-1.0", "-e"] + pipeline.split(),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=stderr_target,
            )
            self._opened = True
            self._writer_thread = threading.Thread(
                target=self._writer_loop, name="MppWriter", daemon=True)
            self._writer_thread.start()
        except (FileNotFoundError, OSError) as e:
            if hasattr(self, "_stderr_log"):
                self._stderr_log.__exit__(None, None, None)
            print(f"[mpp] failed to launch GStreamer pipeline: {e}")
            self._opened = False

    def _writer_loop(self) -> None:
        """Background thread that drains pending frames into the pipe."""
        while not self._stop_event.is_set():
            try:
                frame = self._frame_queue.get(timeout=0.02)
            except queue.Empty:
                continue
            if frame is None:
                continue
            try:
                self._proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError):
                self._opened = False
                return

    def isOpened(self) -> bool:
        return self._opened and self._proc is not None and self._proc.poll() is None

    def write(self, frame: np.ndarray) -> None:
        if not self.isOpened():
            return
        if self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass

    def release(self) -> None:
        self._stop_event.set()
        last_frame = None
        while True:
            try:
                last_frame = self._frame_queue.get_nowait()
            except queue.Empty:
                break
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=2)
        # Drain latest queued frame, if any.
        if self._proc is not None and last_frame is not None:
            try:
                self._proc.stdin.write(last_frame.tobytes())
            except (BrokenPipeError, OSError):
                pass
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._proc = None
        if hasattr(self, "_stderr_log"):
            self._stderr_log.__exit__(None, None, None)
        self._opened = False


def _check_mpp_available() -> bool:
    """Return True if Rockchip MPP GStreamer encoder is available."""
    try:
        result = subprocess.run(
            ["gst-inspect-1.0", "mpph264enc"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _create_video_writer(
    output_path: str, codec: str, fps: float, width: int, height: int
) -> cv2.VideoWriter | _MppVideoWriter:
    """Create the best available video writer.

    If codec is 'mpp_h264' and hardware encoder is available, uses MPP.
    Otherwise falls back to cv2.VideoWriter with the given fourcc codec.
    """
    if codec == "mpp_h264":
        if _check_mpp_available():
            writer = _MppVideoWriter(output_path, fps, width, height)
            if writer.isOpened():
                print(f"[mpp] hardware H.264 encoder active")
                return writer
            print("[mpp] failed to open hardware H.264 encoder")
        else:
            print("[mpp] mpph264enc not available")
        return cv2.VideoWriter()

    fourcc = cv2.VideoWriter_fourcc(*codec)
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def _recording_extension(codec: str) -> str:
    if codec == "mpp_h264":
        return ".ts"
    return ".mp4"


# ── Camera feed functions ─────────────────────────────────────────────

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


def load_multi_camera_feed() -> None:
    """Multi-stream 2x2 grid display (Phase A foundation; no detection yet).

    Opens every currently-connected USB UVC camera into a slot determined
    by its kernel-stable USB port path. Press 'q'/ESC on the window or
    Ctrl+C on the console to quit.
    """
    if not check_display():
        return

    cfg = get_config()
    max_streams = int(cfg.get("streams.max_streams", 4))
    # Use actual display resolution instead of fixed config
    from core.camera import get_display_resolution
    display_w, display_h = get_display_resolution()
    _target_fps = int(cfg.get("display.target_fps", 30))
    watchdog = float(cfg.get("streams.watchdog_stall_s", 2.0))

    # Load camera mapping from config (device path -> position)
    camera_mapping = {
        0: cfg.get("camera_mapping.top_left", None),
        1: cfg.get("camera_mapping.top_right", None),
        2: cfg.get("camera_mapping.bottom_left", None),
        3: cfg.get("camera_mapping.bottom_right", None),
    }

    # Restore saved port_path -> slot_id bindings from zones.json
    saved_bindings: dict[str, int] = {}
    for sid, slot_cfg in cfg.all_slots().items():
        pp = slot_cfg.get("port_path")
        if pp:
            saved_bindings[pp] = sid

    # Apply camera mapping if configured
    # We need to convert device paths to port paths by enumerating cameras
    from core.hotplug import list_cameras
    cameras = list_cameras()
    devnode_to_portpath = {cam.devnode: cam.port_path for cam in cameras}
    
    for slot_id, device_path in camera_mapping.items():
        if device_path and device_path in devnode_to_portpath:
            port_path = devnode_to_portpath[device_path]
            saved_bindings[port_path] = slot_id
            print(f"[camera mapping] slot {slot_id} <- {device_path} (port: {port_path})")

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
    if detection_enabled:
        detector = MultiDetector(manager,
                                 conf_thres=detection_confidence,
                                 detect_every_n=detect_every_n,
                                 use_multi_core=use_multi_core)
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
    max_box_age_s: float | None = None
    if low_latency_profile and detection_enabled:
        max_box_age_s = max_box_age_ms / 1000.0
    def _tile_overlay(slot_id, tile, scale, off_x, off_y, snap):
        zone = zones.get(slot_id)
        if zone is not None:
            zone.draw_on_tile(tile, scale, off_x, off_y, draw_fill=zone_fill)
        if detector is None:
            return
        det = detector.get_result(slot_id)
        overlay_tile_detections(tile, det, scale, off_x, off_y,
                                _now_ts_ref[0], zone=zone,
                                max_age_s=max_box_age_s)
    compositor.set_overlay(_tile_overlay)

    global _quit_requested
    _quit_requested = False
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Setup recording from config
    recording_enabled = cfg.get("recording.enabled", True)
    save_dir = cfg.get("recording.save_dir", "/media/pi/6333-3864")
    fallback_dir = cfg.get("recording.fallback_dir", "~/Videos")
    segment_duration = int(cfg.get("recording.segment_duration_min", 2))
    codec = cfg.get("recording.codec", "mpp_h264")
    storage_threshold = float(cfg.get("recording.storage_threshold_percent", 20))
    
    try:
        os.makedirs(save_dir, exist_ok=True)
    except (OSError, PermissionError):
        save_dir = os.path.expanduser(fallback_dir)
        os.makedirs(save_dir, exist_ok=True)
        print(f"SD card not available, saving to {save_dir}")
    
    fps = 25.0

    # Track segment start time for 2-minute segments
    _segment_start_time = time.monotonic()
    ts = time.strftime("%Y%m%d_%H%M%S")
    ext = _recording_extension(codec)
    output_path = os.path.join(save_dir, f"multifeed_{ts}{ext}")
    writer = _create_video_writer(output_path, codec, fps, display_w, display_h)
    frames_written = 0
    recording = recording_enabled and writer.isOpened()

    if recording:
        print(f"Recording to: {output_path} (segments: {segment_duration} min, codec: {codec})")
    elif recording_enabled:
        print("Warning: Could not start video writer")
    else:
        print("Recording disabled in configuration")

    det_status = "ON" if detection_enabled else "OFF"
    multi_status = "3-CORE" if use_multi_core else "1-CORE"

    print(f"Multi-camera feed: {display_w}x{display_h}, target {_target_fps} FPS, {max_streams} slots")
    print(f"Detection: {det_status}  |  conf: {detection_confidence}  |  every-N: {detect_every_n}  |  NPU: {multi_status}")
    if low_latency_profile:
        fill_status = "ON" if zone_fill else "OFF"
        print(f"Low-latency profile: ON  |  stale box cutoff: {max_box_age_ms} ms  |  zone fill: {fill_status}")
    print("Press 'q'/ESC on the window, or Ctrl+C here to quit.")

    create_fullscreen_window("eyeTool - Multi Feed")
    first_frame = True
    last_stats_ts = time.monotonic()
    frame_interval = 1.0 / _target_fps if _target_fps > 0 else 0.0
    last_present_ts = time.monotonic()
    actual_fps = 0.0
    next_fullscreen_ts = 0.0
    next_window_check_ts = 0.0
    next_segment_check_ts = 0.0

    loop_error_count = 0
    max_loop_errors = 10
    last_error_log_time = 0
    start_time = time.monotonic()
    
    try:
        while not _quit_requested:
            try:
                loop_start = time.monotonic()
                now = loop_start
                _now_ts_ref[0] = now  # tile overlay reads this for staleness colour
                
                try:
                    snapshots = manager.snapshot()
                except Exception as e:
                    print(f"[ERROR] manager.snapshot() failed: {e}")
                    time.sleep(0.1)
                    continue
                
                try:
                    canvas = compositor.render(snapshots)
                except Exception as e:
                    print(f"[ERROR] compositor.render() failed: {e}")
                    time.sleep(0.1)
                    continue

                # Add timestamp overlay for recording
                stamp = time.strftime("%Y-%m-%d  %H:%M:%S")
                cv2.putText(canvas, stamp,
                            (10, display_h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2, cv2.LINE_AA)

                present_elapsed = now - last_present_ts
                if present_elapsed > 0:
                    actual_fps = 1.0 / present_elapsed
                last_present_ts = now

                cv2.putText(canvas, f"{actual_fps:4.1f} FPS",
                            (display_w - 90, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
                            cv2.LINE_AA)

                # Check for segment split less frequently to keep display
                # loop lightweight.
                if now >= next_segment_check_ts:
                    next_segment_check_ts = now + 0.5
                    segment_elapsed = now - _segment_start_time
                else:
                    segment_elapsed = 0.0
                if segment_elapsed >= (segment_duration * 60.0):
                    print(f"Segment split ({segment_duration} min): closing current file...")
                    try:
                        if writer is not None:
                            writer.release()
                    except Exception as e:
                        print(f"[ERROR] Failed to release writer: {e}")

                    ts = time.strftime("%Y%m%d_%H%M%S")
                    new_path = os.path.join(save_dir, f"multifeed_{ts}{ext}")
                    print(f"Segment split: opening new file -> {new_path}")
                    try:
                        new_writer = _create_video_writer(new_path, codec, fps, display_w, display_h)
                        if new_writer.isOpened():
                            writer = new_writer
                            _segment_start_time = now
                            output_path = new_path
                            frames_written = 0
                            print(f"Segment split: success")
                            threading.Thread(
                                target=_cleanup_old_files,
                                args=(save_dir, storage_threshold),
                                daemon=True,
                            ).start()
                        else:
                            print("Error: could not open new file after segment split.")
                            recording = False
                            writer = None
                    except Exception as e:
                        print(f"[ERROR] Failed to open new video file: {e}")
                        recording = False
                        writer = None

                # Write frame if recording
                if recording and writer is not None:
                    try:
                        writer.write(canvas)
                        frames_written += 1
                    except Exception as e:
                        print(f"[ERROR] Failed to write frame: {e}")
                        recording = False
                        writer = None

                if now - last_stats_ts > 2.0:
                    # Print compact two-row format
                    slot_info = []
                    for snap in snapshots:
                        status = snap.state.value
                        fps_str = f"{snap.capture_fps:5.1f}Hz"
                        slot_info.append(f"slot{snap.slot_id}-{status} {fps_str}")
                    
                    slot_line = "  __  ".join(slot_info)
                    print(f"\n{slot_line}")
                    
                    det_fps = detector.fps() if detector else 0.0
                    elapsed_time = now - start_time
                    print(f"Detect FPS {det_fps:5.1f}Hz; Display FPS {actual_fps:5.1f}Hz; Recorded {frames_written} frames ({elapsed_time:.0f}s)")
                    
                    last_stats_ts = now

                cv2.imshow("eyeTool - Multi Feed", canvas)
                if first_frame or now >= next_fullscreen_ts:
                    set_fullscreen("eyeTool - Multi Feed")
                    next_fullscreen_ts = now + 1.0
                first_frame = False

                # Frame pacing: use work time from this iteration.
                work_time = time.monotonic() - loop_start
                wait_s = frame_interval - work_time
                wait_ms = 1 if frame_interval <= 0 else max(1, int(wait_s * 1000))
                key = cv2.waitKey(wait_ms) & 0xFF
                # Check quit flag after waitKey in case signal fired during wait
                if _quit_requested:
                    print("Quit requested via signal")
                    break
                if key in (ord("q"), ord("Q"), 27):
                    print("User requested quit via key press")
                    break
                
                # Throttle window visibility checks to avoid X11 overhead.
                if now >= next_window_check_ts:
                    next_window_check_ts = now + 1.0
                    try:
                        window_visible = cv2.getWindowProperty("eyeTool - Multi Feed", cv2.WND_PROP_VISIBLE)
                        if window_visible < 0.5:  # More lenient threshold
                            print(f"Window visibility check failed: {window_visible}, attempting to recreate window")
                            create_fullscreen_window("eyeTool - Multi Feed")
                            next_fullscreen_ts = now
                    except Exception as e:
                        print(f"[ERROR] Window visibility check failed: {e}")
                
                # Reset error count on successful iteration
                loop_error_count = 0
                
            except KeyboardInterrupt:
                print("Keyboard interrupt received")
                break
            except Exception as e:
                loop_error_count += 1
                current_time = time.time()
                # Log errors but don't spam - only log every 5 seconds
                if current_time - last_error_log_time > 5.0:
                    print(f"[ERROR] Loop iteration {loop_error_count}/{max_loop_errors} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    last_error_log_time = current_time
                
                # If too many consecutive errors, exit to prevent infinite error loop
                if loop_error_count >= max_loop_errors:
                    print(f"[CRITICAL] Too many consecutive errors ({max_loop_errors}), exiting")
                    break
                
                # Sleep briefly before retrying
                time.sleep(0.1)
                continue
    finally:
        if writer is not None:
            writer.release()
            print(f"Recording closed. {frames_written} frames written to '{output_path}'.")
        monitor.stop()
        if detector is not None:
            detector.stop()
        manager.stop_all()
        cv2.destroyAllWindows()
        print("Multi-camera feed closed.")


def record_camera_feed(source: int | str, output: str = "") -> None:
    """Record camera feed to a video file.

    If *output* is empty a timestamped filename is generated automatically.
    Press 'r' in the window to start/stop recording, 'q'/ESC to quit.
    """
    if not check_display():
        return
    cap = open_camera(source)
    if cap is None:
        return

    cfg = get_config()
    codec = cfg.get("recording.codec", "mpp_h264")
    ext = _recording_extension(codec)
    save_dir = os.path.dirname(os.path.abspath(output)) if output else cfg.get("recording.save_dir", "/media/pi/6333-3864")
    if not output:
        try:
            os.makedirs(save_dir, exist_ok=True)
        except (OSError, PermissionError):
            save_dir = os.path.expanduser(cfg.get("recording.fallback_dir", "~/Videos"))
            os.makedirs(save_dir, exist_ok=True)
            print(f"SD card not available, saving to {save_dir}")
        ts = time.strftime("%Y%m%d_%H%M%S")
        output = os.path.join(save_dir, f"recording_{ts}{ext}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    global _quit_requested
    _quit_requested = False
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    print(f"Record feed: {w}x{h} @ {fps:.1f} fps  |  source: {source}")
    print(f"Output file: {output}")
    print("Type 'r' + Enter to start/stop recording, 'q' + Enter to quit.")

    frame_source = FrameSource(cap)
    frame_source.start()

    writer: cv2.VideoWriter | _MppVideoWriter | None = None
    recording = False
    frames_written = 0
    _stop = threading.Event()
    _current_minute = time.localtime().tm_min
    _base_output = output if output else os.path.join(save_dir, f"recording_000000_000000{ext}")

    def _input_loop():
        """Run in a background thread: read terminal commands."""
        nonlocal recording, writer, frames_written, _current_minute, _base_output
        while not _stop.is_set():
            try:
                cmd = input(">>> ").strip().lower()
            except EOFError:
                break
            if cmd == "r":
                if not recording:
                    w_ = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h_ = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    _base_output = os.path.join(save_dir, f"recording_{ts}{ext}")
                    wr = _create_video_writer(_base_output, codec, fps, w_, h_)
                    if not wr.isOpened():
                        print(f"Error: could not open '{_base_output}' for writing.")
                    else:
                        writer = wr
                        recording = True
                        frames_written = 0
                        _current_minute = time.localtime().tm_min
                        print(f"Recording started -> {_base_output}")
                else:
                    recording = False
                    if writer is not None:
                        writer.release()
                        writer = None
                    print(f"Recording stopped. {frames_written} frames written to '{_base_output}'.")
            elif cmd in ("q", "quit"):
                _stop.set()
                break

    input_thread = threading.Thread(target=_input_loop, daemon=True)
    input_thread.start()

    try:
        while not _stop.is_set() and not _quit_requested:
            if not frame_source.wait_new(timeout=1.0):
                continue
            snap = frame_source.get_latest()
            if snap is None:
                continue
            frame, _ = snap
            frame = frame.copy()

            if recording and writer is not None:
                stamp = time.strftime("%Y-%m-%d  %H:%M:%S")
                cv2.putText(frame, stamp,
                            (10, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2, cv2.LINE_AA)
                writer.write(frame)
                frames_written += 1

                current_minute = time.localtime().tm_min
                if current_minute != _current_minute:
                    writer.release()
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    _base_output = os.path.join(save_dir, f"recording_{ts}{ext}")
                    w_ = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h_ = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    wr = _create_video_writer(_base_output, codec, fps, w_, h_)
                    if wr.isOpened():
                        writer = wr
                        _current_minute = current_minute
                        print(f"Minute split: new file -> {_base_output}")
                    else:
                        print(f"Error: could not open new file after minute split.")
                        recording = False
                        writer = None
    finally:
        _stop.set()
        if writer is not None:
            writer.release()
            if recording:
                print(f"Recording stopped (on exit). {frames_written} frames written to '{output}'.")
        frame_source.stop()
        cap.release()
        print("Record feed closed.")


def record_multi_camera_feed() -> None:
    """Record each camera slot independently.

    Type 'r0'/'r1'/... + Enter to start/stop recording for that slot.
    Type 'q' + Enter to quit. Recordings saved to ~/Videos/.
    """
    cfg = get_config()
    max_streams = int(cfg.get("streams.max_streams", 4))
    watchdog = float(cfg.get("streams.watchdog_stall_s", 2.0))

    saved_bindings: dict[str, int] = {}
    for sid, slot_cfg in cfg.all_slots().items():
        pp = slot_cfg.get("port_path")
        if pp:
            saved_bindings[pp] = sid

    _persist_lock = threading.Lock()
    def _persist_binding(port_path: str, slot_id: int) -> None:
        with _persist_lock:
            slot_cfg = cfg.slot(slot_id) or {}
            slot_cfg["port_path"] = port_path
            cfg.update_slot(slot_id, slot_cfg)
            cfg.save_zones()

    saved_preprocess: dict[int, Preprocess] = {}
    for sid, slot_cfg in cfg.all_slots().items():
        pre_cfg = slot_cfg.get("preprocessing")
        if pre_cfg:
            try:
                pp = Preprocess.from_dict(pre_cfg)
                if not pp.is_identity():
                    saved_preprocess[sid] = pp
            except (TypeError, ValueError):
                pass

    manager = StreamManager(max_streams=max_streams,
                            watchdog_stall_s=watchdog,
                            saved_bindings=saved_bindings,
                            on_binding_change=_persist_binding,
                            saved_preprocess=saved_preprocess)
    manager.open_all_present()

    codec = cfg.get("recording.codec", "mpp_h264")
    ext = _recording_extension(codec)
    save_dir = cfg.get("recording.save_dir", "/media/pi/6333-3864")
    try:
        os.makedirs(save_dir, exist_ok=True)
    except (OSError, PermissionError):
        save_dir = os.path.expanduser(cfg.get("recording.fallback_dir", "~/Videos"))
        os.makedirs(save_dir, exist_ok=True)
        print(f"SD card not available, saving to {save_dir}")

    writers: dict[int, cv2.VideoWriter | _MppVideoWriter] = {}
    frames_written: dict[int, int] = {}
    recording: dict[int, bool] = {}
    output_paths: dict[int, str] = {}
    _stop = threading.Event()
    _current_minutes: dict[int, int] = {}

    global _quit_requested
    _quit_requested = False
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    print(f"Multi-camera record: {max_streams} slots")
    print("Auto-starting recording for all active slots...")
    print("Commands: r0, r1, r2, r3 = toggle record for that slot | q = quit")

    # Auto-start recording for all active slots
    active = manager.active_slots()
    print(f"Active slots detected: {active}")
    for sid in active:
        # Retry a few times to get the first frame (FrameSource needs warmup)
        snap = None
        for _ in range(10):
            snap = manager.get_frame(sid)
            if snap is not None:
                break
            time.sleep(0.1)
        if snap is not None:
            frame, _ = snap
            h_, w_ = frame.shape[:2]
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(save_dir, f"slot{sid}_{ts}{ext}")
            fps_ = 25.0
            wr = _create_video_writer(path, codec, fps_, w_, h_)
            if wr.isOpened():
                writers[sid] = wr
                frames_written[sid] = 0
                output_paths[sid] = path
                _current_minutes[sid] = time.localtime().tm_min
                recording[sid] = True
                print(f"Slot {sid}: auto-started recording -> {path}")
            else:
                print(f"Slot {sid}: could not auto-start recording.")
        else:
            print(f"Slot {sid}: no frame available for auto-start (camera may be slow).")

    def _input_loop():
        nonlocal recording, writers, frames_written, output_paths, _current_minutes
        while not _stop.is_set():
            try:
                cmd = input(">>> ").strip().lower()
            except EOFError:
                time.sleep(0.1)
                continue
            except KeyboardInterrupt:
                break
            if cmd in ("q", "quit"):
                _stop.set()
                break
            if len(cmd) == 2 and cmd[0] == "r" and cmd[1].isdigit():
                sid = int(cmd[1])
                if sid >= max_streams:
                    print(f"Slot {sid} does not exist (max {max_streams - 1}).")
                    continue
                snap = manager.get_frame(sid)
                if snap is None:
                    print(f"Slot {sid}: no active camera.")
                    continue
                if not recording.get(sid, False):
                    frame, _ = snap
                    h_, w_ = frame.shape[:2]
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    path = os.path.join(save_dir, f"slot{sid}_{ts}{ext}")
                    fps_ = 25.0
                    wr = _create_video_writer(path, codec, fps_, w_, h_)
                    if not wr.isOpened():
                        print(f"Slot {sid}: could not open '{path}' for writing.")
                    else:
                        writers[sid] = wr
                        frames_written[sid] = 0
                        output_paths[sid] = path
                        _current_minutes[sid] = time.localtime().tm_min
                        recording[sid] = True
                        print(f"Slot {sid}: recording started -> {path}")
                else:
                    recording[sid] = False
                    wr = writers.pop(sid, None)
                    if wr is not None:
                        wr.release()
                    print(f"Slot {sid}: recording stopped. "
                          f"{frames_written.get(sid, 0)} frames written to '{output_paths.get(sid, '')}'.")
            else:
                print("Unknown command. Use r0/r1/r2/r3 or q.")

    input_thread = threading.Thread(target=_input_loop, daemon=True)
    input_thread.start()

    try:
        while not _stop.is_set() and not _quit_requested:
            for sid, wr in list(writers.items()):
                if not recording.get(sid, False):
                    continue
                snap = manager.get_frame(sid)
                if snap is None:
                    continue
                frame, _ = snap
                frame = frame.copy()
                stamp = time.strftime("%Y-%m-%d  %H:%M:%S")
                cv2.putText(frame, stamp,
                            (10, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2, cv2.LINE_AA)

                current_minute = time.localtime().tm_min
                if current_minute != _current_minutes.get(sid, -1):
                    wr.release()
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    new_path = os.path.join(save_dir, f"slot{sid}_{ts}{ext}")
                    h_, w_ = frame.shape[:2]
                    new_wr = _create_video_writer(new_path, codec, 25.0, w_, h_)
                    if new_wr.isOpened():
                        writers[sid] = new_wr
                        _current_minutes[sid] = current_minute
                        output_paths[sid] = new_path
                        print(f"Slot {sid}: minute split -> {new_path}")
                    else:
                        print(f"Slot {sid}: failed to open new file after minute split.")
                        recording[sid] = False
                        del writers[sid]
                    continue

                wr.write(frame)
                frames_written[sid] = frames_written.get(sid, 0) + 1
            time.sleep(0.04)
    finally:
        _stop.set()
        for sid, wr in writers.items():
            wr.release()
            print(f"Slot {sid}: recording closed. "
                  f"{frames_written.get(sid, 0)} frames written to '{output_paths.get(sid, '')}'.")
        manager.stop_all()
        print("Multi-camera record closed.")


# ── Zone setup ────────────────────────────────────────────────────────

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
    if not check_display():
        return False

    # Lazy imports so this menu item works even when running headless tests
    from ui.editors.polygon_editor import run as run_polygon_editor
    from core.hotplug import list_cameras

    cfg = get_config()
    slot_cfg = cfg.slot(slot_id) or {}
    
    # Load camera mapping from config (device path -> position)
    camera_mapping = {
        0: cfg.get("camera_mapping.top_left", None),
        1: cfg.get("camera_mapping.top_right", None),
        2: cfg.get("camera_mapping.bottom_left", None),
        3: cfg.get("camera_mapping.bottom_right", None),
    }
    
    # Check camera mapping first, then fall back to port_path
    device_path = camera_mapping.get(slot_id)
    port_path = slot_cfg.get("port_path")
    
    cams_by_port = {c.port_path: c for c in list_cameras()}
    cams_by_devnode = {c.devnode: c for c in list_cameras()}
    
    info = None
    if device_path and device_path in cams_by_devnode:
        info = cams_by_devnode[device_path]
    elif port_path and port_path in cams_by_port:
        info = cams_by_port[port_path]
    else:
        print(f"Slot {slot_id}: no camera bound. Use the camera mapping "
              f"or run multi-camera feed first to bind a camera to this slot.")
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
    if not check_display():
        return False
    from ui.editors.preprocess_editor import run as run_pp_editor
    from core.hotplug import list_cameras

    cfg = get_config()
    slot_cfg = cfg.slot(slot_id) or {}
    
    # Load camera mapping from config (device path -> position)
    camera_mapping = {
        0: cfg.get("camera_mapping.top_left", None),
        1: cfg.get("camera_mapping.top_right", None),
        2: cfg.get("camera_mapping.bottom_left", None),
        3: cfg.get("camera_mapping.bottom_right", None),
    }
    
    # Check camera mapping first, then fall back to port_path
    device_path = camera_mapping.get(slot_id)
    port_path = slot_cfg.get("port_path")
    
    cams_by_port = {c.port_path: c for c in list_cameras()}
    cams_by_devnode = {c.devnode: c for c in list_cameras()}
    
    info = None
    if device_path and device_path in cams_by_devnode:
        info = cams_by_devnode[device_path]
    elif port_path and port_path in cams_by_port:
        info = cams_by_port[port_path]
    else:
        print(f"Slot {slot_id}: no camera bound. Use the camera mapping "
              f"or run multi-camera feed first to bind a camera to this slot.")
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


# ── Interactive menus ─────────────────────────────────────────────────

def detection_settings_menu() -> None:
    """Interactive sub-menu for YOLO detection settings."""
    global detection_enabled, detection_confidence, target_fps
    global detect_every_n, use_multi_core
    global low_latency_profile, max_box_age_ms, zone_fill
    while True:
        det_status = "ON" if detection_enabled else "OFF"
        multi_status = "3-CORE" if use_multi_core else "1-CORE"
        ll_status = "ON" if low_latency_profile else "OFF"
        fill_status = "ON" if zone_fill else "OFF"
        print(f"\n--- Detection Settings ---")
        print(f"  1. Toggle detection [{det_status}]")
        print(f"  2. Set confidence threshold [{detection_confidence:.2f}]")
        print(f"  3. Set target display FPS [{target_fps}]")
        print(f"  4. Detect every N captured frames [{detect_every_n}]")
        print(f"  5. Use 3 NPU cores (round-robin) [{multi_status}]")
        print(f"  6. Toggle low-latency profile [{ll_status}]")
        print(f"  7. Set max detection box age ms [{max_box_age_ms}]")
        print(f"  8. Toggle zone fill overlay [{fill_status}]")
        print(f"  9. Back to main menu")
        raw = input("Enter choice (1-9): ").strip()
        if raw == "1":
            detection_enabled = not detection_enabled
            print(f"Detection {'enabled' if detection_enabled else 'disabled'}.")
        elif raw == "2":
            val = input(f"Confidence threshold (0.0-1.0) [{detection_confidence:.2f}]: ").strip()
            if val:
                try:
                    v = float(val)
                    if 0.0 <= v <= 1.0:
                        detection_confidence = v
                        print(f"Confidence threshold set to {detection_confidence:.2f}.")
                    else:
                        print("Value must be between 0.0 and 1.0.")
                except ValueError:
                    print("Invalid number.")
        elif raw == "3":
            val = input(f"Target FPS (1-120) [{target_fps}]: ").strip()
            if val:
                try:
                    v = int(val)
                    if 1 <= v <= 120:
                        target_fps = v
                        print(f"Target FPS set to {target_fps}.")
                    else:
                        print("Value must be between 1 and 120.")
                except ValueError:
                    print("Invalid number.")
        elif raw == "4":
            val = input(f"Detect every N frames (1-10) [{detect_every_n}]: ").strip()
            if val:
                try:
                    v = int(val)
                    if 1 <= v <= 10:
                        detect_every_n = v
                        print(f"Detect-every-N set to {detect_every_n}.")
                    else:
                        print("Value must be between 1 and 10.")
                except ValueError:
                    print("Invalid number.")
        elif raw == "5":
            use_multi_core = not use_multi_core
            print(f"Multi-core NPU {'enabled' if use_multi_core else 'disabled'}.")
            print("  3-CORE: round-robin across NPU_CORE_0/1/2 (~3× detection rate)")
            print("  1-CORE: single NPU_CORE_AUTO (lower power, ~10 Hz detect)")
        elif raw == "6":
            low_latency_profile = not low_latency_profile
            print(f"Low-latency profile {'enabled' if low_latency_profile else 'disabled'}.")
        elif raw == "7":
            val = input(f"Max detection box age ms (0-2000) [{max_box_age_ms}]: ").strip()
            if val:
                try:
                    v = int(val)
                    if 0 <= v <= 2000:
                        max_box_age_ms = v
                        print(f"Max detection box age set to {max_box_age_ms} ms.")
                    else:
                        print("Value must be between 0 and 2000.")
                except ValueError:
                    print("Invalid number.")
        elif raw == "8":
            zone_fill = not zone_fill
            print(f"Zone fill {'enabled' if zone_fill else 'disabled'}.")
        elif raw == "9":
            _save_detection_settings()
            print("Detection settings saved to user_settings.json")
            break
        else:
            print("Invalid choice.")


def preprocess_settings_menu() -> None:
    cfg = get_config()
    max_streams = int(cfg.get("streams.max_streams", 4))
    
    # Load camera mapping from config (device path -> position)
    camera_mapping = {
        0: cfg.get("camera_mapping.top_left", None),
        1: cfg.get("camera_mapping.top_right", None),
        2: cfg.get("camera_mapping.bottom_left", None),
        3: cfg.get("camera_mapping.bottom_right", None),
    }
    
    # Create device path -> slot mapping from camera_mapping
    devnode_to_slot = {}
    for slot_id, device_path in camera_mapping.items():
        if device_path:
            devnode_to_slot[device_path] = slot_id
    
    while True:
        from core.hotplug import list_cameras
        connected = {c.port_path: c for c in list_cameras()}
        connected_by_devnode = {c.devnode: c for c in list_cameras()}

        print("\n=== Image preprocessing ===")
        any_bound = False
        for sid in range(max_streams):
            slot_cfg = cfg.slot(sid) or {}
            pp = slot_cfg.get("port_path", "")
            cam = connected.get(pp)
            
            # Also check camera mapping for device path binding
            device_path = camera_mapping.get(sid)
            if device_path and device_path in connected_by_devnode:
                cam = connected_by_devnode[device_path]
                state = f"{cam.model or 'connected'} -> {cam.devnode}"
            elif not pp:
                state = "(no camera bound)"
            else:
                state = (f"{cam.model or 'connected'} -> {cam.devnode}"
                        if cam else "BOUND but UNAVAILABLE")
            
            pre = slot_cfg.get("preprocessing") or {}
            if not pre:
                pre_state = "default"
            else:
                flip_h = "H" if pre.get('flip_h', False) else ""
                flip_v = "V" if pre.get('flip_v', False) else ""
                rot = pre.get('rotate', 0)
                rot_str = f"{rot:+}°" if rot != 0 else ""
                transforms = " ".join(filter(None, [flip_h, flip_v, rot_str]))
                pre_state = (f"B={pre.get('brightness', 0):+.2f} "
                             f"C={pre.get('contrast', 1):.2f} "
                             f"S={pre.get('saturation', 1):.2f} "
                             f"G={pre.get('gamma', 1):.2f}")
                if transforms:
                    pre_state += f" [{transforms}]"
            print(f"  {sid}.  {state:42}  [{pre_state}]")
            if pp or device_path:
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


def recording_settings_menu() -> None:
    """Interactive sub-menu for video recording settings."""
    cfg = get_config()
    while True:
        enabled = cfg.get("recording.enabled", True)
        save_dir = cfg.get("recording.save_dir", "/media/pi/6333-3864")
        segment_duration = cfg.get("recording.segment_duration_min", 2)
        codec = cfg.get("recording.codec", "mp4v")
        storage_threshold = cfg.get("recording.storage_threshold_percent", 20)
        
        enabled_status = "ON" if enabled else "OFF"
        print(f"\n--- Recording Settings ---")
        print(f"  1. Toggle recording [{enabled_status}]")
        print(f"  2. Set save directory [{save_dir}]")
        print(f"  3. Set segment duration (minutes) [{segment_duration}]")
        print(f"  4. Set codec (mpp_h264/mp4v/mjpg/avc1) [{codec}]")
        print(f"  5. Set storage threshold (delete when disk % full) [{storage_threshold}]")
        print(f"  6. Back to main menu")
        raw = input("Enter choice (1-6): ").strip()
        if raw == "1":
            enabled = not enabled
            cfg.set("recording.enabled", enabled)
            cfg.save_user()
            print(f"Recording {'enabled' if enabled else 'disabled'}.")
        elif raw == "2":
            val = input(f"Save directory [{save_dir}]: ").strip()
            if val:
                cfg.set("recording.save_dir", val)
                cfg.save_user()
                print(f"Save directory set to {val}.")
        elif raw == "3":
            val = input(f"Segment duration in minutes (1-60) [{segment_duration}]: ").strip()
            if val:
                try:
                    v = int(val)
                    if 1 <= v <= 60:
                        cfg.set("recording.segment_duration_min", v)
                        cfg.save_user()
                        print(f"Segment duration set to {v} minutes.")
                    else:
                        print("Value must be between 1 and 60.")
                except ValueError:
                    print("Invalid number.")
        elif raw == "4":
            val = input(f"Codec (mpp_h264/mp4v/mjpg/avc1) [{codec}]: ").strip()
            if val:
                if val in ("mpp_h264", "mp4v", "mjpg", "avc1"):
                    cfg.set("recording.codec", val)
                    cfg.save_user()
                    hw = " (hardware)" if val == "mpp_h264" else " (software)"
                    print(f"Codec set to {val}{hw}.")
                else:
                    print("Invalid codec. Use mpp_h264, mp4v, mjpg, or avc1.")
        elif raw == "5":
            val = input(f"Storage threshold (delete when disk % full) (5-95) [{storage_threshold}]: ").strip()
            if val:
                try:
                    v = float(val)
                    if 5 <= v <= 95:
                        cfg.set("recording.storage_threshold_percent", v)
                        cfg.save_user()
                        print(f"Storage threshold set to {v}% (delete when disk is {v}% full).")
                    else:
                        print("Value must be between 5 and 95.")
                except ValueError:
                    print("Invalid number.")
        elif raw == "6":
            break
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
    
    # Load camera mapping from config (device path -> position)
    camera_mapping = {
        0: cfg.get("camera_mapping.top_left", None),
        1: cfg.get("camera_mapping.top_right", None),
        2: cfg.get("camera_mapping.bottom_left", None),
        3: cfg.get("camera_mapping.bottom_right", None),
    }

    while True:
        from core.hotplug import list_cameras
        connected = {c.port_path: c for c in list_cameras()}
        connected_by_devnode = {c.devnode: c for c in list_cameras()}

        print("\n=== Setup alarm zones ===")
        any_bound = False
        for sid in range(max_streams):
            slot_cfg = cfg.slot(sid) or {}
            pp = slot_cfg.get("port_path", "")
            poly = slot_cfg.get("polygon") or []
            cam = connected.get(pp)
            
            # Also check camera mapping for device path binding
            device_path = camera_mapping.get(sid)
            if device_path and device_path in connected_by_devnode:
                cam = connected_by_devnode[device_path]
                state = f"{cam.model or 'connected'} -> {cam.devnode}"
            elif not pp:
                state = "(no camera bound)"
            else:
                state = (f"{cam.model or 'connected'} -> {cam.devnode}"
                        if cam else "BOUND but UNAVAILABLE (plug in)")
            
            poly_state = f"{len(poly)} verts" if poly else "no polygon"
            print(f"  {sid}.  {state:42}  [{poly_state}]")
            if pp or device_path:
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
    # Load detection settings from config at startup
    _load_detection_settings()

    while True:
        print("\n=== eyeTool - Camera Application ===")
        print(f"Camera source: {source}")
        print(f"Display target: {os.environ.get('DISPLAY', '(not set)')}")
        det_status = "ON" if detection_enabled else "OFF"
        cfg = get_config()
        rec_status = "ON" if cfg.get("recording.enabled", True) else "OFF"
        print(" 1. Multi-camera feed (with recording)")
        print(" 2. Live camera feed (single)")
        print(" 3. Setup alarm zones")
        print(" 4. Image preprocessing")
        print(" 5. Probe camera (no GUI)")
        print(" 6. Monitoring TUI")
        print(" 7. Capture single image")
        print(" 8. Select display target")
        print(f" 9. Detection settings [{det_status}]")
        print(f"10. Recording settings [{rec_status}]")
        print("11. Configuration Saving")
        print("12. Exit")

        choice = input("\nEnter your choice (1-12): ").strip()
        if choice == "1":
            load_multi_camera_feed()
        elif choice == "2":
            select_camera_for_feed()
        elif choice == "3":
            setup_zones_menu()
        elif choice == "4":
            preprocess_settings_menu()
        elif choice == "5":
            probe_camera(source)
        elif choice == "6":
            from ui.monitor import run as run_monitor
            run_monitor()
        elif choice == "7":
            capture_single_image(source, output)
        elif choice == "8":
            select_display_menu()
        elif choice == "9":
            detection_settings_menu()
        elif choice == "10":
            recording_settings_menu()
        elif choice == "11":
            configuration_menu()
        elif choice == "12":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1-12.")


def camera_mapping_menu(cameras: list[str]) -> None:
    """Interactive menu for mapping cameras to multi-feed screen positions."""
    cfg = get_config()
    
    # Load current mapping
    mapping = {
        "a": cfg.get("camera_mapping.top_left", None),
        "b": cfg.get("camera_mapping.top_right", None),
        "c": cfg.get("camera_mapping.bottom_left", None),
        "d": cfg.get("camera_mapping.bottom_right", None),
    }
    
    position_names = {
        "a": "Top-Left",
        "b": "Top-Right",
        "c": "Bottom-Left",
        "d": "Bottom-Right",
    }
    
    while True:
        print("\n___Multi-camera feed screen mapping___")
        
        # Display current mapping
        for pos, name in position_names.items():
            camera = mapping[pos]
            if camera:
                # Find the index of the camera in the available list
                try:
                    idx = cameras.index(camera) + 1
                    camera_display = f"/dev/video{camera.split('video')[1]} (index {idx})"
                except (ValueError, IndexError):
                    camera_display = camera
                print(f"{pos}. {name}:  {camera_display}")
            else:
                print(f"{pos}. {name}:  **unmapped**")
        
        print("\nFeed Available Cameras:")
        for i, camera in enumerate(cameras, 1):
            print(f"  {i}. {camera}")
        
        print('\nPress "q" to return main menu.')
        cmd = input('Enter your mapping command (e.g. a3 means feeding /dev/video24 to Top-left split screen)\nYour command: ').strip().lower()
        
        if cmd == "q":
            # Save mapping before returning
            cfg.set("camera_mapping.top_left", mapping["a"])
            cfg.set("camera_mapping.top_right", mapping["b"])
            cfg.set("camera_mapping.bottom_left", mapping["c"])
            cfg.set("camera_mapping.bottom_right", mapping["d"])
            cfg.save_user()
            print("Mapping saved to user_settings.json")
            return
        
        # Parse command (e.g., "a3" means map camera 3 to position a)
        if len(cmd) >= 2 and cmd[0] in position_names:
            pos = cmd[0]
            try:
                camera_idx = int(cmd[1:]) - 1
                if 0 <= camera_idx < len(cameras):
                    mapping[pos] = cameras[camera_idx]
                    print(f"Mapped {cameras[camera_idx]} to {position_names[pos]}")
                else:
                    print(f"Invalid camera index. Please enter 1-{len(cameras)}")
            except ValueError:
                print("Invalid command format. Use format like 'a3' (position + camera number)")
        else:
            print("Invalid command. Use format like 'a3' (position + camera number)")


def select_camera_for_feed() -> None:
    """Select camera for live feed by probing all available cameras."""
    print("\n--- Select a Mode to Proceed ---")
    
    from core.camera import find_all_cameras, test_camera, resolve_camera_source
    
    # Get all available cameras
    all_cameras = find_all_cameras()
    working_cameras = []
    
    for camera in all_cameras:
        if test_camera(camera):
            working_cameras.append(camera)
    
    if not working_cameras:
        print("\nNo working cameras found. Using default auto-detection.")
        source = resolve_camera_source(None)
        load_camera_feed(source)
        return
    
    # Show submenu
    print("\n 0. Camera Mapping Multi-camera Feed")
    print("\nFeed Available Cameras:")
    for i, camera in enumerate(working_cameras, 1):
        print(f"  {i}. {camera}")
    
    while True:
        choice = input(f"\nEnter choice (0-{len(working_cameras)}): ").strip()
        
        if choice == "0":
            camera_mapping_menu(working_cameras)
            return
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(working_cameras):
                selected_camera = working_cameras[idx]
                print(f"Selected: {selected_camera}")
                load_camera_feed(selected_camera)
                return
            else:
                print("Invalid choice. Please try again.")
        else:
            print("Please enter a number.")
