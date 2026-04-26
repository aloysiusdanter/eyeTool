"""USB camera enumeration and hot-plug detection via pyudev.

Each connected UVC camera exposes one or more ``/dev/videoN`` nodes; only
the first one (the streaming/capture node) is useful to OpenCV. We
identify cameras by their kernel-stable ``ID_PATH`` (USB topology path)
so that "the camera plugged into hub port 1.2" always lands in the same
grid slot, regardless of which ``/dev/videoN`` index udev hands out.

Public API
----------
``list_cameras()``                -- snapshot of currently connected USB
                                     UVC capture devices.
``HotplugMonitor(callback)``      -- background thread that fires
                                     ``callback(event, CameraInfo)`` on
                                     ``"add"`` / ``"remove"`` events.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

try:
    import pyudev  # type: ignore
    _HAS_PYUDEV = True
except ImportError:  # pragma: no cover
    pyudev = None  # type: ignore
    _HAS_PYUDEV = False


@dataclass(frozen=True)
class CameraInfo:
    """Identity of a connected UVC capture device."""

    devnode: str          # /dev/videoN (the capture node)
    port_path: str        # kernel ID_PATH, e.g. 'platform-fc800000.usb-usb-0:1:1.0'
    model: str            # ID_MODEL (human friendly)
    vendor: str           # ID_VENDOR
    serial: str           # ID_SERIAL_SHORT (may be '')

    @property
    def label(self) -> str:
        return f"{self.model or 'camera'} @ {self.port_path}"


def _is_capture_node(device) -> bool:
    """A UVC chip exposes both video (capture) and metadata nodes; only the
    capture node has the V4L2_CAP_VIDEO_CAPTURE bit. udev tags it
    ``ID_V4L_CAPABILITIES`` containing ``:capture:``."""
    caps = device.get("ID_V4L_CAPABILITIES", "") or ""
    if ":capture:" in caps:
        return True
    # Fallback: inspect device sysfs node modalias / interfaces. Some kernels
    # don't tag CAPABILITIES; treat the lowest-numbered video node per usb
    # parent as the capture node (handled at the dedup stage in list_cameras).
    return False


def _is_usb_video(device) -> bool:
    if device.subsystem != "video4linux":
        return False
    id_path = device.get("ID_PATH", "")
    # USB devices have ID_BUS=usb or ID_PATH containing 'usb-'
    if device.get("ID_BUS") == "usb":
        return True
    if "usb-" in id_path:
        return True
    return False


def _to_camera(device) -> CameraInfo | None:
    if not _is_usb_video(device):
        return None
    devnode = device.device_node
    if not devnode:
        return None
    return CameraInfo(
        devnode=devnode,
        port_path=device.get("ID_PATH", "") or "",
        model=device.get("ID_MODEL", "") or "",
        vendor=device.get("ID_VENDOR", "") or "",
        serial=device.get("ID_SERIAL_SHORT", "") or "",
    )


def list_cameras() -> list[CameraInfo]:
    """Snapshot of currently-connected USB UVC capture devices.

    De-duplicates per ``port_path``: a UVC chip exposes 2 ``/dev/videoN``
    nodes (capture + metadata); we keep only the capture node, falling back
    to the lowest-numbered devnode if udev does not tag capabilities.
    """
    if not _HAS_PYUDEV:
        return _fallback_list()

    ctx = pyudev.Context()
    by_port: dict[str, CameraInfo] = {}
    for dev in ctx.list_devices(subsystem="video4linux"):
        cam = _to_camera(dev)
        if cam is None or not cam.port_path:
            continue
        is_cap = _is_capture_node(dev)
        existing = by_port.get(cam.port_path)
        if existing is None:
            by_port[cam.port_path] = cam
            continue
        # Prefer node tagged as capture; otherwise the lower devnode wins.
        if is_cap:
            by_port[cam.port_path] = cam
        elif existing.devnode > cam.devnode and not _is_capture_node(_lookup(ctx, existing.devnode)):
            by_port[cam.port_path] = cam
    return sorted(by_port.values(), key=lambda c: c.port_path)


def _lookup(ctx, devnode: str):
    for dev in ctx.list_devices(subsystem="video4linux"):
        if dev.device_node == devnode:
            return dev
    return None


def _fallback_list() -> list[CameraInfo]:  # pragma: no cover
    """Minimal fallback when pyudev is unavailable: scan /dev/video*."""
    out: list[CameraInfo] = []
    for n in sorted(os.listdir("/dev")):
        if not n.startswith("video"):
            continue
        out.append(CameraInfo(devnode=f"/dev/{n}", port_path=f"/dev/{n}",
                              model="", vendor="", serial=""))
    return out


# ---------------------------------------------------------------------------
# Live monitor
# ---------------------------------------------------------------------------

class HotplugMonitor:
    """Background thread that calls back on USB UVC add/remove events.

    The callback signature is ``callback(event_type: str, info: CameraInfo)``
    where ``event_type`` is ``"add"`` or ``"remove"``. Callback is invoked
    on the monitor thread, so it must be quick / thread-safe.
    """

    def __init__(self, callback) -> None:
        self._cb = callback
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._monitor = None

    def start(self) -> bool:
        if not _HAS_PYUDEV:
            print("[hotplug] pyudev not installed; live hot-plug disabled")
            return False
        if self._thread is not None:
            return True
        ctx = pyudev.Context()
        self._monitor = pyudev.Monitor.from_netlink(ctx)
        self._monitor.filter_by(subsystem="video4linux")
        self._thread = threading.Thread(target=self._run, name="HotplugMonitor",
                                        daemon=True)
        self._thread.start()
        return True

    def stop(self, timeout: float = 0.5) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        # poll() blocks for up to 1 s, then returns None; we re-check _stop.
        while not self._stop.is_set():
            try:
                dev = self._monitor.poll(timeout=1.0)
            except Exception as e:  # noqa: BLE001
                print(f"[hotplug] monitor poll failed: {e}")
                return
            if dev is None:
                continue
            cam = _to_camera(dev)
            if cam is None or not cam.port_path:
                continue
            action = dev.action  # 'add', 'remove', 'change', ...
            if action not in ("add", "remove"):
                continue
            # On 'add' a UVC chip emits one event per /dev/videoN node
            # (capture + metadata). Keep only the capture node so we don't
            # try to open the metadata node into a slot.
            if action == "add" and not _is_capture_node(dev):
                continue
            try:
                self._cb(action, cam)
            except Exception as e:  # noqa: BLE001
                print(f"[hotplug] callback failed: {e}")
