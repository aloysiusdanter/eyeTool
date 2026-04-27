"""Multi-camera stream management for eyeTool.

A ``Stream`` wraps one ``cv2.VideoCapture`` plus its own ``FrameSource``
capture thread. ``StreamManager`` owns up to N (=4) streams keyed by
*slot id* (0..N-1) and provides a snapshot API the compositor can call
without blocking.

Slot mapping
------------
The persistent zones.json holds ``port_path -> slot_id`` bindings. When
the manager starts (or hot-plug fires), an enumerated ``CameraInfo`` is
routed to its assigned slot. Unbound cameras fall into the lowest free
slot and the binding is recorded.

State per slot:
    EMPTY      -- no camera ever bound here
    UNAVAILABLE-- bound, but device gone (waiting for hot-plug)
    OPENING    -- VideoCapture being created
    ACTIVE     -- frames flowing
    STALLED    -- no fresh frame for > watchdog_stall_s
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from core.camera import open_camera
from core.hotplug import CameraInfo, list_cameras
from detection.pipeline import FrameSource


class SlotState(str, Enum):
    EMPTY = "empty"
    UNAVAILABLE = "unavailable"
    OPENING = "opening"
    ACTIVE = "active"
    STALLED = "stalled"


@dataclass
class StreamSnapshot:
    """Per-frame data the compositor needs to render one tile."""
    slot_id: int
    state: SlotState
    label: str
    frame: np.ndarray | None
    frame_ts: float
    capture_fps: float
    port_path: str = ""


@dataclass
class _Slot:
    slot_id: int
    name: str = ""
    port_path: str = ""
    state: SlotState = SlotState.EMPTY
    cap: cv2.VideoCapture | None = None
    source: FrameSource | None = None
    last_frame_ts: float = 0.0
    cam_info: CameraInfo | None = None
    polygon: list[tuple[int, int]] = field(default_factory=list)
    preprocessing: dict = field(default_factory=dict)


class StreamManager:
    """Manage up to ``max_streams`` camera streams.

    Hot-plug events feed ``on_hotplug``. The compositor calls
    ``snapshot()`` each render frame to get a list of ``StreamSnapshot``
    (one per configured slot, even if EMPTY/UNAVAILABLE).
    """

    def __init__(self, max_streams: int = 4, watchdog_stall_s: float = 2.0,
                 saved_bindings: dict[str, int] | None = None,
                 on_binding_change=None,
                 saved_preprocess: dict[int, object] | None = None) -> None:
        self.max_streams = max_streams
        self.watchdog_stall_s = watchdog_stall_s
        self._lock = threading.Lock()
        self._slots: dict[int, _Slot] = {i: _Slot(slot_id=i) for i in range(max_streams)}
        # port_path -> slot_id from persisted config
        self._bindings: dict[str, int] = dict(saved_bindings or {})
        # Optional callback fired after a *new* port_path is bound to a slot
        # (so the host can persist zones.json from the main thread).
        self._on_binding_change = on_binding_change
        # Per-slot preprocess hook (Preprocess instance or any callable).
        # Stored separately from _Slot so it survives attach/detach cycles
        # without depending on whether the camera is currently connected.
        self._preprocess: dict[int, object] = dict(saved_preprocess or {})
        # Reverse lookup convenience: slot_id -> port_path
        for pp, sid in self._bindings.items():
            if 0 <= sid < max_streams:
                self._slots[sid].port_path = pp
                self._slots[sid].state = SlotState.UNAVAILABLE

    # --- bindings ----------------------------------------------------
    def bindings(self) -> dict[str, int]:
        with self._lock:
            return dict(self._bindings)

    def assign(self, info: CameraInfo) -> int | None:
        """Decide which slot a camera goes to. Returns the slot id, or None
        if all slots are full and this camera has no prior binding.

        If a *new* binding is created, fires ``on_binding_change(port_path,
        slot_id)`` outside the lock so callers can persist zones.json.
        """
        new_binding: tuple[str, int] | None = None
        with self._lock:
            # 1) honour an existing binding
            if info.port_path and info.port_path in self._bindings:
                return self._bindings[info.port_path]
            # 2) take the lowest empty slot
            for sid in range(self.max_streams):
                slot = self._slots[sid]
                if slot.state == SlotState.EMPTY:
                    self._bindings[info.port_path] = sid
                    slot.port_path = info.port_path
                    new_binding = (info.port_path, sid)
                    break
            else:
                return None
        if new_binding and self._on_binding_change is not None:
            try:
                self._on_binding_change(*new_binding)
            except Exception as e:  # noqa: BLE001
                print(f"[stream] on_binding_change failed: {e}")
        return new_binding[1] if new_binding else None

    # --- lifecycle ---------------------------------------------------
    def open_all_present(self) -> None:
        """Enumerate currently-connected cameras and open each into its slot."""
        for cam in list_cameras():
            self.attach(cam)

    def attach(self, info: CameraInfo) -> None:
        """Open *info* into its assigned (or freshly chosen) slot."""
        sid = self.assign(info)
        if sid is None:
            print(f"[stream] no free slot for {info.label}; ignored")
            return
        with self._lock:
            slot = self._slots[sid]
            if slot.source is not None:
                # Already attached (e.g. duplicate udev event)
                return
            slot.state = SlotState.OPENING
            slot.cam_info = info
            slot.port_path = info.port_path
            if not slot.name:
                slot.name = info.model or f"slot {sid}"
        # Open camera outside the lock (cv2.VideoCapture may block briefly).
        # On a fresh hot-plug, V4L2 may not be ready for ~100 ms; retry once.
        cap = open_camera(info.devnode)
        if cap is None:
            time.sleep(0.25)
            cap = open_camera(info.devnode)
        if cap is None:
            with self._lock:
                slot.state = SlotState.UNAVAILABLE
            return
        src = FrameSource(cap, preprocess=self._preprocess.get(sid))
        src.start()
        with self._lock:
            slot.cap = cap
            slot.source = src
            slot.state = SlotState.ACTIVE
            slot.last_frame_ts = time.monotonic()
        print(f"[stream] slot {sid}: ACTIVE  {info.label}  ({info.devnode})")

    def set_preprocess(self, slot_id: int, preprocess) -> None:
        """Install a new preprocess callable for *slot_id*.

        Effective immediately on the next captured frame -- works while
        the slot is live or detached. Pass ``None`` to disable.
        """
        with self._lock:
            if preprocess is None:
                self._preprocess.pop(slot_id, None)
            else:
                self._preprocess[slot_id] = preprocess
            slot = self._slots.get(slot_id)
            src = slot.source if slot else None
        if src is not None:
            src.preprocess = preprocess

    def get_preprocess(self, slot_id: int):
        with self._lock:
            return self._preprocess.get(slot_id)

    def detach_by_port(self, port_path: str) -> None:
        """Tear down whichever slot is currently using *port_path*."""
        with self._lock:
            target = next((s for s in self._slots.values()
                           if s.port_path == port_path and s.source is not None),
                          None)
            if target is None:
                return
            sid = target.slot_id
        self._teardown(sid, set_state=SlotState.UNAVAILABLE)
        print(f"[stream] slot {sid}: UNAVAILABLE  ({port_path})")

    def detach_slot(self, slot_id: int) -> None:
        """Tear down *slot_id* (used at shutdown)."""
        self._teardown(slot_id, set_state=SlotState.EMPTY)

    def _teardown(self, slot_id: int, set_state: SlotState) -> None:
        with self._lock:
            slot = self._slots.get(slot_id)
            if slot is None:
                return
            src, cap = slot.source, slot.cap
            slot.source = None
            slot.cap = None
            slot.state = set_state
        if src is not None:
            src.stop()
        if cap is not None:
            try:
                cap.release()
            except Exception:  # noqa: BLE001
                pass

    def stop_all(self) -> None:
        for sid in list(self._slots.keys()):
            self.detach_slot(sid)

    # --- hot-plug ----------------------------------------------------
    def on_hotplug(self, action: str, info: CameraInfo) -> None:
        """Callback for ``HotplugMonitor``."""
        if action == "add":
            # udev fires multiple events for one chip; ignore if we already
            # have an active stream for this port.
            with self._lock:
                already = any(s.source is not None and s.port_path == info.port_path
                              for s in self._slots.values())
            if already:
                return
            self.attach(info)
        elif action == "remove":
            self.detach_by_port(info.port_path)

    # --- snapshot ----------------------------------------------------
    def snapshot(self) -> list[StreamSnapshot]:
        """Non-blocking per-slot view used by the compositor each frame."""
        out: list[StreamSnapshot] = []
        now = time.monotonic()
        with self._lock:
            slots = list(self._slots.values())

        for slot in slots:
            frame = None
            ts = 0.0
            cap_fps = 0.0
            state = slot.state
            if slot.source is not None:
                snap = slot.source.get_latest()
                cap_fps = slot.source.fps()
                if snap is not None:
                    frame, ts = snap
                    slot.last_frame_ts = ts
                # Watchdog: stall if no fresh frame for too long
                age = now - slot.last_frame_ts if slot.last_frame_ts else 0.0
                if slot.state == SlotState.ACTIVE and age > self.watchdog_stall_s:
                    state = SlotState.STALLED
            label = slot.name or slot.port_path or f"slot {slot.slot_id}"
            out.append(StreamSnapshot(
                slot_id=slot.slot_id,
                state=state,
                label=label,
                frame=frame,
                frame_ts=ts,
                capture_fps=cap_fps,
                port_path=slot.port_path,
            ))
        return out

    # --- introspection helpers --------------------------------------
    def active_slots(self) -> list[int]:
        with self._lock:
            return [s.slot_id for s in self._slots.values()
                    if s.source is not None]

    def get_frame(self, slot_id: int) -> tuple[np.ndarray, float] | None:
        with self._lock:
            slot = self._slots.get(slot_id)
            src = slot.source if slot else None
        if src is None:
            return None
        return src.get_latest()
