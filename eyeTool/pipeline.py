"""Async capture + inference pipeline for eyeTool.

Two background threads feed the main display loop:

- ``FrameSource`` reads from an already-opened ``cv2.VideoCapture`` as
  fast as the camera produces frames and keeps **only the most recent
  one** in a single-slot buffer. Older frames are dropped, which is the
  whole point -- never accumulate latency in a queue.

- ``Detector`` pulls the latest frame whenever the NPU is free, runs
  ``rknn_yolov8.infer``, and publishes the latest detection result
  (boxes + scale/pad + source-frame timestamp) into a single-slot buffer.

The main thread never blocks on either producer; ``get_latest()`` is
always non-blocking and returns ``None`` when nothing has been produced
yet.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from rknn_yolov8 import (infer as rknn_infer, warmup as rknn_warmup,
                          warmup_all_cores as rknn_warmup_all_cores,
                          _get_rknn_all_cores)


@dataclass
class DetectionResult:
    boxes: np.ndarray          # (N, 4) xyxy in 640-input letterboxed space
    classes: np.ndarray        # (N,) int
    scores: np.ndarray         # (N,) float
    scale: float               # letterbox scale used at inference time
    pad: tuple[int, int]       # (pad_w, pad_h) in the 640x640 canvas
    frame_ts: float            # source-frame capture timestamp (monotonic)
    frame_w: int               # source-frame width
    frame_h: int               # source-frame height


class FrameSource:
    """Background thread that keeps only the latest camera frame.

    Designed to be started once per ``cv2.VideoCapture``. Call
    ``get_latest()`` from the display loop; call ``stop()`` before
    releasing the capture.
    """

    def __init__(self, cap: cv2.VideoCapture,
                 preprocess=None) -> None:
        self._cap = cap
        self._lock = threading.Lock()
        self._latest: tuple[np.ndarray, float] | None = None
        self._stop = threading.Event()
        self._new_frame = threading.Event()
        self._thread = threading.Thread(target=self._run, name="FrameSource",
                                        daemon=True)
        self._frame_count = 0
        self._start_ts = 0.0
        # Optional callable ``frame -> frame`` applied to every frame
        # before it becomes visible to consumers (detector + compositor).
        # Atomic reference swap is enough -- CPython attribute writes are
        # atomic, and the worker reads it once per frame.
        self.preprocess = preprocess

    # --- public ------------------------------------------------------
    def start(self) -> None:
        self._start_ts = time.monotonic()
        self._thread.start()

    def stop(self, timeout: float = 0.5) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)

    def get_latest(self) -> tuple[np.ndarray, float] | None:
        """Return a reference to the most recent (frame, ts) or None."""
        with self._lock:
            return self._latest

    def wait_new(self, timeout: float = 1.0) -> bool:
        """Block until a new frame arrives or *timeout* elapses."""
        got = self._new_frame.wait(timeout)
        self._new_frame.clear()
        return got

    def fps(self) -> float:
        elapsed = time.monotonic() - self._start_ts
        return self._frame_count / elapsed if elapsed > 0 else 0.0

    # --- worker ------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                # Avoid busy-spinning if the camera glitches for a moment.
                time.sleep(0.005)
                continue
            # Per-camera preprocessing (brightness/contrast/saturation/
            # gamma). The hook is hot-swappable: see Preprocess in
            # preprocess.py and the live editor in main.preprocess_menu.
            pre = self.preprocess
            if pre is not None:
                try:
                    frame = pre(frame)
                except Exception as e:  # noqa: BLE001
                    print(f"[FrameSource] preprocess failed: {e}; "
                          f"disabling for this stream")
                    self.preprocess = None
            ts = time.monotonic()
            with self._lock:
                self._latest = (frame, ts)
            self._frame_count += 1
            self._new_frame.set()


class Detector:
    """Background thread running NPU inference on the latest frame.

    ``detect_every_n`` > 1 throttles the detector (publish a new result
    every N source frames). Set ``conf_thres`` once at construction or
    update it live via the ``conf_thres`` attribute (plain float, atomic
    assignment in CPython).

    ``use_multi_core`` enables all-core mode: one RKNN instance that uses
    NPU_CORE_0|1|2 simultaneously, reducing per-inference latency ~3×.
    """

    def __init__(self, source: FrameSource, conf_thres: float = 0.5,
                 detect_every_n: int = 1, use_multi_core: bool = False) -> None:
        self._source = source
        self.conf_thres = conf_thres
        self.detect_every_n = max(1, int(detect_every_n))
        self.use_multi_core = use_multi_core
        self._lock = threading.Lock()
        self._latest: DetectionResult | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="Detector",
                                        daemon=True)
        self._infer_count = 0
        self._start_ts = 0.0

    # --- public ------------------------------------------------------
    def start(self) -> None:
        # Warm up on the caller thread so the first inference latency
        # doesn't stall the first display frame.
        if self.use_multi_core:
            rknn_warmup_all_cores()
        else:
            rknn_warmup()
        self._start_ts = time.monotonic()
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)

    def get_latest(self) -> DetectionResult | None:
        with self._lock:
            return self._latest

    def fps(self) -> float:
        elapsed = time.monotonic() - self._start_ts
        return self._infer_count / elapsed if elapsed > 0 else 0.0

    # --- worker ------------------------------------------------------
    def _run(self) -> None:
        last_seen_ts: float | None = None
        frames_since_infer = 0
        # Resolve RKNN instance once (all-core or single-core AUTO)
        rknn_inst = _get_rknn_all_cores() if self.use_multi_core else None
        # Timing for profiling
        cycle_start = time.monotonic()
        overhead_time_sum = 0.0
        while not self._stop.is_set():
            # Poll for the latest frame (no wait_new to avoid missing frames)
            snap = self._source.get_latest()
            if snap is None:
                time.sleep(0.001)
                continue
            frame, ts = snap
            if ts == last_seen_ts:
                # Same frame we just processed; skip and retry
                time.sleep(0.001)
                continue
            frames_since_infer += 1
            if frames_since_infer < self.detect_every_n:
                last_seen_ts = ts
                continue
            frames_since_infer = 0
            last_seen_ts = ts

            t0 = time.monotonic()
            h, w = frame.shape[:2]
            boxes, classes, scores, scale, pad = rknn_infer(
                frame, conf_thres=self.conf_thres, rknn=rknn_inst)
            result = DetectionResult(boxes=boxes, classes=classes,
                                     scores=scores, scale=scale, pad=pad,
                                     frame_ts=ts, frame_w=w, frame_h=h)
            with self._lock:
                self._latest = result
            self._infer_count += 1
            t1 = time.monotonic()

            # Track overhead (total cycle time - inference time)
            cycle_time = t1 - cycle_start
            infer_time = t1 - t0
            overhead_time_sum += (cycle_time - infer_time)
            cycle_start = t1

            # Print stats every 100 inferences
            if self._infer_count % 100 == 0:
                avg_overhead = overhead_time_sum / self._infer_count * 1000
                print(f"Detector overhead: avg {avg_overhead:.1f}ms per cycle")


class MultiDetector:
    """Round-robin NPU detector across all active stream slots.

    The NPU is a single hardware resource, so running multiple detector
    threads would only serialize on it and add contention. Instead, one
    worker thread cycles through ``stream_manager.active_slots()`` in
    order, running inference on the *latest* frame of each slot it visits.

    The per-slot latest result is published into ``_by_slot`` and read
    non-blocking via ``get_result(slot_id)``.
    """

    def __init__(self, stream_manager, conf_thres: float = 0.5,
                 detect_every_n: int = 1, use_multi_core: bool = False) -> None:
        self._mgr = stream_manager
        self.conf_thres = conf_thres
        self.detect_every_n = max(1, int(detect_every_n))
        self.use_multi_core = use_multi_core
        self._lock = threading.Lock()
        self._by_slot: dict[int, DetectionResult] = {}
        self._per_slot_frames_since: dict[int, int] = {}
        self._per_slot_last_ts: dict[int, float] = {}
        self._infer_count = 0
        self._start_ts = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="MultiDetector",
                                        daemon=True)

    # --- public ------------------------------------------------------
    def start(self) -> None:
        if self.use_multi_core:
            rknn_warmup_all_cores()
        else:
            rknn_warmup()
        self._start_ts = time.monotonic()
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)

    def get_result(self, slot_id: int) -> DetectionResult | None:
        with self._lock:
            return self._by_slot.get(slot_id)

    def fps(self) -> float:
        elapsed = time.monotonic() - self._start_ts
        return self._infer_count / elapsed if elapsed > 0 else 0.0

    # --- worker ------------------------------------------------------
    def _run(self) -> None:
        rknn_inst = _get_rknn_all_cores() if self.use_multi_core else None
        cycle_idx = 0
        while not self._stop.is_set():
            active = self._mgr.active_slots()
            if not active:
                time.sleep(0.02)
                continue

            # Round-robin: pick next slot in the active list.
            slot_id = active[cycle_idx % len(active)]
            cycle_idx += 1

            snap = self._mgr.get_frame(slot_id)
            if snap is None:
                time.sleep(0.001)
                continue
            frame, ts = snap
            last_ts = self._per_slot_last_ts.get(slot_id)
            if ts == last_ts:
                # No new frame for this slot; move on quickly.
                continue

            frames_since = self._per_slot_frames_since.get(slot_id, 0) + 1
            if frames_since < self.detect_every_n:
                self._per_slot_frames_since[slot_id] = frames_since
                self._per_slot_last_ts[slot_id] = ts
                continue
            self._per_slot_frames_since[slot_id] = 0
            self._per_slot_last_ts[slot_id] = ts

            h, w = frame.shape[:2]
            try:
                boxes, classes, scores, scale, pad = rknn_infer(
                    frame, conf_thres=self.conf_thres, rknn=rknn_inst)
            except Exception as e:  # noqa: BLE001
                print(f"[multi-det] slot {slot_id} inference failed: {e}")
                continue
            result = DetectionResult(boxes=boxes, classes=classes,
                                     scores=scores, scale=scale, pad=pad,
                                     frame_ts=ts, frame_w=w, frame_h=h)
            with self._lock:
                self._by_slot[slot_id] = result
            self._infer_count += 1


# ------ overlay helpers ------------------------------------------------

GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
PERSON_CLASS_ID = 0
STALE_THRESHOLD_S = 0.2  # boxes older than 200 ms render yellow


def overlay_detections(frame: np.ndarray, det: DetectionResult | None,
                       now_ts: float, person_only: bool = True) -> int:
    """Draw the *det* boxes on *frame* in-place; return person count.

    Boxes are rendered yellow if the source frame is > ``STALE_THRESHOLD_S``
    behind ``now_ts`` (i.e. the detector is falling behind).
    """
    if det is None or det.boxes.shape[0] == 0:
        return 0

    age = now_ts - det.frame_ts
    colour = YELLOW if age > STALE_THRESHOLD_S else GREEN

    # Boxes are in 640-input letterboxed space referenced to (det.frame_w,
    # det.frame_h). Map back to original source-frame coords.
    scale = det.scale
    pad_w, pad_h = det.pad
    fh, fw = frame.shape[:2]

    count = 0
    for (bx1, by1, bx2, by2), score, cls in zip(det.boxes, det.scores, det.classes):
        if person_only and int(cls) != PERSON_CLASS_ID:
            continue
        x1 = int((bx1 - pad_w) / scale)
        x2 = int((bx2 - pad_w) / scale)
        y1 = int((by1 - pad_h) / scale)
        y2 = int((by2 - pad_h) / scale)
        # If the detection was made against a different-sized frame, rescale.
        if (det.frame_w, det.frame_h) != (fw, fh):
            sx = fw / det.frame_w
            sy = fh / det.frame_h
            x1 = int(x1 * sx); x2 = int(x2 * sx)
            y1 = int(y1 * sy); y2 = int(y2 * sy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(frame, f"person {float(score):.2f}",
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
        count += 1
    return count


RED = (0, 0, 255)


def overlay_tile_detections(tile: np.ndarray, det: DetectionResult | None,
                            tile_scale: float, tile_off_x: int, tile_off_y: int,
                            now_ts: float,
                            person_only: bool = True,
                            stale_age_s: float = STALE_THRESHOLD_S,
                            zone=None) -> tuple[int, int]:
    """Draw *det* boxes on a compositor *tile* (post-letterbox).

    Two transforms compose:
        infer-space (bx) -> original-frame (ox) -> tile-space (tx)
            ox = (bx - infer_pad) / infer_scale
            tx = ox * tile_scale + tile_off_x

    Color rules (per box):
        * detection age > stale_age_s                -> YELLOW (stale)
        * zone defined AND bbox **touches/overlaps** -> RED    (alarm)
        * else                                       -> GREEN

    The alarm condition is full bbox-vs-polygon intersection (corners-in,
    vertices-in, or edge crossings) so a box turns red the moment it
    grazes the polygon boundary -- not only once it's fully inside.

    Returns ``(boxes_drawn, alarm_count)`` -- number of red boxes is the
    second element.
    """
    if det is None or det.boxes.shape[0] == 0:
        return 0, 0
    age = now_ts - det.frame_ts
    is_stale = age > stale_age_s
    pad_w, pad_h = det.pad
    inv_infer = 1.0 / det.scale if det.scale else 1.0
    th, tw = tile.shape[:2]

    drawn = 0
    alarms = 0
    for (bx1, by1, bx2, by2), score, cls in zip(det.boxes, det.scores, det.classes):
        if person_only and int(cls) != PERSON_CLASS_ID:
            continue
        # infer -> original
        ox1 = (bx1 - pad_w) * inv_infer
        oy1 = (by1 - pad_h) * inv_infer
        ox2 = (bx2 - pad_w) * inv_infer
        oy2 = (by2 - pad_h) * inv_infer

        # zone test in original-frame coords: bbox-vs-polygon intersection
        # (touching the edge counts as alarm).
        inside = False
        if zone is not None and zone.is_valid:
            inside = zone.intersects_bbox(ox1, oy1, ox2, oy2)

        if is_stale:
            colour = YELLOW
        elif inside:
            colour = RED
            alarms += 1
        else:
            colour = GREEN

        # original -> tile
        tx1 = int(round(ox1 * tile_scale + tile_off_x))
        ty1 = int(round(oy1 * tile_scale + tile_off_y))
        tx2 = int(round(ox2 * tile_scale + tile_off_x))
        ty2 = int(round(oy2 * tile_scale + tile_off_y))
        # clip
        tx1 = max(0, min(tw - 1, tx1)); tx2 = max(0, min(tw - 1, tx2))
        ty1 = max(0, min(th - 1, ty1)); ty2 = max(0, min(th - 1, ty2))
        if tx2 <= tx1 or ty2 <= ty1:
            continue
        thickness = 2 if (inside and not is_stale) else 1
        cv2.rectangle(tile, (tx1, ty1), (tx2, ty2), colour, thickness)
        label = f"{float(score):.2f}"
        cv2.putText(tile, label, (tx1, max(0, ty1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1, cv2.LINE_AA)
        drawn += 1
    return drawn, alarms
