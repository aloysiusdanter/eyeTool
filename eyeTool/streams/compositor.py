"""Multi-stream 2x2 grid compositor for the 800x480 NanoPi LCD.

Lays out up to 4 ``StreamSnapshot`` tiles in this order:

    +-------+-------+
    | slot0 | slot1 |
    +-------+-------+
    | slot2 | slot3 |
    +-------+-------+

Each tile is 400x240 (display_w/2, display_h/2). Frames are letterboxed
into the tile preserving aspect ratio; "Unavailable" slots get a dark
panel with a centred status message.

The compositor is pure rendering: it never opens cameras and never
runs inference. Detection overlays and zone polygons are drawn here
but the data is supplied by callers (Detector cache, config).
"""

from __future__ import annotations

import cv2
import numpy as np

from streams.stream import SlotState, StreamSnapshot

# BGR colors
_BG = (24, 24, 24)
_TILE_BG = (32, 32, 32)
_TEXT = (240, 240, 240)
_TEXT_DIM = (140, 140, 140)
_BORDER = (60, 60, 60)
_STATE_COLOR: dict[SlotState, tuple[int, int, int]] = {
    SlotState.EMPTY: (90, 90, 90),
    SlotState.UNAVAILABLE: (0, 0, 200),     # red-ish
    SlotState.OPENING: (0, 200, 200),       # yellow
    SlotState.STALLED: (0, 165, 255),       # orange
    SlotState.ACTIVE: (0, 200, 0),          # green
}
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _letterbox_into(canvas_tile: np.ndarray, frame: np.ndarray) -> tuple[float, int, int]:
    """Resize *frame* to fit *canvas_tile* preserving aspect ratio.

    Writes the result into ``canvas_tile`` (modified in-place). Returns
    ``(scale, off_x, off_y)`` so callers can map original-frame coords
    onto the tile.
    """
    th, tw = canvas_tile.shape[:2]
    fh, fw = frame.shape[:2]
    scale = min(tw / fw, th / fh)
    new_w = max(1, int(round(fw * scale)))
    new_h = max(1, int(round(fh * scale)))
    resized = cv2.resize(frame, (new_w, new_h),
                         interpolation=cv2.INTER_LINEAR)
    off_x = (tw - new_w) // 2
    off_y = (th - new_h) // 2
    canvas_tile[:] = _TILE_BG
    canvas_tile[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    return scale, off_x, off_y


def _draw_unavailable_tile(tile: np.ndarray, snap: StreamSnapshot) -> None:
    """Fill *tile* with a placeholder for a slot with no live frame."""
    th, tw = tile.shape[:2]
    tile[:] = _BG
    msg_map = {
        SlotState.EMPTY: "EMPTY SLOT",
        SlotState.UNAVAILABLE: "UNAVAILABLE",
        SlotState.OPENING: "OPENING...",
        SlotState.STALLED: "STALLED",
    }
    msg = msg_map.get(snap.state, snap.state.value.upper())
    color = _STATE_COLOR.get(snap.state, _TEXT_DIM)

    # Center main message
    (tw_text, th_text), _ = cv2.getTextSize(msg, _FONT, 0.7, 2)
    x = (tw - tw_text) // 2
    y = (th + th_text) // 2
    cv2.putText(tile, msg, (x, y), _FONT, 0.7, color, 2, cv2.LINE_AA)

    # Sub-line: port path so the user knows which physical port is offline
    if snap.port_path:
        sub = snap.port_path
        if len(sub) > 38:
            sub = "..." + sub[-35:]
        (sw, sh), _ = cv2.getTextSize(sub, _FONT, 0.4, 1)
        cv2.putText(tile, sub, ((tw - sw) // 2, y + sh + 8),
                    _FONT, 0.4, _TEXT_DIM, 1, cv2.LINE_AA)


def _draw_label_strip(tile: np.ndarray, snap: StreamSnapshot) -> None:
    """Top status bar: state dot + slot label + capture FPS."""
    th, tw = tile.shape[:2]
    bar_h = 18
    cv2.rectangle(tile, (0, 0), (tw, bar_h), (0, 0, 0), -1)
    # state dot
    dot_color = _STATE_COLOR.get(snap.state, _TEXT_DIM)
    cv2.circle(tile, (10, bar_h // 2), 4, dot_color, -1)
    # label
    text = f" {snap.slot_id}: {snap.label}"
    if len(text) > 38:
        text = text[:35] + "..."
    cv2.putText(tile, text, (16, bar_h - 5), _FONT, 0.4, _TEXT, 1, cv2.LINE_AA)
    # right side: fps
    if snap.capture_fps > 0:
        fps_txt = f"{snap.capture_fps:5.1f} Hz"
        (fw, _), _ = cv2.getTextSize(fps_txt, _FONT, 0.4, 1)
        cv2.putText(tile, fps_txt, (tw - fw - 6, bar_h - 5),
                    _FONT, 0.4, _TEXT_DIM, 1, cv2.LINE_AA)


class GridCompositor:
    """2x2 (or smaller) compositor that produces one display canvas frame.

    Use ``render(snapshots)`` each tick. The result is a fresh
    ``(display_h, display_w, 3)`` ``uint8`` array ready for ``cv2.imshow``.
    Tile overlays for detection / polygon zones are added via the
    ``overlay_callbacks`` mechanism (set with ``set_overlay``).
    """

    def __init__(self, display_w: int = 800, display_h: int = 480,
                 grid_cols: int = 2, grid_rows: int = 2) -> None:
        self.display_w = display_w
        self.display_h = display_h
        self.cols = grid_cols
        self.rows = grid_rows
        self.tile_w = display_w // grid_cols
        self.tile_h = display_h // grid_rows
        self._overlay_cb = None  # callable(slot_id, tile, scale, off_x, off_y, snap)

    def set_overlay(self, callback) -> None:
        """Register a per-tile overlay drawer (e.g. detection boxes)."""
        self._overlay_cb = callback

    def _tile_origin(self, slot_id: int) -> tuple[int, int]:
        col = slot_id % self.cols
        row = slot_id // self.cols
        return col * self.tile_w, row * self.tile_h

    def render(self, snapshots: list[StreamSnapshot]) -> np.ndarray:
        canvas = np.full((self.display_h, self.display_w, 3),
                         _BG, dtype=np.uint8)
        # Index by slot id so we always render in slot order.
        by_slot = {s.slot_id: s for s in snapshots}
        for slot_id in range(self.cols * self.rows):
            x0, y0 = self._tile_origin(slot_id)
            tile = canvas[y0:y0 + self.tile_h, x0:x0 + self.tile_w]
            snap = by_slot.get(slot_id)
            if snap is None or snap.frame is None:
                if snap is None:
                    snap = StreamSnapshot(slot_id=slot_id, state=SlotState.EMPTY,
                                          label="", frame=None, frame_ts=0.0,
                                          capture_fps=0.0)
                _draw_unavailable_tile(tile, snap)
                _draw_label_strip(tile, snap)
                # divider lines
                cv2.rectangle(tile, (0, 0),
                              (self.tile_w - 1, self.tile_h - 1), _BORDER, 1)
                continue

            scale, off_x, off_y = _letterbox_into(tile, snap.frame)
            if self._overlay_cb is not None:
                try:
                    self._overlay_cb(snap.slot_id, tile, scale, off_x, off_y, snap)
                except Exception as e:  # noqa: BLE001
                    print(f"[compositor] overlay cb failed on slot {snap.slot_id}: {e}")
            _draw_label_strip(tile, snap)
            cv2.rectangle(tile, (0, 0),
                          (self.tile_w - 1, self.tile_h - 1), _BORDER, 1)
        return canvas
