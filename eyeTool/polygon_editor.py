"""Frozen-frame polygon editor for alarm-zone setup.

Input sources (any combination works simultaneously):

* **OpenCV window** keystrokes (``cv2.waitKeyEx``) -- only fires when the
  window has X11 keyboard focus. Useless over plain SSH because the
  terminal grabs the keystrokes; fine over ``ssh -X`` or local console.

* **SSH/stdin** keystrokes -- the controlling terminal is put into
  cbreak mode for the editor's lifetime so single key presses (and ANSI
  arrow-key escape sequences) reach us without waiting for Enter. This
  is what lets you drive the editor entirely over a normal SSH session.

* **Mouse on the OpenCV window** -- left-click adds a vertex at the
  click, right-click undoes the last vertex, double left-click finishes
  & saves. Movement just tracks the crosshair to the cursor. Works
  whenever the X11 display has any pointing device.

Polygon storage convention is unchanged: vertices live in **original
frame coordinates** so resolution / letterbox changes never invalidate
them.

Workflow
--------
The caller hands a single still frame plus an optional pre-existing
polygon. ``run()`` blocks until the user finishes (returning the new
polygon as a list of ``(x, y)`` tuples), explicitly deletes the polygon
(returning ``[]``), or cancels (returning ``None``).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

import cv2
import numpy as np

from terminal_input import RawStdin, normalise_cv2_key


def _detect_screen_size(default: tuple[int, int] = (800, 480)
                        ) -> tuple[int, int]:
    """Best-effort current X11 screen resolution.

    Falls back to *default* if ``xrandr`` is unavailable, returns
    nothing parseable, or DISPLAY isn't set. The first ``current
    NNNN x NNNN`` token from ``xrandr --current`` wins; that always
    matches the active screen on a single-output setup like the LCD.
    """
    if not os.environ.get("DISPLAY"):
        return default
    try:
        out = subprocess.check_output(
            ["xrandr", "--current"], stderr=subprocess.DEVNULL,
            timeout=2.0,
        ).decode("utf-8", errors="ignore")
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return default
    m = re.search(r"current\s+(\d+)\s*x\s*(\d+)", out)
    if not m:
        return default
    try:
        return int(m.group(1)), int(m.group(2))
    except ValueError:
        return default

_BG = (24, 24, 24)
_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Visual styling
_COLOR_VERT = (0, 255, 0)
_COLOR_EDGE = (0, 200, 0)
_COLOR_PREVIEW = (0, 200, 200)
_COLOR_CLOSE = (60, 60, 60)
_COLOR_CROSS = (0, 200, 255)
_COLOR_TEXT = (240, 240, 240)
_COLOR_TEXT_DIM = (160, 160, 160)
_COLOR_FILL = (71, 179, 255)
_FILL_ALPHA = 0.18


# ---------------------------------------------------------------------------
# Letterbox helpers + visual rendering
# ---------------------------------------------------------------------------

def _letterbox(frame: np.ndarray, canvas_w: int,
               canvas_h: int) -> tuple[np.ndarray, float, int, int]:
    fh, fw = frame.shape[:2]
    scale = min(canvas_w / fw, canvas_h / fh)
    new_w = max(1, int(round(fw * scale)))
    new_h = max(1, int(round(fh * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((canvas_h, canvas_w, 3), _BG, dtype=np.uint8)
    off_x = (canvas_w - new_w) // 2
    off_y = (canvas_h - new_h) // 2
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    return canvas, scale, off_x, off_y


def _paste_frame(canvas: np.ndarray, frame: np.ndarray,
                 scale: float, off_x: int, off_y: int) -> None:
    """Resize *frame* using pre-computed letterbox params and blit it
    into the image area of an already-allocated canvas. Cheaper than
    re-running ``_letterbox`` because the canvas + black bars stay put.
    """
    fh, fw = frame.shape[:2]
    new_w = max(1, int(round(fw * scale)))
    new_h = max(1, int(round(fh * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized


def _draw_dashed_line(img, p1, p2, color, dash=6) -> None:
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)
    dist = float(np.linalg.norm(p2 - p1))
    if dist < 1:
        return
    n = max(1, int(dist // dash))
    for i in range(0, n, 2):
        a = p1 + (p2 - p1) * (i / n)
        b = p1 + (p2 - p1) * (min(i + 1, n) / n)
        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                 color, 1, cv2.LINE_AA)


def _draw_overlay(canvas: np.ndarray, vertices_disp: list[tuple[int, int]],
                  cross_xy: tuple[int, int], slot_label: str,
                  status: str) -> None:
    n = len(vertices_disp)
    if n >= 3:
        overlay = canvas.copy()
        pts = np.array(vertices_disp, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], _COLOR_FILL)
        cv2.addWeighted(overlay, _FILL_ALPHA, canvas, 1.0 - _FILL_ALPHA,
                        0, dst=canvas)
    for i in range(n - 1):
        cv2.line(canvas, vertices_disp[i], vertices_disp[i + 1],
                 _COLOR_EDGE, 1, cv2.LINE_AA)
    if n >= 1:
        cv2.line(canvas, vertices_disp[-1], cross_xy, _COLOR_PREVIEW, 1,
                 cv2.LINE_AA)
    if n >= 2:
        _draw_dashed_line(canvas, cross_xy, vertices_disp[0], _COLOR_CLOSE)
    for i, (vx, vy) in enumerate(vertices_disp):
        cv2.circle(canvas, (vx, vy), 4, _COLOR_VERT, -1, cv2.LINE_AA)
        cv2.putText(canvas, str(i + 1), (vx + 6, vy - 6),
                    _FONT, 0.4, _COLOR_VERT, 1, cv2.LINE_AA)
    cx, cy = cross_xy
    cv2.line(canvas, (cx - 8, cy), (cx + 8, cy), _COLOR_CROSS, 1)
    cv2.line(canvas, (cx, cy - 8), (cx, cy + 8), _COLOR_CROSS, 1)
    cv2.circle(canvas, (cx, cy), 2, _COLOR_CROSS, 1)

    bar_h = 22
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], bar_h), (0, 0, 0), -1)
    cv2.putText(canvas, f"ZONE EDIT  {slot_label}   vertices: {n}",
                (8, bar_h - 6), _FONT, 0.45, _COLOR_TEXT, 1, cv2.LINE_AA)
    if status:
        (sw, _), _ = cv2.getTextSize(status, _FONT, 0.45, 1)
        cv2.putText(canvas, status, (canvas.shape[1] - sw - 8, bar_h - 6),
                    _FONT, 0.45, _COLOR_TEXT_DIM, 1, cv2.LINE_AA)

    help_lines = [
        "MOUSE: L-click=add  R-click=undo  DBL-click=finish",
        "KEYS:  arrows / WASD = move   ENTER=add  U=undo  C=clear  F=finish  X=delete  ESC=cancel",
    ]
    h = canvas.shape[0]
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, h - 36), (canvas.shape[1], h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, dst=canvas)
    cv2.putText(canvas, help_lines[0], (8, h - 22),
                _FONT, 0.4, _COLOR_TEXT, 1, cv2.LINE_AA)
    cv2.putText(canvas, help_lines[1], (8, h - 6),
                _FONT, 0.4, _COLOR_TEXT, 1, cv2.LINE_AA)


def _disp_to_frame(x: int, y: int, scale: float, off_x: int,
                   off_y: int) -> tuple[int, int]:
    if scale <= 0:
        return 0, 0
    return (int(round((x - off_x) / scale)),
            int(round((y - off_y) / scale)))


def _frame_to_disp(fx: int, fy: int, scale: float, off_x: int,
                   off_y: int) -> tuple[int, int]:
    return int(round(fx * scale + off_x)), int(round(fy * scale + off_y))


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run(frame: np.ndarray,
        existing_polygon: list[tuple[int, int]] | None = None,
        slot_label: str = "",
        canvas_w: int | None = None,
        canvas_h: int | None = None,
        window_name: str = "eyeTool - Zone Editor",
        cap: "cv2.VideoCapture | None" = None,
        ) -> list[tuple[int, int]] | None:
    """Run the editor against either a frozen frame or a live feed.

    ``frame`` is the initial frame -- it sets the original-frame
    coordinate system used for stored vertices. When ``cap`` is also
    supplied, every loop iteration grabs a fresh frame from the
    capture device so the user sees a live view while editing. If
    ``cap`` is ``None`` the original behaviour (frozen still) applies.

    ``canvas_w`` / ``canvas_h`` default to the **current screen
    resolution** (auto-detected via ``xrandr``) so the editor fills the
    LCD with the camera frame letterboxed to preserve aspect ratio.

    Returns:
      * ``list[(x, y)]`` -- the new polygon in *frame* coords (>=3 verts)
        when the user pressed F / double-clicked.
      * ``[]``           -- when the user pressed X (delete polygon).
      * ``None``         -- on cancel (Esc / Q / window closed).
    """
    if canvas_w is None or canvas_h is None:
        sw, sh = _detect_screen_size()
        canvas_w = canvas_w or sw
        canvas_h = canvas_h or sh

    fh, fw = frame.shape[:2]
    canvas, scale, off_x, off_y = _letterbox(frame, canvas_w, canvas_h)
    img_x0, img_y0 = off_x, off_y
    img_x1 = off_x + int(round(fw * scale)) - 1
    img_y1 = off_y + int(round(fh * scale)) - 1
    last_good_frame = frame
    consecutive_read_failures = 0

    vertices_frame: list[tuple[int, int]] = list(existing_polygon or [])
    cross_x = (img_x0 + img_x1) // 2
    cross_y = (img_y0 + img_y1) // 2
    status = ""

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Initial fullscreen request -- we re-assert it every frame below
    # because some compositors (XWayland in particular) drop the flag
    # when the window first becomes mapped.
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    # Mouse state (set by callback, consumed in main loop)
    mouse_state: dict = {"x": cross_x, "y": cross_y,
                         "click": False, "rclick": False, "dblclk": False}

    def _on_mouse(event, x, y, flags, _param):  # noqa: ANN001
        if event == cv2.EVENT_MOUSEMOVE:
            mouse_state["x"] = x
            mouse_state["y"] = y
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            mouse_state["dblclk"] = True
        elif event == cv2.EVENT_LBUTTONDOWN:
            mouse_state["click"] = True
            mouse_state["x"] = x
            mouse_state["y"] = y
        elif event == cv2.EVENT_RBUTTONDOWN:
            mouse_state["rclick"] = True

    cv2.setMouseCallback(window_name, _on_mouse)

    # Console hint so the user knows their SSH terminal is now driving things
    if sys.stdin.isatty():
        print("[zone editor] terminal in cbreak mode -- press keys here "
              "(or click on the LCD window). 'F' to save, 'X' to delete, "
              "'Q'/Esc to cancel.")

    def _commit_at(disp_x: int, disp_y: int) -> None:
        fx, fy = _disp_to_frame(disp_x, disp_y, scale, img_x0, img_y0)
        fx = max(0, min(fw - 1, fx))
        fy = max(0, min(fh - 1, fy))
        vertices_frame.append((fx, fy))

    try:
        with RawStdin() as raw_in:
            while True:
                # Mouse may have moved the crosshair; clamp into image area.
                cross_x = max(img_x0, min(img_x1, mouse_state["x"]))
                cross_y = max(img_y0, min(img_y1, mouse_state["y"]))

                # Live mode: blit the latest camera frame into the
                # canvas image area each iteration. Coordinate system
                # (img_x0/img_y0/scale) is fixed because frame size
                # never changes mid-session.
                stream_lost = False
                if cap is not None:
                    ok, new_frame = cap.read()
                    if ok and new_frame is not None:
                        if new_frame.shape[:2] == (fh, fw):
                            last_good_frame = new_frame
                            consecutive_read_failures = 0
                            _paste_frame(canvas, new_frame, scale,
                                          img_x0, img_y0)
                        else:
                            # Resolution changed unexpectedly -- ignore
                            # the bad frame, keep last good one.
                            consecutive_read_failures += 1
                    else:
                        consecutive_read_failures += 1
                    if consecutive_read_failures >= 8:
                        stream_lost = True

                view = canvas.copy()
                verts_disp = [_frame_to_disp(fx, fy, scale, img_x0, img_y0)
                              for fx, fy in vertices_frame]
                _draw_overlay(view, verts_disp, (cross_x, cross_y),
                              slot_label,
                              "STREAM LOST" if stream_lost else status)
                cv2.imshow(window_name, view)
                # Re-assert fullscreen every frame -- some compositors
                # drop the flag once the window becomes mapped, which is
                # why the multi-camera feed does the same thing.
                try:
                    cv2.setWindowProperty(window_name,
                                          cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN)
                except cv2.error:
                    pass

                # ---- collect input from all sources ----------------
                key = normalise_cv2_key(cv2.waitKeyEx(15))
                if key is None:
                    key = raw_in.poll(0.0)

                # ---- mouse-driven commits / undo / finish ----------
                if mouse_state["dblclk"]:
                    mouse_state["dblclk"] = False
                    if len(vertices_frame) >= 3:
                        return list(vertices_frame)
                    if len(vertices_frame) >= 2:
                        # Treat double-click on a non-vertex as: commit
                        # current pos, then finish (still need >=3 total)
                        _commit_at(cross_x, cross_y)
                        if len(vertices_frame) >= 3:
                            return list(vertices_frame)
                    status = "need >= 3 vertices to save"
                if mouse_state["click"]:
                    mouse_state["click"] = False
                    _commit_at(cross_x, cross_y)
                    status = ""
                if mouse_state["rclick"]:
                    mouse_state["rclick"] = False
                    if vertices_frame:
                        vertices_frame.pop()
                    status = ""

                # ---- keyboard from either source -------------------
                if key is None:
                    # Window closed by WM
                    if cv2.getWindowProperty(window_name,
                                             cv2.WND_PROP_VISIBLE) < 1:
                        return None
                    continue

                status = ""
                # movement
                if key == "left":
                    mouse_state["x"] = cross_x - 1
                elif key == "right":
                    mouse_state["x"] = cross_x + 1
                elif key == "up":
                    mouse_state["y"] = cross_y - 1
                elif key == "down":
                    mouse_state["y"] = cross_y + 1
                elif key == "a":
                    mouse_state["x"] = cross_x - 10
                elif key == "d":
                    mouse_state["x"] = cross_x + 10
                elif key == "w":
                    mouse_state["y"] = cross_y - 10
                elif key == "s":
                    mouse_state["y"] = cross_y + 10
                # vertex ops
                elif key == "enter":
                    _commit_at(cross_x, cross_y)
                elif key == "backspace" or key == "u":
                    if vertices_frame:
                        vertices_frame.pop()
                elif key == "c":
                    vertices_frame.clear()
                # finalise
                elif key == "f":
                    if len(vertices_frame) >= 3:
                        return list(vertices_frame)
                    status = "need >= 3 vertices to save"
                elif key == "x":
                    return []
                elif key in ("esc", "q"):
                    return None
    finally:
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass
