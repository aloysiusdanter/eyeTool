"""Live brightness/contrast/saturation/gamma editor for one camera.

Renders an SSH-friendly fullscreen view that shows the current camera
frame **as the detector will see it** (i.e. with the live preprocessing
applied) plus a sidebar panel of the four sliders. Keys:

    1 / 2 / 3 / 4   select Brightness / Contrast / Saturation / Gamma
    UP / DOWN       same as 1..4 (cycle through parameters)
    LEFT / -        decrement the selected parameter by one step
    RIGHT / +       increment the selected parameter by one step
    R               reset selected parameter to its neutral value
    A               reset ALL parameters
    S / ENTER       save and exit
    Q / ESC         cancel and exit (no changes saved)

Driven by stdin in cbreak mode (works over plain SSH) and by the cv2
window when it has X11 keyboard focus, sharing the same dispatcher.
The frame rate of the editor is whatever the camera delivers; the
display itself never blocks waiting for input.

Returns a ``Preprocess`` instance on save, ``None`` on cancel.
"""

from __future__ import annotations

import cv2
import numpy as np

from preprocessing.preprocess import (BRIGHTNESS_RANGE, CONTRAST_RANGE, GAMMA_RANGE,
                        SATURATION_RANGE, Preprocess)
from utils.terminal_input import RawStdin, normalise_cv2_key

# UI constants
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_BG = (24, 24, 24)
_PANEL_BG = (12, 12, 12)
_TEXT = (240, 240, 240)
_TEXT_DIM = (160, 160, 160)
_HIGHLIGHT = (0, 200, 255)
_BAR_FILL = (71, 179, 255)
_BAR_TRACK = (60, 60, 60)


# (label, attr, range, step, neutral, neutral_keys)
_PARAMS = [
    ("Brightness", "brightness", BRIGHTNESS_RANGE, 0.05, 0.0),
    ("Contrast",   "contrast",   CONTRAST_RANGE,   0.05, 1.0),
    ("Saturation", "saturation", SATURATION_RANGE, 0.05, 1.0),
    ("Gamma",      "gamma",      GAMMA_RANGE,      0.05, 1.0),
]


def _detect_screen_size(default: tuple[int, int] = (800, 480)
                        ) -> tuple[int, int]:
    """Re-uses ``polygon_editor`` heuristic; imported lazily to keep this
    module independently importable in headless tests."""
    from ui.editors.polygon_editor import _detect_screen_size as _f
    return _f(default)


def _letterbox_into(canvas: np.ndarray, frame: np.ndarray,
                    x0: int, y0: int, x1: int, y1: int) -> None:
    """Fit *frame* into the rectangle (x0..x1, y0..y1) of *canvas*,
    preserving aspect ratio with black bars."""
    bw = x1 - x0
    bh = y1 - y0
    if bw <= 0 or bh <= 0:
        return
    fh, fw = frame.shape[:2]
    scale = min(bw / fw, bh / fh)
    new_w = max(1, int(round(fw * scale)))
    new_h = max(1, int(round(fh * scale)))
    resized = cv2.resize(frame, (new_w, new_h),
                         interpolation=cv2.INTER_LINEAR)
    canvas[y0:y1, x0:x1] = _BG  # clear region first
    ox = x0 + (bw - new_w) // 2
    oy = y0 + (bh - new_h) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = resized


def _draw_panel(canvas: np.ndarray, panel_x: int, params_state: list[float],
                selected: int, slot_label: str, dirty: bool,
                status: str) -> None:
    """Render the right-side parameter panel in-place onto *canvas*."""
    h, w = canvas.shape[:2]
    canvas[:, panel_x:w] = _PANEL_BG

    title = f"PREPROC  {slot_label}"
    cv2.putText(canvas, title, (panel_x + 12, 28), _FONT, 0.55,
                _TEXT, 1, cv2.LINE_AA)
    if dirty:
        cv2.putText(canvas, "*unsaved*", (panel_x + 12, 50), _FONT, 0.42,
                    _HIGHLIGHT, 1, cv2.LINE_AA)

    y = 90
    panel_w = w - panel_x
    bar_w = panel_w - 32
    bar_x0 = panel_x + 16
    for i, (label, _attr, (lo, hi), _step, _neutral) in enumerate(_PARAMS):
        val = params_state[i]
        sel = (i == selected)
        color = _HIGHLIGHT if sel else _TEXT
        prefix = "*" if sel else " "
        cv2.putText(canvas, f"{prefix} {i + 1}. {label}",
                    (panel_x + 12, y), _FONT, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{val:+.2f}",
                    (panel_x + panel_w - 76, y), _FONT, 0.5, color, 1,
                    cv2.LINE_AA)
        # bar
        track_y = y + 14
        cv2.rectangle(canvas, (bar_x0, track_y),
                      (bar_x0 + bar_w, track_y + 8), _BAR_TRACK, -1)
        ratio = (val - lo) / (hi - lo) if hi > lo else 0.0
        ratio = max(0.0, min(1.0, ratio))
        fill_x = bar_x0 + int(round(ratio * bar_w))
        cv2.rectangle(canvas, (bar_x0, track_y),
                      (fill_x, track_y + 8), _BAR_FILL, -1)
        # neutral tick
        neutral_ratio = (_neutral - lo) / (hi - lo) if hi > lo else 0.5
        tick_x = bar_x0 + int(round(neutral_ratio * bar_w))
        cv2.line(canvas, (tick_x, track_y - 2),
                 (tick_x, track_y + 10), _TEXT_DIM, 1)
        y += 50

    # Help footer (panel-bottom)
    help_lines = [
        "1-4 / UP DN  select",
        "+/-  LEFT/RT  adjust",
        "R reset one  A reset all",
        "S/ENTER save  Q cancel",
    ]
    yy = h - 20 - 18 * (len(help_lines) - 1)
    for line in help_lines:
        cv2.putText(canvas, line, (panel_x + 12, yy), _FONT, 0.42,
                    _TEXT_DIM, 1, cv2.LINE_AA)
        yy += 18

    if status:
        cv2.putText(canvas, status, (panel_x + 12, h - 80),
                    _FONT, 0.45, _HIGHLIGHT, 1, cv2.LINE_AA)


def run(cap: "cv2.VideoCapture", initial: Preprocess | None = None,
        slot_label: str = "",
        canvas_w: int | None = None,
        canvas_h: int | None = None,
        window_name: str = "eyeTool - Preprocess Editor",
        ) -> Preprocess | None:
    """Run the live editor on *cap*. Returns the final ``Preprocess`` on
    save (S/Enter), or ``None`` on cancel (Q/Esc/window-close)."""
    if canvas_w is None or canvas_h is None:
        sw, sh = _detect_screen_size()
        canvas_w = canvas_w or sw
        canvas_h = canvas_h or sh

    if initial is None:
        params = [neutral for _, _, _, _, neutral in _PARAMS]
    else:
        params = [getattr(initial, attr)
                  for _label, attr, _r, _s, _n in _PARAMS]
    selected = 0
    dirty = False
    status = ""

    panel_w = max(220, canvas_w // 4)
    image_x1 = canvas_w - panel_w

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1 \
            or True:  # always print -- helpful over SSH
        import sys
        if sys.stdin.isatty():
            print("[preproc editor] terminal in cbreak mode -- press keys "
                  "here. 'S' to save, 'Q'/Esc to cancel.")

    canvas = np.full((canvas_h, canvas_w, 3), _BG, dtype=np.uint8)
    last_good = None
    consecutive_failures = 0

    def _build_pp() -> Preprocess:
        return Preprocess(brightness=params[0], contrast=params[1],
                          saturation=params[2], gamma=params[3])

    pp = _build_pp()

    try:
        with RawStdin() as raw_in:
            while True:
                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    last_good = frame
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1

                if last_good is None:
                    canvas[:] = _BG
                    cv2.putText(canvas, "waiting for first frame...",
                                (40, canvas_h // 2), _FONT, 0.7,
                                _TEXT, 1, cv2.LINE_AA)
                else:
                    processed = pp(last_good)
                    _letterbox_into(canvas, processed, 0, 0,
                                    image_x1, canvas_h)

                stream_status = ("STREAM LOST" if consecutive_failures > 8
                                 else status)
                _draw_panel(canvas, image_x1, params, selected,
                            slot_label, dirty, stream_status)
                cv2.imshow(window_name, canvas)
                try:
                    cv2.setWindowProperty(window_name,
                                          cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN)
                except cv2.error:
                    pass

                key = normalise_cv2_key(cv2.waitKeyEx(10))
                if key is None:
                    key = raw_in.poll(0.0)
                if key is None:
                    if cv2.getWindowProperty(window_name,
                                             cv2.WND_PROP_VISIBLE) < 1:
                        return None
                    continue
                status = ""

                if key in ("1", "2", "3", "4"):
                    selected = int(key) - 1
                elif key == "up":
                    selected = (selected - 1) % len(_PARAMS)
                elif key == "down":
                    selected = (selected + 1) % len(_PARAMS)
                elif key in ("plus", "right"):
                    _label, _attr, (lo, hi), step, _n = _PARAMS[selected]
                    params[selected] = max(lo, min(hi, params[selected] + step))
                    dirty = True
                    pp = _build_pp()
                elif key in ("minus", "left"):
                    _label, _attr, (lo, hi), step, _n = _PARAMS[selected]
                    params[selected] = max(lo, min(hi, params[selected] - step))
                    dirty = True
                    pp = _build_pp()
                elif key == "r":
                    _label, _attr, _r, _s, neutral = _PARAMS[selected]
                    params[selected] = neutral
                    dirty = True
                    pp = _build_pp()
                elif key == "a":
                    params = [neutral for _, _, _, _, neutral in _PARAMS]
                    dirty = True
                    pp = _build_pp()
                elif key in ("s", "enter"):
                    return pp
                elif key in ("esc", "q"):
                    return None
    finally:
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass
