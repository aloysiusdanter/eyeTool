"""Passive curses dashboard for the eyeTool runtime.

Watches:

* ``zones.json`` (slot bindings + polygons + preprocessing) -- via the
  shared ``Config`` reload, refreshed every tick.
* connected USB cameras                                     -- via
  ``hotplug.list_cameras``.
* attach / detach transitions                               -- inferred
  by diffing the camera set between ticks; surfaced in a rolling event
  log at the bottom of the screen.

Doesn't open any video device (no V4L2 contention with the multi-camera
feed). Safe to run in a second SSH window while the main app is live.

Controls
--------
``q`` / ``Q`` / ``Ctrl-C``  exit
``r`` / ``R``               force-reload zones.json
"""

from __future__ import annotations

import curses
import time
from collections import deque
from datetime import datetime

from config import get_config
from hotplug import list_cameras

REFRESH_HZ = 5
EVENT_LOG_LEN = 12


def _stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _safe_addstr(win, y: int, x: int, text: str, attr: int = 0) -> None:
    """``addstr`` that silently truncates instead of raising on the
    bottom-right cell (a known curses quirk)."""
    h, w = win.getmaxyx()
    if y < 0 or y >= h or x >= w:
        return
    text = text[: max(0, w - x - 1)]
    try:
        win.addstr(y, x, text, attr)
    except curses.error:
        pass


def _draw(stdscr, *, slots_state: dict, cams: dict, events: deque,
          cfg, last_reload_ts: float) -> None:
    stdscr.erase()
    h, w = stdscr.getmaxyx()

    # --- header ---------------------------------------------------------
    title = " eyeTool monitor "
    _safe_addstr(stdscr, 0, 0, "=" * w, curses.A_DIM)
    _safe_addstr(stdscr, 0, max(2, (w - len(title)) // 2),
                 title, curses.A_BOLD | curses.A_REVERSE)
    _safe_addstr(stdscr, 0, max(0, w - 11), f" {_stamp()} ", curses.A_DIM)

    row = 2
    _safe_addstr(stdscr, row, 0, "Slots:", curses.A_BOLD)
    row += 1
    _safe_addstr(stdscr, row, 2,
                 f"{'ID':>2}  {'state':<10} {'model':<18} "
                 f"{'devnode':<14} {'polygon':<10} {'preproc'}",
                 curses.A_DIM)
    row += 1
    max_streams = int(cfg.get("streams.max_streams", 4))
    for sid in range(max_streams):
        if row >= h - 6:
            break
        slot_cfg = cfg.slot(sid) or {}
        port_path = slot_cfg.get("port_path", "")
        cam = cams.get(port_path) if port_path else None
        if not port_path:
            state, model, devnode = "empty", "-", "-"
            colour = curses.A_DIM
        elif cam is None:
            state, model, devnode = "UNAVAIL", "-", "-"
            colour = curses.color_pair(2) | curses.A_BOLD
        else:
            state = "ACTIVE"
            model = (cam.model or "")[:18]
            devnode = cam.devnode
            colour = curses.color_pair(1)
        poly = slot_cfg.get("polygon") or []
        poly_str = f"{len(poly)} verts" if poly else "-"
        pre = slot_cfg.get("preprocessing")
        if not pre:
            pre_str = "default"
        else:
            pre_str = (f"B{pre.get('brightness', 0):+.1f} "
                       f"C{pre.get('contrast', 1):.1f} "
                       f"S{pre.get('saturation', 1):.1f} "
                       f"G{pre.get('gamma', 1):.1f}")
        line = (f"{sid:>2}  {state:<10} {model:<18} "
                f"{devnode:<14} {poly_str:<10} {pre_str}")
        _safe_addstr(stdscr, row, 2, line, colour)
        row += 1

    # --- detected cameras ----------------------------------------------
    row += 1
    if row < h - 4:
        _safe_addstr(stdscr, row, 0,
                     f"Detected cameras ({len(cams)}):", curses.A_BOLD)
        row += 1
        for cam in sorted(cams.values(), key=lambda c: c.devnode):
            if row >= h - 4:
                break
            _safe_addstr(stdscr, row, 2,
                         f"{cam.devnode:<14}  {cam.model or '?':<20}  "
                         f"@ {cam.port_path}",
                         curses.color_pair(1))
            row += 1

    # --- config summary -------------------------------------------------
    row += 1
    if row < h - 4:
        _safe_addstr(stdscr, row, 0, "Config:", curses.A_BOLD)
        row += 1
        _safe_addstr(stdscr, row, 2,
                     f"manufacturer zones: "
                     f"{'present' if cfg.has_manufacturer_zones() else 'absent'}")
        row += 1
        _safe_addstr(stdscr, row, 2,
                     f"zones.json: {cfg.zones_path}")
        row += 1

    # --- footer / event log --------------------------------------------
    log_top = max(row + 1, h - 6)
    _safe_addstr(stdscr, log_top, 0, "Recent events:", curses.A_BOLD)
    for i, ev in enumerate(list(events)[-(h - log_top - 2):]):
        if log_top + 1 + i >= h - 1:
            break
        _safe_addstr(stdscr, log_top + 1 + i, 2, ev)

    _safe_addstr(stdscr, h - 1, 0,
                 " q quit  r reload-config ", curses.A_REVERSE)
    stdscr.refresh()


def _diff_cameras(prev: dict, curr: dict) -> list[str]:
    out: list[str] = []
    for pp, cam in curr.items():
        if pp not in prev:
            out.append(f"{_stamp()}  ATTACH  {cam.devnode}  "
                       f"{cam.model or '?'}  @ {pp}")
    for pp, cam in prev.items():
        if pp not in curr:
            out.append(f"{_stamp()}  DETACH  {cam.devnode}  "
                       f"(was @ {pp})")
    return out


def _loop(stdscr) -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(int(1000 / REFRESH_HZ))

    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)
        curses.init_pair(2, curses.COLOR_RED, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)

    cfg = get_config()
    events: deque[str] = deque(maxlen=EVENT_LOG_LEN)
    prev_cams: dict = {}
    last_reload = time.monotonic()

    while True:
        # auto-reload config every ~2 s so external edits become visible
        now = time.monotonic()
        if now - last_reload > 2.0:
            cfg.reload()
            last_reload = now

        cams = {c.port_path: c for c in list_cameras()}
        for ev in _diff_cameras(prev_cams, cams):
            events.append(ev)
        prev_cams = cams

        _draw(stdscr, slots_state={}, cams=cams, events=events,
              cfg=cfg, last_reload_ts=last_reload)

        try:
            ch = stdscr.getch()
        except KeyboardInterrupt:
            return
        if ch in (ord("q"), ord("Q"), 27):
            return
        if ch in (ord("r"), ord("R")):
            cfg.reload()
            last_reload = time.monotonic()
            events.append(f"{_stamp()}  config reloaded")


def run() -> None:
    """Entry point. Wraps ``curses.wrapper`` for proper teardown."""
    try:
        curses.wrapper(_loop)
    except KeyboardInterrupt:
        pass
