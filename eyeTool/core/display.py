"""X11 display detection and configuration for eyeTool.

Provides functions to:
- Detect available X displays
- Set DISPLAY environment variable
- Handle XWayland authentication for SSH
- Interactive display selection menu
"""

from __future__ import annotations

import glob
import os
import re
import subprocess

from core.config import get_config


def _display_index(display: str | None = None) -> int | None:
    raw = display if display is not None else os.environ.get("DISPLAY", "")
    m = re.search(r":(\d+)", raw or "")
    if not m:
        return None
    return int(m.group(1))


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


def _can_query_display(display: str) -> bool:
    old_display = os.environ.get("DISPLAY")
    old_xauthority = os.environ.get("XAUTHORITY")
    set_display(display)
    try:
        result = subprocess.run(
            ["xrandr", "--query"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False
    finally:
        if old_display is None:
            os.environ.pop("DISPLAY", None)
        else:
            os.environ["DISPLAY"] = old_display
        if old_xauthority is None:
            os.environ.pop("XAUTHORITY", None)
        else:
            os.environ["XAUTHORITY"] = old_xauthority


def _choose_usable_display(candidates: list[str]) -> str | None:
    for display in candidates:
        if _can_query_display(display):
            return display
    return None


def auto_set_display() -> str | None:
    """Choose a DISPLAY for this run.

    Precedence (highest first):
      1. ``display.target`` in ``user_settings.json`` / manufacturer
         default -- the user's persistent preference.
      2. ``--display`` CLI flag (handled by main.py before calling this).
      3. ``DISPLAY`` already exported by the parent shell.
      4. First X11 socket found in ``/tmp/.X11-unix/`` -- typically ``:0``.

    Returns the display string actually set, or ``None`` if nothing
    suitable was found.
    """
    # 1. saved preference (highest priority)
    try:
        saved = (get_config().get("display.target", "") or "").strip()
    except Exception:  # noqa: BLE001 -- config init should never fail us here
        saved = ""
    available = detect_x_displays()
    if saved:
        if saved in available and _can_query_display(saved):
            set_display(saved)
            print(f"Display target: {saved} (saved preference)")
            return saved
        print(f"[main] saved display preference {saved!r} is not usable; "
              f"available={available}; falling back")

    # 2. environment DISPLAY (if saved preference not set or invalid)
    env_display = os.environ.get("DISPLAY")
    if env_display and _can_query_display(env_display):
        set_display(env_display)
        print(f"Using environment DISPLAY: {env_display}")
        return None

    # 3. first detected socket
    chosen = _choose_usable_display(available)
    if chosen is None:
        return None
    set_display(chosen)
    print(f"Auto-detected display: {chosen} "
          f"(use menu option 9 or --display to change)")
    return chosen


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


def get_display_geometry(default: tuple[int, int, int, int] = (0, 0, 800, 480)
                         ) -> tuple[int, int, int, int]:
    """Return ``(x, y, width, height)`` for the selected monitor output."""
    if not os.environ.get("DISPLAY"):
        return default
    try:
        result = subprocess.run(
            ["xrandr", "--query"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return default
    if result.returncode != 0:
        return default

    outputs: list[dict[str, object]] = []
    for line in result.stdout.splitlines():
        if line and not line.startswith((" ", "\t")):
            parts = line.split()
            if len(parts) < 2 or parts[1] != "connected":
                continue
            m = re.search(r"\b(\d+)x(\d+)\+(\d+)\+(\d+)\b", line)
            if not m:
                continue
            outputs.append({
                "name": parts[0],
                "primary": " primary " in f" {line} ",
                "width": int(m.group(1)),
                "height": int(m.group(2)),
                "x": int(m.group(3)),
                "y": int(m.group(4)),
            })

    if not outputs:
        m = re.search(r"current\s+(\d+)\s*x\s*(\d+)", result.stdout)
        if m:
            return 0, 0, int(m.group(1)), int(m.group(2))
        return default

    display_idx = _display_index()
    chosen = None
    if display_idx == 1:
        chosen = next((o for o in outputs if str(o["name"]).upper().startswith("HDMI")), None)
    elif display_idx == 0:
        chosen = next((o for o in outputs if bool(o["primary"])), None)
        if chosen is None:
            chosen = next((o for o in outputs if str(o["name"]).upper().startswith(("DSI", "LVDS", "EDP"))), None)
    if chosen is None:
        chosen = next((o for o in outputs if bool(o["primary"])), outputs[0])

    return int(chosen["x"]), int(chosen["y"]), int(chosen["width"]), int(chosen["height"])


def get_display_resolution(default: tuple[int, int] = (800, 480)) -> tuple[int, int]:
    """Return the selected monitor resolution as ``(width, height)``."""
    _x, _y, width, height = get_display_geometry((0, 0, default[0], default[1]))
    return width, height


def create_fullscreen_window(window_name: str) -> None:
    import cv2

    x, y, width, height = get_display_geometry()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, x, y)
    cv2.resizeWindow(window_name, width, height)
    set_fullscreen(window_name)


def set_fullscreen(window_name: str) -> None:
    import cv2

    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
    except cv2.error:
        pass


def check_display() -> bool:
    """Return True if DISPLAY is set; print an error and return False otherwise."""
    if not os.environ.get("DISPLAY"):
        print("Error: No X display available.")
        print("Hints:")
        print("  - Run menu option 9 to select a display target.")
        print("  - Or pass --display :0 on the command line.")
        print("  - On the NanoPi display, run: xhost +local:")
        return False
    return True
