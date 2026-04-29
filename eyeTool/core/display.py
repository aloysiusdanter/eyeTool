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
import subprocess

from core.config import get_config


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
    if saved:
        available = detect_x_displays()
        if not available or saved in available:
            set_display(saved)
            print(f"Display target: {saved} (saved preference)")
            return saved
        print(f"[main] saved display preference {saved!r} not present "
              f"on this system; available={available}; falling back")

    # 2. environment DISPLAY (if saved preference not set or invalid)
    if os.environ.get("DISPLAY"):
        print(f"Using environment DISPLAY: {os.environ['DISPLAY']}")
        return None

    # 3. first detected socket
    displays = detect_x_displays()
    if not displays:
        return None
    chosen = displays[0]
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
