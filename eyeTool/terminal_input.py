"""Shared SSH-friendly keyboard input for full-screen cv2 editors.

The polygon editor and the preprocessing editor both need to read keys
from the controlling SSH terminal because cv2.waitKey only fires when
the OpenCV window has X11 keyboard focus -- which it never gets over
plain SSH. ``RawStdin`` solves that by putting the terminal into
cbreak mode for the duration of a ``with`` block, and exposing a
non-blocking ``poll(timeout)`` that returns the next key.

Returned events use a flat string vocabulary that maps cleanly to
``cv2.waitKeyEx`` codes too (see ``normalise_cv2_key``). Both editors
can therefore feed both input sources through one dispatch table.

Vocabulary
----------
``"left"``, ``"right"``, ``"up"``, ``"down"``, ``"enter"``, ``"esc"``,
``"backspace"``, ``"plus"``, ``"minus"``, or a single lowercase
character like ``"a"`` / ``"f"`` / ``"q"``.
"""

from __future__ import annotations

import os
import select
import sys
import termios
import tty


# X11 / Linux extended key codes returned by ``cv2.waitKeyEx``.
_KEY_LEFT = 65361
_KEY_UP = 65362
_KEY_RIGHT = 65363
_KEY_DOWN = 65364
_KEY_ENTER = 13
_KEY_SPACE = 32
_KEY_ESC = 27
_KEY_BACKSPACE = 65288
_KEY_BACKSPACE2 = 8


class RawStdin:
    """Put stdin in cbreak mode for the duration of a ``with`` block."""

    def __init__(self) -> None:
        self._fd = -1
        self._old: list | None = None
        self._enabled = False

    def __enter__(self):
        if not sys.stdin.isatty():
            return self
        try:
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            self._enabled = True
        except (termios.error, OSError, ValueError):
            self._enabled = False
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._enabled and self._old is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            except (termios.error, OSError):
                pass
        try:
            termios.tcflush(self._fd, termios.TCIFLUSH)
        except (termios.error, OSError, ValueError):
            pass

    def poll(self, timeout: float) -> str | None:
        if not self._enabled:
            return None
        try:
            r, _, _ = select.select([self._fd], [], [], timeout)
        except (OSError, ValueError):
            return None
        if not r:
            return None
        try:
            ch = os.read(self._fd, 1)
        except OSError:
            return None
        if not ch:
            return None
        c = ch.decode("utf-8", errors="ignore")
        if c == "\x1b":
            r, _, _ = select.select([self._fd], [], [], 0.02)
            if not r:
                return "esc"
            seq = os.read(self._fd, 2).decode("utf-8", errors="ignore")
            if seq.startswith("[") and len(seq) >= 2:
                return {"A": "up", "B": "down",
                        "C": "right", "D": "left"}.get(seq[1], "esc")
            return "esc"
        if c in ("\r", "\n"):
            return "enter"
        if c in ("\x7f", "\x08"):
            return "backspace"
        if c == " ":
            return "enter"
        if c == "+":
            return "plus"
        if c == "-":
            return "minus"
        if c == "=":
            return "plus"   # convenience: '=' on US layout = '+' without shift
        if c.isprintable():
            return c.lower()
        return None


def normalise_cv2_key(key: int) -> str | None:
    """Translate a ``cv2.waitKeyEx`` return code into the same flat string
    vocabulary used by ``RawStdin.poll``."""
    if key == -1:
        return None
    if key == _KEY_LEFT:
        return "left"
    if key == _KEY_RIGHT:
        return "right"
    if key == _KEY_UP:
        return "up"
    if key == _KEY_DOWN:
        return "down"
    if key in (_KEY_ENTER, _KEY_SPACE):
        return "enter"
    if key in (_KEY_BACKSPACE, _KEY_BACKSPACE2):
        return "backspace"
    if key == _KEY_ESC:
        return "esc"
    low = key & 0xFF
    if low == ord("+") or low == ord("="):
        return "plus"
    if low == ord("-"):
        return "minus"
    if 32 <= low < 127:
        return chr(low).lower()
    return None
