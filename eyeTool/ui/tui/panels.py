"""Panel rendering for the htop-style TUI.

Provides functions to draw the top panel (camera status, metrics) and bottom panel
(event log, help) with consistent styling.
"""

from __future__ import annotations

import curses
from collections import deque
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ui.tui.layout import Layout

EVENT_LOG_LEN = 12


class EventLog:
    """Rolling event log for the bottom panel."""

    def __init__(self, max_len: int = EVENT_LOG_LEN):
        """Initialize the event log.

        Args:
            max_len: Maximum number of events to keep
        """
        self.max_len = max_len
        self.events: deque[tuple[str, str]] = deque(maxlen=max_len)

    def add(self, level: str, message: str) -> None:
        """Add an event to the log.

        Args:
            level: Log level (INFO, WARN, ERROR)
            message: Log message
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.events.append((timestamp, f"[{level}] {message}"))

    def get_recent(self, count: int | None = None) -> list[tuple[str, str]]:
        """Get the most recent events.

        Args:
            count: Number of events to return (None for all)

        Returns:
            List of (timestamp, message) tuples
        """
        if count is None:
            return list(self.events)
        return list(self.events)[-count:]


def _draw_border(stdscr: curses.window, y: int, x: int, width: int, height: int) -> None:
    """Draw a box border.

    Args:
        stdscr: Curses window
        y: Top Y position
        x: Left X position
        width: Width of the box
        height: Height of the box
    """
    max_y, max_x = stdscr.getmaxyx()
    
    # Don't draw if out of bounds
    if y >= max_y or x >= max_x or y + height > max_y or x + width > max_x:
        return
    
    try:
        stdscr.hline(y, x, curses.ACS_HLINE, width)
        stdscr.addstr(y, x, "┌", curses.ACS_ULCORNER)
        stdscr.addstr(y, x + width - 1, "┐", curses.ACS_URCORNER)
        stdscr.vline(y + 1, x, curses.ACS_VLINE, height - 1)
        stdscr.vline(y + 1, x + width - 1, curses.ACS_VLINE, height - 1)
        stdscr.hline(y + height - 1, x, curses.ACS_HLINE, width)
        stdscr.addstr(y + height - 1, x, "└", curses.ACS_LLCORNER)
        stdscr.addstr(y + height - 1, x + width - 1, "┘", curses.ACS_LRCORNER)
    except curses.error:
        pass  # Ignore drawing errors


def draw_top_panel(stdscr: curses.window, layout: Layout, event_log: EventLog) -> None:
    """Draw the top panel with camera status, slot bindings, and detection metrics.

    Args:
        stdscr: Curses window
        layout: Layout object
        event_log: Event log for status updates
    """
    height = layout.top_panel_height()
    y = layout.top_panel_y()
    width = layout.width

    # Draw border
    _draw_border(stdscr, y, 0, width, height)

    # Fetch camera status
    try:
        from core.hotplug import list_cameras
        cameras = list_cameras()
        camera_count = len(cameras)
        camera_str = f"Cameras: {camera_count} connected"
    except Exception:  # noqa: BLE001
        camera_str = "Cameras: [error]"

    # Read live detection state from ui.menus
    try:
        from ui.menus import detection_enabled, detection_confidence, detect_every_n, use_multi_core
        det_str = "ON" if detection_enabled else "OFF"
        conf_str = f"{detection_confidence:.2f}"
        every_str = f"{detect_every_n}"
        npu_str = "3-CORE" if use_multi_core else "1-CORE"
    except Exception:  # noqa: BLE001
        det_str, conf_str, every_str, npu_str = "OFF", "0.50", "1", "1-CORE"

    # Read display config
    try:
        from ui.menus import get_config
        cfg = get_config()
        display_w = int(cfg.get("display.width", 800))
        display_h = int(cfg.get("display.height", 480))
        target_fps = int(cfg.get("display.target_fps", 30))
        max_streams = int(cfg.get("streams.max_streams", 4))
        res_str = f"{display_w}x{display_h}"
        fps_str = f"{target_fps}"
    except Exception:  # noqa: BLE001
        res_str, fps_str, max_streams = "800x480", "30", 4

    # Draw content (all wrapped for safety)
    try:
        stdscr.addstr(y + 1, 2, "eyeTool - Camera Status & Metrics", curses.color_pair(1))
        stdscr.addstr(y + 2, 2, camera_str, curses.color_pair(5))
        stdscr.addstr(y + 3, 2, f"Detection: {det_str} | Conf: {conf_str} | Every-N: {every_str} | NPU: {npu_str}", curses.color_pair(5))
        stdscr.addstr(y + 4, 2, f"Display: {res_str} | Target FPS: {fps_str} | Max Streams: {max_streams}", curses.color_pair(5))
    except curses.error:
        pass


def draw_bottom_panel(stdscr: curses.window, layout: Layout, event_log: EventLog, help_text: str) -> None:
    """Draw the bottom panel with event log and context-sensitive help.

    Args:
        stdscr: Curses window
        layout: Layout object
        event_log: Event log with recent events
        help_text: Context-sensitive help text for current mode
    """
    height = layout.bottom_panel_height()
    y = layout.bottom_panel_y()
    width = layout.width

    # Draw border
    _draw_border(stdscr, y, 0, width, height)

    # Split panel: left for event log, right for help
    split_x = width // 2

    try:
        # Draw section divider
        stdscr.vline(y + 1, split_x, curses.ACS_VLINE, height - 1)
        stdscr.addstr(y, split_x, "┬", curses.ACS_TTEE)
        stdscr.addstr(y + height - 1, split_x, "┴", curses.ACS_BTEE)

        # Draw event log title
        stdscr.addstr(y + 1, 2, "Event Log", curses.color_pair(3))

        # Draw help title
        stdscr.addstr(y + 1, split_x + 2, "Help", curses.color_pair(3))

        # Draw recent events
        events = event_log.get_recent(height - 3)
        for i, (timestamp, message) in enumerate(events):
            line_y = y + 2 + i
            if line_y >= y + height - 1:
                break
            max_msg_len = split_x - len(timestamp) - 5
            if len(message) > max_msg_len:
                message = message[:max_msg_len - 3] + "..."
            stdscr.addstr(line_y, 2, f"{timestamp} {message}", curses.color_pair(5))

        # Draw help text
        help_lines = help_text.split("\n")
        for i, line in enumerate(help_lines):
            line_y = y + 2 + i
            if line_y >= y + height - 1:
                break
            max_len = width - split_x - 4
            if len(line) > max_len:
                line = line[:max_len - 3] + "..."
            stdscr.addstr(line_y, split_x + 2, line, curses.color_pair(5))
    except curses.error:
        pass


def draw_main_panel_border(stdscr: curses.window, layout: Layout, title: str) -> None:
    """Draw the main panel border with title.

    Args:
        stdscr: Curses window
        layout: Layout object
        title: Panel title
    """
    y = layout.main_panel_y()
    height = layout.main_panel_height()
    width = layout.width

    # Draw border
    _draw_border(stdscr, y, 0, width, height)

    try:
        stdscr.addstr(y + 1, 2, title, curses.color_pair(4))
    except curses.error:
        pass
