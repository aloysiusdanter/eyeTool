"""Mode classes for the htop-style TUI.

Each mode represents a different screen in the TUI (main menu, sub-menus, etc.).
"""

from __future__ import annotations

import curses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ui.tui.panels import EventLog

# Main menu options (from existing main.py interactive_menu)
MAIN_MENU_OPTIONS = [
    ("1", "Live camera feed"),
    ("2", "Multi-camera feed"),
    ("3", "Setup alarm zones"),
    ("4", "Image preprocessing"),
    ("5", "Monitoring TUI"),
    ("6", "Configuration"),
    ("7", "Capture single image"),
    ("8", "Probe camera"),
    ("9", "Select display target"),
    ("10", "Detection settings"),
    ("11", "Recording settings"),
    ("12", "Exit"),
]


class Mode:
    """Base class for TUI modes."""

    def __init__(self, name: str):
        """Initialize the mode.

        Args:
            name: Mode name/identifier
        """
        self.name = name
        self.selected_index = 0

    def render(self, stdscr: curses.window, y: int, x: int, width: int, height: int) -> None:
        """Render the mode's content to the main panel.

        Args:
            stdscr: Curses window
            y: Y position of the main panel
            x: X position of the main panel
            width: Width of the main panel
            height: Height of the main panel
        """
        raise NotImplementedError

    def handle_input(self, key: int) -> str | None:
        """Handle keyboard input for this mode.

        Args:
            key: The key code

        Returns:
            New mode name to switch to, or None to stay in current mode
        """
        raise NotImplementedError

    def get_help(self) -> str:
        """Return help text for this mode."""
        return "q/ESC Quit"


class MainMenuMode(Mode):
    """Main menu mode with 11 menu options."""

    def __init__(self):
        """Initialize the main menu mode."""
        super().__init__("main_menu")
        self.selected_index = 0
        self.options = MAIN_MENU_OPTIONS

    def render(self, stdscr: curses.window, y: int, x: int, width: int, height: int) -> None:
        """Render the main menu options.

        Args:
            stdscr: Curses window
            y: Y position of the main panel
            x: X position of the main panel
            width: Width of the main panel
            height: Height of the main panel
        """
        # Calculate starting position to center the menu
        menu_width = max(len(option[1]) for option in self.options)
        start_x = x + (width - menu_width - 50) // 2  # Leave space for parameters on right
        start_y = y + 3

        # Fetch parameters
        try:
            from ui.menus import get_config, detection_enabled, detection_confidence, detect_every_n, use_multi_core
            from core.hotplug import list_cameras
            cfg = get_config()
            display_w = int(cfg.get("display.width", 800))
            display_h = int(cfg.get("display.height", 480))
            target_fps = int(cfg.get("display.target_fps", 30))
            max_streams = int(cfg.get("streams.max_streams", 4))
            cameras = list_cameras()
            camera_count = len(cameras)
            det_str = "ON" if detection_enabled else "OFF"
            conf_str = f"{detection_confidence:.2f}"
            every_str = f"{detect_every_n}"
            npu_str = "3-CORE" if use_multi_core else "1-CORE"
            rec_enabled = cfg.get("recording.enabled", True)
            rec_str = "ON" if rec_enabled else "OFF"
            rec_seg = cfg.get("recording.segment_duration_min", 2)
            rec_codec = cfg.get("recording.codec", "mp4v")
        except Exception:
            display_w, display_h, target_fps, max_streams = 800, 480, 30, 4
            camera_count = 0
            det_str, conf_str, every_str, npu_str = "OFF", "0.50", "1", "1-CORE"
            rec_str, rec_seg, rec_codec = "ON", 2, "mp4v"

        # Render parameters on the right side
        param_x = start_x + menu_width + 10
        param_y = start_y
        param_lines = [
            "Multi-Camera Parameters",
            "",
            f"Display: {display_w}x{display_h}",
            f"Target FPS: {target_fps}",
            f"Max Streams: {max_streams}",
            "",
            f"Cameras: {camera_count}",
            f"Detection: {det_str}",
            f"Confidence: {conf_str}",
            f"Every-N: {every_str}",
            f"NPU: {npu_str}",
            "",
            "Recording",
            f"Status: {rec_str}",
            f"Segment: {rec_seg}min",
            f"Codec: {rec_codec}",
        ]

        try:
            # Draw separator
            sep_x = start_x + menu_width + 5
            for i in range(height - 5):
                stdscr.addch(start_y + i, sep_x, curses.ACS_VLINE, curses.color_pair(5))

            # Draw parameters
            for i, line in enumerate(param_lines):
                line_y = param_y + i
                if line_y >= y + height - 2:
                    break
                if i == 0:  # Title
                    stdscr.addstr(line_y, param_x, line, curses.color_pair(1) | curses.A_BOLD)
                else:
                    stdscr.addstr(line_y, param_x, line, curses.color_pair(5))
        except curses.error:
            pass

        # Render menu options
        try:
            for i, (key, label) in enumerate(self.options):
                line_y = start_y + i
                if line_y >= y + height - 1:
                    break

                if i == self.selected_index:
                    stdscr.addstr(line_y, start_x - 2, "► ", curses.color_pair(1))
                    stdscr.addstr(line_y, start_x, f"{key}. {label}", curses.color_pair(1) | curses.A_BOLD)
                else:
                    stdscr.addstr(line_y, start_x - 2, "  ", curses.color_pair(5))
                    stdscr.addstr(line_y, start_x, f"{key}. {label}", curses.color_pair(5))
        except curses.error:
            pass

    def handle_input(self, key: int) -> str | None:
        """Handle keyboard input for the main menu.

        Args:
            key: The key code

        Returns:
            New mode name to switch to, or None to stay in current mode
        """
        if key == curses.KEY_UP:
            self.selected_index = max(0, self.selected_index - 1)
        elif key == curses.KEY_DOWN:
            self.selected_index = min(len(self.options) - 1, self.selected_index + 1)
        elif key == curses.KEY_ENTER or key == 10:  # 10 = Enter
            # Handle selection
            selected = self.options[self.selected_index][0]
            return self._handle_selection(selected)
        elif key == 27:  # ESC
            return None  # Stay in main menu
        elif key in (ord("q"), ord("Q")):
            return "quit"

        return None

    def _handle_selection(self, selected: str) -> str | None:
        """Handle menu item selection.

        Args:
            selected: The selected menu option key

        Returns:
            New mode name to switch to, or None to stay in current mode
        """
        # Phase 9: Integration with main.py will wire these to actual functions
        # For now, just return mode names
        if selected == "1":
            return "feed"  # Live camera feed
        elif selected == "2":
            return "multi_feed"  # Multi-camera feed (shows parameters, then launches)
        elif selected == "3":
            return "zones_menu"  # Setup alarm zones
        elif selected == "4":
            return "preprocess_menu"  # Image preprocessing
        elif selected == "5":
            return "monitor"  # Monitoring TUI
        elif selected == "6":
            return "config_menu"  # Configuration
        elif selected == "7":
            return "capture"  # Capture single image
        elif selected == "8":
            return "probe"  # Probe camera
        elif selected == "9":
            return "display_menu"  # Select display target
        elif selected == "10":
            return "detection_menu"  # Detection settings
        elif selected == "11":
            return "recording_menu"  # Recording settings
        elif selected == "12":
            return "quit"  # Exit

        return None

    def get_help(self) -> str:
        """Return help text for the main menu."""
        return "↑↓ Navigate | Enter Select | q/ESC Quit"


class StatusMode(Mode):
    """Status mode for showing messages when OpenCV features are active."""

    def __init__(self, message: str, return_mode: str = "main_menu"):
        """Initialize the status mode.

        Args:
            message: Status message to display
            return_mode: Mode to return to after feature completes
        """
        super().__init__("status")
        self.message = message
        self.return_mode = return_mode

    def render(self, stdscr: curses.window, y: int, x: int, width: int, height: int) -> None:
        """Render the status message.

        Args:
            stdscr: Curses window
            y: Y position of the main panel
            x: X position of the main panel
            width: Width of the main panel
            height: Height of the main panel
        """
        # Center the message
        message_lines = self.message.split("\n")
        start_y = y + (height - len(message_lines)) // 2
        start_x = x + (width - len(self.message)) // 2

        try:
            for i, line in enumerate(message_lines):
                line_y = start_y + i
                if line_y >= y + height - 1:
                    break
                stdscr.addstr(line_y, start_x, line, curses.color_pair(3) | curses.A_BOLD)

            instruction = "Press 'q' to return to menu"
            stdscr.addstr(y + height - 2, x + (width - len(instruction)) // 2,
                         instruction, curses.color_pair(5))
        except curses.error:
            pass

    def handle_input(self, key: int) -> str | None:
        """Handle keyboard input for status mode.

        Args:
            key: The key code

        Returns:
            Return mode name on 'q', None otherwise
        """
        if key in (ord("q"), ord("Q"), 27):  # 27 = ESC
            return self.return_mode
        return None

    def get_help(self) -> str:
        """Return help text for status mode."""
        return "q/ESC Return to Menu"
