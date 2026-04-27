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
    ("11", "Exit"),
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
        start_x = x + (width - menu_width) // 2
        start_y = y + 3

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
            return "multi_feed"  # Multi-camera feed
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
