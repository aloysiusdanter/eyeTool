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

    def handle_input(self, key: int) -> str | None | Mode:
        """Handle keyboard input for this mode.

        Args:
            key: The key code

        Returns:
            New mode object to switch to, "back"/"quit" string, or None to stay
        """
        raise NotImplementedError

    def get_help(self) -> str:
        """Return help text for this mode."""
        return "q/ESC Quit"


class InputComponent:
    """Reusable input field component for TUI modes."""

    def __init__(self, prompt: str = "> "):
        """Initialize the input component.

        Args:
            prompt: Prompt text to display before input
        """
        self.prompt = prompt
        self.buffer: str = ""
        self.cursor_pos: int = 0

    def handle_key(self, key: int) -> str | None:
        """Handle a keystroke for the input field.

        Args:
            key: The key code

        Returns:
            Input string if Enter pressed, None otherwise
        """
        if key == curses.KEY_ENTER or key == 10:  # Enter
            result = self.buffer
            self.clear()
            return result if result else None
        elif key == curses.KEY_BACKSPACE or key == 127:  # Backspace
            if self.cursor_pos > 0:
                self.buffer = self.buffer[:self.cursor_pos - 1] + self.buffer[self.cursor_pos:]
                self.cursor_pos -= 1
        elif key == curses.KEY_DC or key == 330:  # Delete
            if self.cursor_pos < len(self.buffer):
                self.buffer = self.buffer[:self.cursor_pos] + self.buffer[self.cursor_pos + 1:]
        elif key == curses.KEY_LEFT:
            self.cursor_pos = max(0, self.cursor_pos - 1)
        elif key == curses.KEY_RIGHT:
            self.cursor_pos = min(len(self.buffer), self.cursor_pos + 1)
        elif key == curses.KEY_HOME:
            self.cursor_pos = 0
        elif key == curses.KEY_END:
            self.cursor_pos = len(self.buffer)
        elif 32 <= key <= 126:  # Printable ASCII
            self.buffer = self.buffer[:self.cursor_pos] + chr(key) + self.buffer[self.cursor_pos:]
            self.cursor_pos += 1

        return None

    def clear(self) -> None:
        """Clear the input buffer."""
        self.buffer = ""
        self.cursor_pos = 0

    def render(self, stdscr: curses.window, y: int, x: int, width: int) -> None:
        """Render the input field.

        Args:
            stdscr: Curses window
            y: Y position
            x: X position
            width: Available width
        """
        try:
            # Display prompt + buffer
            display_text = self.prompt + self.buffer
            if len(display_text) > width:
                # Truncate if too long
                display_text = display_text[:width - 1]
            stdscr.addstr(y, x, display_text, curses.color_pair(5))

            # Draw cursor
            cursor_x = x + len(self.prompt) + self.cursor_pos
            if cursor_x < x + width:
                try:
                    stdscr.move(cursor_x, y)
                    curses.curs_set(1)  # Show cursor
                except curses.error:
                    pass
        except curses.error:
            pass


class MenuMode(Mode):
    """Base class for menu modes with input field support.

    Implements the design philosophy:
    - Fixed header rows (2-3): Mode title + usage guide
    - Data area: Scrollable content (menu options, status tables, etc.)
    - Input row: Curses input field at bottom
    """

    def __init__(self, name: str, prompt: str = "> "):
        """Initialize the menu mode.

        Args:
            name: Mode name/identifier
            prompt: Input prompt text
        """
        super().__init__(name)
        self.input = InputComponent(prompt)

    def render_header(self, stdscr: curses.window, y: int, x: int, width: int) -> None:
        """Render fixed header rows (mode title + usage guide).

        Args:
            stdscr: Curses window
            y: Y position
            x: X position
            width: Width
        """
        try:
            # Title row
            title = self.name.replace("_", " ").title()
            stdscr.addstr(y, x + 2, title, curses.color_pair(1) | curses.A_BOLD)

            # Usage guide row
            help_text = self.get_help()
            stdscr.addstr(y + 1, x + 2, help_text, curses.color_pair(5))
        except curses.error:
            pass

    def render_data(self, stdscr: curses.window, y: int, x: int, width: int, height: int) -> None:
        """Render rolling data content (override in subclasses).

        Args:
            stdscr: Curses window
            y: Y position of data area
            x: X position
            width: Width
            height: Height of data area
        """
        pass

    def render_input_row(self, stdscr: curses.window, y: int, x: int, width: int) -> None:
        """Render the input row at the bottom of the main panel.

        Args:
            stdscr: Curses window
            y: Y position of input row
            x: X position
            width: Width
        """
        try:
            # Draw separator line
            stdscr.hline(y, x, curses.ACS_HLINE, width)
            self.input.render(stdscr, y + 1, x + 2, width - 2)
        except curses.error:
            pass

    def render(self, stdscr: curses.window, y: int, x: int, width: int, height: int) -> None:
        """Render the complete mode (header + data + input row).

        Args:
            stdscr: Curses window
            y: Y position of main panel
            x: X position
            width: Width of main panel
            height: Height of main panel
        """
        # Layout: 2 header rows, remaining-1 for data, 1 for input row
        # All of this must fit inside the box borders drawn by draw_main_panel_border
        # Box interior starts at y+1 and ends at y+height-2
        interior_y = y + 1
        interior_height = height - 2
        
        header_height = 2
        input_row_height = 1
        data_height = interior_height - header_height - input_row_height

        if data_height < 1:
            data_height = 1  # Ensure at least 1 row for data

        self.render_header(stdscr, interior_y, x, width)
        self.render_data(stdscr, interior_y + header_height, x, width, data_height)
        self.render_input_row(stdscr, interior_y + header_height + data_height, x, width)

    def handle_input(self, key: int) -> str | None | Mode:
        """Handle keyboard input.

        Args:
            key: The key code

        Returns:
            New mode object, "back"/"quit" string, or None
        """
        # ESC returns to previous mode
        if key == 27:
            return "back"

        # Pass to input component
        result = self.input.handle_key(key)
        if result is not None:
            return self.handle_input_submit(result)

        return None

    def handle_input_submit(self, input_str: str) -> str | None | Mode:
        """Handle input submission (override in subclasses).

        Args:
            input_str: The submitted input string

        Returns:
            New mode object, "back"/"quit" string, or None
        """
        return None


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

    def handle_input(self, key: int) -> str | None | Mode:
        """Handle keyboard input for the main menu.

        Args:
            key: The key code

        Returns:
            New mode object to switch to, "quit"/camera mode string, or None to stay
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

    def _handle_selection(self, selected: str) -> str | None | Mode:
        """Handle menu item selection.

        Args:
            selected: The selected menu option key

        Returns:
            New mode object for menu modes, string for camera modes, or None
        """
        # Camera modes - return strings for dispatching
        if selected == "1":
            return "feed"  # Live camera feed
        elif selected == "2":
            return "multi_feed"  # Multi-camera feed
        elif selected == "5":
            return "monitor"  # Monitoring TUI
        elif selected == "7":
            return "capture"  # Capture single image
        elif selected == "8":
            return "probe"  # Probe camera

        # Menu modes - return Mode objects
        if selected == "3":
            return ZonesMenuMode()  # Setup alarm zones
        elif selected == "4":
            return PreprocessMenuMode()  # Image preprocessing
        elif selected == "6":
            return ConfigurationMenuMode()  # Configuration
        elif selected == "9":
            return "display_menu"  # Select display target (TODO: implement DisplayMenuMode)
        elif selected == "10":
            return DetectionMenuMode()  # Detection settings
        elif selected == "11":
            return RecordingMenuMode()  # Recording settings
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
        self.camera_finished = False

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
                # Only use valid coordinates
                if start_x > 0:
                    stdscr.addstr(line_y, start_x, line, curses.color_pair(3) | curses.A_BOLD)

            instruction = "Press 'q' to return to menu"
            stdscr.addstr(y + height - 2, x + (width - len(instruction)) // 2,
                         instruction, curses.color_pair(5))
        except curses.error:
            pass

    def handle_input(self, key: int) -> str | None | Mode:
        """Handle keyboard input for status mode.

        Args:
            key: The key code

        Returns:
            Return mode name on 'q', None otherwise
        """
        if self.camera_finished:
            return "back"
            
        if key in (ord("q"), ord("Q"), 27):  # 27 = ESC
            # Signal the camera thread to quit via the global flag in menus
            try:
                import ui.menus as menus
                menus._quit_requested = True
            except Exception:
                pass
            return "back"
        return None

    def get_help(self) -> str:
        """Return help text for status mode."""
        return "q/ESC Return to Menu"


class ConfigurationMenuMode(MenuMode):
    """Configuration menu mode for saving/restoring manufacturer defaults."""

    def __init__(self):
        """Initialize the configuration menu mode."""
        super().__init__("configuration_menu", "Enter choice (1-5): ")
        self.options = [
            ("1", "Save current settings + zones as manufacturer default"),
            ("2", "Restore manufacturer default"),
            ("3", "Clear user setting overrides only (keep zones)"),
            ("4", "Show config file paths"),
            ("5", "Back"),
        ]

    def get_help(self) -> str:
        """Return help text for configuration menu."""
        return "↑↓ Navigate | Enter Select | ESC Back"

    def render_data(self, stdscr: curses.window, y: int, x: int, width: int, height: int) -> None:
        """Render configuration options.

        Args:
            stdscr: Curses window
            y: Y position of data area
            x: X position
            width: Width
            height: Height of data area
        """
        try:
            from core.config import get_config
            cfg = get_config()
            has_zones = cfg.has_manufacturer_zones()
        except Exception:
            has_zones = False

        start_y = y + 1
        for i, (key, label) in enumerate(self.options):
            line_y = start_y + i
            if line_y >= y + height - 1:
                break

            # Update label 2 with zones status
            display_label = label
            if key == "2":
                display_label = f"Restore manufacturer default ({'zones available' if has_zones else 'no zones archive yet'})"

            if i == self.selected_index:
                stdscr.addstr(line_y, x + 2, "► ", curses.color_pair(1))
                stdscr.addstr(line_y, x + 4, display_label, curses.color_pair(1) | curses.A_BOLD)
            else:
                stdscr.addstr(line_y, x + 2, "  ", curses.color_pair(5))
                stdscr.addstr(line_y, x + 4, display_label, curses.color_pair(5))

    def handle_input(self, key: int) -> str | None | Mode:
        """Handle keyboard input.

        Args:
            key: The key code

        Returns:
            New mode object, "back"/"quit" string, or None
        """
        # Handle arrow keys for navigation
        if key == curses.KEY_UP:
            self.selected_index = max(0, self.selected_index - 1)
            return None
        elif key == curses.KEY_DOWN:
            self.selected_index = min(len(self.options) - 1, self.selected_index + 1)
            return None
        elif key == curses.KEY_ENTER or key == 10:
            selected = self.options[self.selected_index][0]
            return self._handle_selection(selected)
        elif key == 27:  # ESC
            return "back"

        # Pass to input component for numeric input
        result = self.input.handle_key(key)
        if result is not None:
            return self.handle_input_submit(result)

        return None

    def _handle_selection(self, selected: str) -> str | None:
        """Handle menu selection.

        Args:
            selected: Selected option key

        Returns:
            "back" if returning to main menu, None otherwise
        """
        try:
            from core.config import get_config
            cfg = get_config()

            if selected == "1":
                # Save manufacturer default
                cfg.save_as_manufacturer_default(include_zones=True)
                self.event_log_add("INFO", "Saved manufacturer default")
            elif selected == "2":
                # Restore manufacturer default
                cfg.restore_manufacturer_default(include_zones=True)
                self.event_log_add("INFO", "Restored manufacturer default")
            elif selected == "3":
                # Clear user overrides
                cfg.restore_manufacturer_default(include_zones=False)
                self.event_log_add("INFO", "Cleared user setting overrides")
            elif selected == "4":
                # Show config paths
                self.event_log_add("INFO", f"Config: {cfg.config_path}")
                self.event_log_add("INFO", f"Zones: {cfg.zones_path}")
            elif selected == "5":
                return "back"
        except Exception as e:
            self.event_log_add("ERROR", f"Configuration error: {e}")

        return None

    def handle_input_submit(self, input_str: str) -> str | None:
        """Handle numeric input submission.

        Args:
            input_str: The submitted input

        Returns:
            "back" if back selected, None otherwise
        """
        if input_str.isdigit():
            return self._handle_selection(input_str)
        return None

    def event_log_add(self, level: str, message: str) -> None:
        """Add message to event log (placeholder until we have access).

        Args:
            level: Log level
            message: Log message
        """
        # This will be replaced with actual event log access
        # For now, we can't add to event log from here
        pass


class DetectionMenuMode(MenuMode):
    """Detection settings menu mode for YOLO detection configuration."""

    def __init__(self):
        """Initialize the detection menu mode."""
        super().__init__("detection_menu", "Enter choice (1-6): ")
        self.options = [
            ("1", "Toggle detection"),
            ("2", "Set confidence threshold"),
            ("3", "Set target display FPS"),
            ("4", "Detect every N captured frames"),
            ("5", "Use 3 NPU cores (round-robin)"),
            ("6", "Back"),
        ]

    def get_help(self) -> str:
        """Return help text for detection menu."""
        return "↑↓ Navigate | Enter Select | ESC Back"

    def render_data(self, stdscr: curses.window, y: int, x: int, width: int, height: int) -> None:
        """Render detection options.

        Args:
            stdscr: Curses window
            y: Y position of data area
            x: X position
            width: Width
            height: Height of data area
        """
        try:
            from ui.menus import detection_enabled, detection_confidence, detect_every_n, use_multi_core
            det_status = "ON" if detection_enabled else "OFF"
            multi_status = "3-CORE" if use_multi_core else "1-CORE"
        except Exception:
            det_status = "OFF"
            multi_status = "1-CORE"
            detection_confidence = 0.5
            detect_every_n = 1

        start_y = y + 1
        for i, (key, label) in enumerate(self.options):
            line_y = start_y + i
            if line_y >= y + height - 1:
                break

            # Update labels with current values
            display_label = label
            if key == "1":
                display_label = f"Toggle detection [{det_status}]"
            elif key == "2":
                display_label = f"Set confidence threshold [{detection_confidence:.2f}]"
            elif key == "3":
                display_label = f"Set target display FPS [{detect_every_n}]"
            elif key == "5":
                display_label = f"Use 3 NPU cores (round-robin) [{multi_status}]"

            if i == self.selected_index:
                stdscr.addstr(line_y, x + 2, "► ", curses.color_pair(1))
                stdscr.addstr(line_y, x + 4, display_label, curses.color_pair(1) | curses.A_BOLD)
            else:
                stdscr.addstr(line_y, x + 2, "  ", curses.color_pair(5))
                stdscr.addstr(line_y, x + 4, display_label, curses.color_pair(5))

    def handle_input(self, key: int) -> str | None | Mode:
        """Handle keyboard input.

        Args:
            key: The key code

        Returns:
            New mode object, "back"/"quit" string, or None
        """
        # Handle arrow keys for navigation
        if key == curses.KEY_UP:
            self.selected_index = max(0, self.selected_index - 1)
            return None
        elif key == curses.KEY_DOWN:
            self.selected_index = min(len(self.options) - 1, self.selected_index + 1)
            return None
        elif key == curses.KEY_ENTER or key == 10:
            selected = self.options[self.selected_index][0]
            return self._handle_selection(selected)
        elif key == 27:  # ESC
            return "back"

        # Pass to input component for numeric input
        result = self.input.handle_key(key)
        if result is not None:
            return self.handle_input_submit(result)

        return None

    def _handle_selection(self, selected: str) -> str | None:
        """Handle menu selection.

        Args:
            selected: Selected option key

        Returns:
            "back" if returning to main menu, None otherwise
        """
        try:
            import ui.menus as menus
            if selected == "1":
                menus.detection_enabled = not menus.detection_enabled
                self.event_log_add("INFO", f"Detection {'enabled' if menus.detection_enabled else 'disabled'}")
            elif selected == "2":
                # Will handle via input field
                pass
            elif selected == "3":
                # Will handle via input field
                pass
            elif selected == "4":
                # Will handle via input field
                pass
            elif selected == "5":
                menus.use_multi_core = not menus.use_multi_core
                self.event_log_add("INFO", f"NPU cores: {'3-CORE' if menus.use_multi_core else '1-CORE'}")
            elif selected == "6":
                return "back"
        except Exception as e:
            self.event_log_add("ERROR", f"Detection settings error: {e}")

        return None

    def handle_input_submit(self, input_str: str) -> str | None:
        """Handle numeric input submission.

        Args:
            input_str: The submitted input

        Returns:
            "back" if back selected, None otherwise
        """
        if input_str.isdigit():
            selected = input_str
            if selected == "2":
                # Set confidence
                try:
                    import ui.menus as menus
                    val = float(input_str)
                    if 0.0 <= val <= 1.0:
                        menus.detection_confidence = val
                        self.event_log_add("INFO", f"Confidence threshold set to {val:.2f}")
                except ValueError:
                    pass
            elif selected == "3":
                # Set FPS
                try:
                    import ui.menus as menus
                    val = int(input_str)
                    if 1 <= val <= 120:
                        menus.target_fps = val
                        self.event_log_add("INFO", f"Target FPS set to {val}")
                except ValueError:
                    pass
            elif selected == "4":
                # Set every-N
                try:
                    import ui.menus as menus
                    val = int(input_str)
                    if val >= 1:
                        menus.detect_every_n = val
                        self.event_log_add("INFO", f"Detect every N frames set to {val}")
                except ValueError:
                    pass
            elif selected == "6":
                return "back"
        return None

    def event_log_add(self, level: str, message: str) -> None:
        """Add message to event log (placeholder)."""
        pass


class RecordingMenuMode(MenuMode):
    """Recording settings menu mode for video recording configuration."""

    def __init__(self):
        """Initialize the recording menu mode."""
        super().__init__("recording_menu", "Enter choice (1-6): ")
        self.options = [
            ("1", "Toggle recording"),
            ("2", "Set save directory"),
            ("3", "Set segment duration (minutes)"),
            ("4", "Set codec (mpp_h264/mp4v/mjpg/avc1)"),
            ("5", "Set storage threshold (delete when disk % full)"),
            ("6", "Back"),
        ]

    def get_help(self) -> str:
        """Return help text for recording menu."""
        return "↑↓ Navigate | Enter Select | ESC Back"

    def render_data(self, stdscr: curses.window, y: int, x: int, width: int, height: int) -> None:
        """Render recording options.

        Args:
            stdscr: Curses window
            y: Y position of data area
            x: X position
            width: Width
            height: Height of data area
        """
        try:
            from core.config import get_config
            cfg = get_config()
            enabled = cfg.get("recording.enabled", True)
            save_dir = cfg.get("recording.save_dir", "/media/pi/6333-3864")
            segment_duration = cfg.get("recording.segment_duration_min", 2)
            codec = cfg.get("recording.codec", "mp4v")
            storage_threshold = cfg.get("recording.storage_threshold_percent", 20)
        except Exception:
            enabled = True
            save_dir = "/media/pi/6333-3864"
            segment_duration = 2
            codec = "mp4v"
            storage_threshold = 20

        enabled_status = "ON" if enabled else "OFF"

        start_y = y + 1
        for i, (key, label) in enumerate(self.options):
            line_y = start_y + i
            if line_y >= y + height - 1:
                break

            # Update labels with current values
            display_label = label
            if key == "1":
                display_label = f"Toggle recording [{enabled_status}]"
            elif key == "2":
                display_label = f"Set save directory [{save_dir}]"
            elif key == "3":
                display_label = f"Set segment duration (minutes) [{segment_duration}]"
            elif key == "4":
                display_label = f"Set codec (mpp_h264/mp4v/mjpg/avc1) [{codec}]"
            elif key == "5":
                display_label = f"Set storage threshold (delete when disk % full) [{storage_threshold}]"

            if i == self.selected_index:
                stdscr.addstr(line_y, x + 2, "► ", curses.color_pair(1))
                stdscr.addstr(line_y, x + 4, display_label, curses.color_pair(1) | curses.A_BOLD)
            else:
                stdscr.addstr(line_y, x + 2, "  ", curses.color_pair(5))
                stdscr.addstr(line_y, x + 4, display_label, curses.color_pair(5))

    def handle_input(self, key: int) -> str | None | Mode:
        """Handle keyboard input.

        Args:
            key: The key code

        Returns:
            New mode object, "back"/"quit" string, or None
        """
        # Handle arrow keys for navigation
        if key == curses.KEY_UP:
            self.selected_index = max(0, self.selected_index - 1)
            return None
        elif key == curses.KEY_DOWN:
            self.selected_index = min(len(self.options) - 1, self.selected_index + 1)
            return None
        elif key == curses.KEY_ENTER or key == 10:
            selected = self.options[self.selected_index][0]
            return self._handle_selection(selected)
        elif key == 27:  # ESC
            return "back"

        # Pass to input component for numeric input
        result = self.input.handle_key(key)
        if result is not None:
            return self.handle_input_submit(result)

        return None

    def _handle_selection(self, selected: str) -> str | None:
        """Handle menu selection.

        Args:
            selected: Selected option key

        Returns:
            "back" if returning to main menu, None otherwise
        """
        try:
            from core.config import get_config
            cfg = get_config()

            if selected == "1":
                enabled = cfg.get("recording.enabled", True)
                cfg.set("recording.enabled", not enabled)
                cfg.save_user()
                self.event_log_add("INFO", f"Recording {'enabled' if not enabled else 'disabled'}")
            elif selected == "2":
                # Will handle via input field
                pass
            elif selected == "3":
                # Will handle via input field
                pass
            elif selected == "4":
                # Will handle via input field
                pass
            elif selected == "5":
                # Will handle via input field
                pass
            elif selected == "6":
                return "back"
        except Exception as e:
            self.event_log_add("ERROR", f"Recording settings error: {e}")

        return None

    def handle_input_submit(self, input_str: str) -> str | None:
        """Handle input submission.

        Args:
            input_str: The submitted input

        Returns:
            "back" if back selected, None otherwise
        """
        if input_str.isdigit():
            selected = input_str
            if selected == "2":
                # Set save directory
                try:
                    from core.config import get_config
                    cfg = get_config()
                    cfg.set("recording.save_dir", input_str)
                    cfg.save_user()
                    self.event_log_add("INFO", f"Save directory set to {input_str}")
                except Exception:
                    pass
            elif selected == "3":
                # Set segment duration
                try:
                    from core.config import get_config
                    cfg = get_config()
                    val = int(input_str)
                    if val >= 1:
                        cfg.set("recording.segment_duration_min", val)
                        cfg.save_user()
                        self.event_log_add("INFO", f"Segment duration set to {val}min")
                except ValueError:
                    pass
            elif selected == "4":
                # Set codec
                try:
                    from core.config import get_config
                    cfg = get_config()
                    valid_codecs = ["mpp_h264", "mp4v", "mjpg", "avc1"]
                    if input_str in valid_codecs:
                        cfg.set("recording.codec", input_str)
                        cfg.save_user()
                        self.event_log_add("INFO", f"Codec set to {input_str}")
                except Exception:
                    pass
            elif selected == "5":
                # Set storage threshold
                try:
                    from core.config import get_config
                    cfg = get_config()
                    val = int(input_str)
                    if 0 <= val <= 100:
                        cfg.set("recording.storage_threshold_percent", val)
                        cfg.save_user()
                        self.event_log_add("INFO", f"Storage threshold set to {val}%")
                except ValueError:
                    pass
            elif selected == "6":
                return "back"
        return None

    def event_log_add(self, level: str, message: str) -> None:
        """Add message to event log (placeholder)."""
        pass


class PreprocessMenuMode(MenuMode):
    """Preprocessing menu mode for slot-based image preprocessing configuration."""

    def __init__(self):
        """Initialize the preprocessing menu mode."""
        super().__init__("preprocess_menu", "Enter slot (0-3) or 4 for camera mapping: ")
        self.slots = 4  # 4 slots

    def get_help(self) -> str:
        """Return help text for preprocessing menu."""
        return "Enter slot # | 4 Camera Mapping | ESC Back"

    def render_data(self, stdscr: curses.window, y: int, x: int, width: int, height: int) -> None:
        """Render slot table with camera bindings and preprocessing state.

        Args:
            stdscr: Curses window
            y: Y position of data area
            x: X position
            width: Width
            height: Height of data area
        """
        try:
            from core.config import get_config
            cfg = get_config()
            zones = cfg.get("zones", {})
        except Exception:
            zones = {}

        start_y = y + 1

        # Header row
        header = f"{'Slot':<6} {'Camera':<20} {'B':<4} {'C':<4} {'S':<4} {'G':<4}"
        try:
            stdscr.addstr(start_y, x + 2, header, curses.color_pair(3) | curses.A_BOLD)
        except curses.error:
            pass

        # Slot rows
        for i in range(self.slots):
            line_y = start_y + 2 + i
            if line_y >= y + height - 1:
                break

            slot_key = f"slot_{i}"
            slot_data = zones.get(slot_key, {})
            camera = slot_data.get("camera_path", "None")
            brightness = slot_data.get("brightness", 0)
            contrast = slot_data.get("contrast", 0)
            saturation = slot_data.get("saturation", 0)
            gamma = slot_data.get("gamma", 0)

            # Truncate camera path if too long
            if len(camera) > 18:
                camera = camera[:15] + "..."

            slot_line = f"{i:<6} {camera:<20} {brightness:<4} {contrast:<4} {saturation:<4} {gamma:<4}"
            try:
                stdscr.addstr(line_y, x + 2, slot_line, curses.color_pair(5))
            except curses.error:
                pass

        # Instructions
        instr_y = start_y + 2 + self.slots + 1
        if instr_y < y + height - 1:
            try:
                stdscr.addstr(instr_y, x + 2, "B=Brightness, C=Contrast, S=Saturation, G=Gamma", curses.color_pair(5))
            except curses.error:
                pass

    def handle_input(self, key: int) -> str | None | Mode:
        """Handle keyboard input.

        Args:
            key: The key code

        Returns:
            New mode object, "back"/"quit" string, or None
        """
        if key == 27:  # ESC
            return "back"

        # Pass to input component
        result = self.input.handle_key(key)
        if result is not None:
            return self.handle_input_submit(result)

        return None

    def handle_input_submit(self, input_str: str) -> str | None | Mode:
        """Handle slot selection.

        Args:
            input_str: The submitted input

        Returns:
            PreprocessSlotMode for valid slot, "back" for 4, None otherwise
        """
        if input_str.isdigit():
            val = int(input_str)
            if 0 <= val <= 3:
                # Return slot mode (TODO: implement PreprocessSlotMode)
                return f"preprocess_slot_{val}"
            elif val == 4:
                # Camera mapping (TODO: implement)
                self.event_log_add("INFO", "Camera mapping not yet implemented")
            elif val == 5:
                return "back"
        return None

    def event_log_add(self, level: str, message: str) -> None:
        """Add message to event log (placeholder)."""
        pass


class ZonesMenuMode(MenuMode):
    """Zones menu mode for slot-based alarm zone configuration."""

    def __init__(self):
        """Initialize the zones menu mode."""
        super().__init__("zones_menu", "Enter slot (0-3) or 5 for back: ")
        self.slots = 4  # 4 slots

    def get_help(self) -> str:
        """Return help text for zones menu."""
        return "Enter slot # to edit zones | ESC Back"

    def render_data(self, stdscr: curses.window, y: int, x: int, width: int, height: int) -> None:
        """Render slot table with camera bindings and polygon state.

        Args:
            stdscr: Curses window
            y: Y position of data area
            x: X position
            width: Width
            height: Height of data area
        """
        try:
            from core.config import get_config
            cfg = get_config()
            zones = cfg.get("zones", {})
        except Exception:
            zones = {}

        start_y = y + 1

        # Header row
        header = f"{'Slot':<6} {'Camera':<20} {'Points':<8} {'Status':<10}"
        try:
            stdscr.addstr(start_y, x + 2, header, curses.color_pair(3) | curses.A_BOLD)
        except curses.error:
            pass

        # Slot rows
        for i in range(self.slots):
            line_y = start_y + 2 + i
            if line_y >= y + height - 1:
                break

            slot_key = f"slot_{i}"
            slot_data = zones.get(slot_key, {})
            camera = slot_data.get("camera_path", "None")
            polygon = slot_data.get("polygon", [])
            point_count = len(polygon) if polygon else 0
            status = "Active" if point_count >= 3 else "Inactive"

            # Truncate camera path if too long
            if len(camera) > 18:
                camera = camera[:15] + "..."

            slot_line = f"{i:<6} {camera:<20} {point_count:<8} {status:<10}"
            try:
                stdscr.addstr(line_y, x + 2, slot_line, curses.color_pair(5))
            except curses.error:
                pass

        # Instructions
        instr_y = start_y + 2 + self.slots + 1
        if instr_y < y + height - 1:
            try:
                stdscr.addstr(instr_y, x + 2, "Select slot to edit alarm zones (min 3 points)", curses.color_pair(5))
            except curses.error:
                pass

    def handle_input(self, key: int) -> str | None | Mode:
        """Handle keyboard input.

        Args:
            key: The key code

        Returns:
            New mode object, "back"/"quit" string, or None
        """
        if key == 27:  # ESC
            return "back"

        # Pass to input component
        result = self.input.handle_key(key)
        if result is not None:
            return self.handle_input_submit(result)

        return None

    def handle_input_submit(self, input_str: str) -> str | None | Mode:
        """Handle slot selection.

        Args:
            input_str: The submitted input

        Returns:
            ZonesSlotMode for valid slot, "back" for 5, None otherwise
        """
        if input_str.isdigit():
            val = int(input_str)
            if 0 <= val <= 3:
                # Return slot mode (TODO: implement ZonesSlotMode)
                return f"zones_slot_{val}"
            elif val == 5:
                return "back"
        return None

    def event_log_add(self, level: str, message: str) -> None:
        """Add message to event log (placeholder)."""
        pass


