"""Main curses TUI application for eyeTool.

Provides the main curses loop, input handling, mode switching, and refresh scheduling.
"""

from __future__ import annotations

import curses
import time

from ui.tui.layout import Layout
from ui.tui.modes import MainMenuMode
from ui.tui.panels import EventLog, draw_bottom_panel, draw_main_panel_border, draw_top_panel

REFRESH_MS = 400  # 400ms refresh rate (2.5 Hz)

# Map mode names returned by MainMenuMode to functions in ui.menus.
# Each value is (function_name, needs_source, needs_output).
_MODE_DISPATCH = {
    "feed":            ("load_camera_feed",       True,  False),
    "multi_feed":      ("load_multi_camera_feed",  False, False),
    "capture":         ("capture_single_image",    True,  True),
    "probe":           ("probe_camera",            True,  False),
    "zones_menu":      ("setup_zones_menu",        False, False),
    "preprocess_menu": ("preprocess_settings_menu", False, False),
    "config_menu":     ("configuration_menu",      False, False),
    "display_menu":    ("select_display_menu",     False, False),
    "detection_menu":  ("detection_settings_menu",  False, False),
    "monitor":         (None, False, False),  # special-cased
}


class TUIApp:
    """Main curses TUI application with full feature integration."""

    def __init__(self, source: int | str = 0, output: str = "captured_image.jpg"):
        """Initialize the TUI application.

        Args:
            source: Camera source (device index or path)
            output: Output filename for capture mode
        """
        self.source = source
        self.output = output
        self.layout: Layout | None = None
        self.running = False
        self.current_mode = MainMenuMode()
        self.stdscr: curses.window | None = None
        self.event_log = EventLog()
        self.event_log.add("INFO", "TUI initialized")

    def run(self, stdscr: curses.window) -> None:
        """Run the TUI application.

        Args:
            stdscr: The curses standard screen
        """
        self.stdscr = stdscr
        self.running = True

        self._init_curses(stdscr)

        height, width = stdscr.getmaxyx()
        try:
            self.layout = Layout(width, height)
        except ValueError as e:
            stdscr.clear()
            try:
                stdscr.addstr(0, 0, str(e), curses.color_pair(2))
            except curses.error:
                pass
            stdscr.refresh()
            time.sleep(3)
            return

        while self.running:
            self._handle_resize()
            self._draw()
            self._handle_input()

        curses.curs_set(1)
        curses.echo()
        curses.nocbreak()
        stdscr.nodelay(False)

    @staticmethod
    def _init_curses(stdscr: curses.window) -> None:
        """One-time curses setup."""
        curses.curs_set(0)
        curses.noecho()
        curses.cbreak()
        stdscr.nodelay(True)
        stdscr.timeout(REFRESH_MS)
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLACK)

    def _handle_resize(self) -> None:
        """Handle terminal resize events."""
        if self.stdscr is None or self.layout is None:
            return
        height, width = self.stdscr.getmaxyx()
        if self.layout.is_valid_size(width, height):
            self.layout.resize(width, height)

    def _handle_input(self) -> None:
        """Handle keyboard input."""
        if self.stdscr is None:
            return
        try:
            key = self.stdscr.getch()
        except curses.error:
            return
        if key == curses.ERR:
            return
        if key in (ord("q"), ord("Q"), 27):
            self.running = False
            return
        new_mode = self.current_mode.handle_input(key)
        if new_mode == "quit":
            self.running = False
        elif new_mode is not None:
            self._dispatch_mode(new_mode)

    def _dispatch_mode(self, mode_name: str) -> None:
        """Exit curses, run the selected feature, then resume curses.

        Args:
            mode_name: Mode identifier returned by the menu mode.
        """
        self.event_log.add("INFO", f"Launching: {mode_name}")
        curses.endwin()

        try:
            if mode_name == "monitor":
                from ui.monitor import run as run_monitor
                run_monitor()
            elif mode_name in _MODE_DISPATCH:
                import ui.menus as menus
                func_name, needs_source, needs_output = _MODE_DISPATCH[mode_name]
                func = getattr(menus, func_name)
                args: list = []
                if needs_source:
                    args.append(self.source)
                if needs_output:
                    args.append(self.output)
                func(*args)
        except Exception as e:  # noqa: BLE001
            print(f"Error: {e}")

        # Re-initialise curses after the sub-process returned.
        self.stdscr = curses.initscr()
        self._init_curses(self.stdscr)
        self.event_log.add("INFO", f"Returned from: {mode_name}")

    def _draw(self) -> None:
        """Draw all panels."""
        if self.stdscr is None or self.layout is None:
            return
        self.stdscr.clear()
        draw_top_panel(self.stdscr, self.layout, self.event_log)
        draw_main_panel_border(self.stdscr, self.layout,
                               self.current_mode.name.replace("_", " ").title())
        self.current_mode.render(self.stdscr, self.layout.main_panel_y(), 0,
                                 self.layout.width, self.layout.main_panel_height())
        draw_bottom_panel(self.stdscr, self.layout, self.event_log,
                          self.current_mode.get_help())
        self.stdscr.refresh()


def run_tui(source: int | str = 0, output: str = "captured_image.jpg") -> None:
    """Entry point to run the TUI application.

    Args:
        source: Camera source
        output: Output filename for capture mode
    """
    app = TUIApp(source=source, output=output)
    curses.wrapper(app.run)
