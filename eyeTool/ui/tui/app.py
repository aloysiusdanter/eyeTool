"""Main curses TUI application for eyeTool.

Provides the main curses loop, input handling, mode switching, and refresh scheduling.
"""

from __future__ import annotations

import curses
import sys
import time
from io import StringIO
from typing import TYPE_CHECKING

from ui.tui.layout import Layout
from ui.tui.modes import MainMenuMode
from ui.tui.panels import EventLog, draw_bottom_panel, draw_main_panel_border, draw_top_panel

if TYPE_CHECKING:
    from ui.tui.modes import Mode

LOOP_TIMEOUT_MS = 15  # 15ms for real-time responsiveness (~67 FPS)
DATA_REFRESH_MS = 400  # 400ms for data panel refresh (2.5 Hz)

# Map mode names returned by MainMenuMode to functions in ui.menus.
# Each value is (function_name, needs_source, needs_output).
# Only used for camera-based modes that need special handling.
_CAMERA_MODE_DISPATCH = {
    "feed":            ("load_camera_feed",       True,  False),
    "multi_feed":      ("load_multi_camera_feed",  False, False),
    "capture":         ("capture_single_image",    True,  True),
    "probe":           ("probe_camera",            True,  False),
    "record":          ("record_camera_feed",        True,  False),
    "record_multi":    ("record_multi_camera_feed",  False, False),
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
        self.mode_stack: list[Mode] = []
        self.event_log.add("INFO", "TUI initialized")
        self._last_data_refresh = 0
        self._setup_stdout_hijacking()

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
            self._handle_input()
            # Decouple data refresh from input loop
            current_time = time.monotonic() * 1000
            if current_time - self._last_data_refresh >= DATA_REFRESH_MS:
                self._draw()
                self._last_data_refresh = current_time
            # Always call cv2.waitKey(1) if OpenCV windows are active
            self._process_opencv_windows()

        curses.curs_set(1)
        curses.echo()
        curses.nocbreak()
        stdscr.nodelay(False)
        self._restore_stdout()

    @staticmethod
    def _init_curses(stdscr: curses.window) -> None:
        """One-time curses setup."""
        curses.curs_set(0)
        curses.noecho()
        curses.cbreak()
        stdscr.nodelay(True)
        stdscr.timeout(LOOP_TIMEOUT_MS)
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
        """Handle keyboard input and mode dispatch."""
        if self.stdscr is None:
            return

        # Check if the current mode auto-finished (e.g. background camera thread done)
        if hasattr(self.current_mode, "camera_finished") and self.current_mode.camera_finished:
            self._pop_mode()
            return

        try:
            key = self.stdscr.getch()
        except curses.error:
            return

        if key != curses.ERR:
            new_mode = self.current_mode.handle_input(key)

            if new_mode == "quit":
                self.running = False
            elif new_mode == "back":
                self._pop_mode()
            elif isinstance(new_mode, Mode):
                self._push_mode(new_mode)
            elif new_mode is not None:
                # String name for a camera mode
                self._dispatch_camera_mode(new_mode)

    def _push_mode(self, new_mode: Mode) -> None:
        """Push a new mode onto the mode stack.

        Args:
            new_mode: The new mode to switch to.
        """
        self.mode_stack.append(self.current_mode)
        self.current_mode = new_mode
        self.event_log.add("INFO", f"Entered mode: {new_mode.name}")
        self._draw()  # Immediate redraw on mode switch

    def _pop_mode(self) -> None:
        """Pop the current mode and return to the previous mode."""
        if self.mode_stack:
            previous_mode = self.mode_stack.pop()
            self.current_mode = previous_mode
            self.event_log.add("INFO", f"Returned to mode: {previous_mode.name}")
            self._draw()  # Immediate redraw on mode switch
        else:
            # No previous mode, return to main menu
            self.current_mode = MainMenuMode()
            self.event_log.add("INFO", "Returned to main menu")
            self._draw()

    def _dispatch_camera_mode(self, mode_name: str) -> None:
        """Run camera-based mode alongside TUI (no curses teardown).

        Args:
            mode_name: Mode identifier for camera mode.
        """
        self.event_log.add("INFO", f"Launching camera mode: {mode_name}")

        try:
            if mode_name == "monitor":
                # Monitor mode requires curses teardown, but it's deprecated in the new TUI
                self.event_log.add("WARN", "Monitor mode is deprecated in unified TUI")
                return

            if mode_name in _CAMERA_MODE_DISPATCH:
                import ui.menus as menus
                import threading
                
                func_name, needs_source, needs_output = _CAMERA_MODE_DISPATCH[mode_name]
                func = getattr(menus, func_name)
                args: list = []
                if needs_source:
                    args.append(self.source)
                if needs_output:
                    args.append(self.output)
                
                # Switch to a status mode while camera is running
                from ui.tui.modes import StatusMode
                status_mode = StatusMode(
                    f"Camera Mode: {mode_name}\n\nPress 'q' in console or OpenCV window to exit."
                )
                self._push_mode(status_mode)
                self._draw()  # Draw immediately to show status
                
                # Run camera mode in a separate thread so it doesn't block the TUI loop
                # The camera thread will manage its own cv2.namedWindow and cv2.waitKey
                def camera_thread_worker():
                    try:
                        func(*args)
                    except Exception as e:
                        self.event_log.add("ERROR", f"Camera mode error: {e}")
                    finally:
                        self.event_log.add("INFO", f"Returned from camera mode: {mode_name}")
                        # Auto-pop the status mode if we're still on it
                        if self.current_mode is status_mode:
                            # We can't pop directly from here safely if curses is drawing, 
                            # but we can set a flag for the main loop
                            status_mode.camera_finished = True

                camera_thread = threading.Thread(target=camera_thread_worker, daemon=True)
                camera_thread.start()
                
        except Exception as e:  # noqa: BLE001
            self.event_log.add("ERROR", f"Camera mode launch error: {e}")

    def _process_opencv_windows(self) -> None:
        """Process OpenCV window events (cv2.waitKey(1)) to keep windows responsive.

        This is called in the main loop to allow OpenCV windows to render
        alongside the TUI without blocking.
        """
        try:
            import cv2
            cv2.waitKey(1)
        except Exception:  # noqa: BLE001
            # OpenCV not available or no windows open
            pass

    def _setup_stdout_hijacking(self) -> None:
        """Redirect stdout/stderr to EventLog to prevent screen corruption.

        This catches print() calls from core modules and routes them to
        the EventLog buffer instead of corrupting the curses display.
        """
        class StdoutRedirector:
            def __init__(self, event_log: EventLog):
                self.event_log = event_log
                self.original_stdout = sys.stdout
                self.original_stderr = sys.stderr

            def write(self, text: str) -> None:
                if text.strip():  # Only log non-empty lines
                    self.event_log.add("STDOUT", text.strip())
                # Also write to original for debugging (can be removed later)
                # self.original_stdout.write(text)

            def flush(self) -> None:
                pass

        self._stdout_redirector = StdoutRedirector(self.event_log)
        sys.stdout = self._stdout_redirector
        sys.stderr = self._stdout_redirector

    def _restore_stdout(self) -> None:
        """Restore original stdout/stderr on exit."""
        if hasattr(self, '_stdout_redirector'):
            sys.stdout = self._stdout_redirector.original_stdout
            sys.stderr = self._stdout_redirector.original_stderr

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
