"""Layout management for the htop-style TUI.

Calculates panel heights based on terminal size and handles terminal resize.
"""

from __future__ import annotations

MIN_TERMINAL_WIDTH = 80
MIN_TERMINAL_HEIGHT = 24

TOP_PANEL_HEIGHT = 4
BOTTOM_PANEL_HEIGHT = 5


class Layout:
    """Manages panel layout for the TUI."""

    def __init__(self, width: int, height: int):
        """Initialize layout with terminal dimensions.

        Args:
            width: Terminal width in columns
            height: Terminal height in rows
        """
        self.width = width
        self.height = height
        self._validate_size()

    def _validate_size(self) -> None:
        """Ensure terminal meets minimum size requirements."""
        if self.width < MIN_TERMINAL_WIDTH or self.height < MIN_TERMINAL_HEIGHT:
            raise ValueError(
                f"Terminal too small: {self.width}x{self.height}. "
                f"Minimum required: {MIN_TERMINAL_WIDTH}x{MIN_TERMINAL_HEIGHT}"
            )

    def top_panel_height(self) -> int:
        """Return height of the top panel (camera status, metrics)."""
        return TOP_PANEL_HEIGHT

    def bottom_panel_height(self) -> int:
        """Return height of the bottom panel (event log, help)."""
        return BOTTOM_PANEL_HEIGHT

    def main_panel_height(self) -> int:
        """Return height of the main panel (interactive content)."""
        return self.height - self.top_panel_height() - self.bottom_panel_height()

    def top_panel_y(self) -> int:
        """Return Y position of the top panel (0-based)."""
        return 0

    def main_panel_y(self) -> int:
        """Return Y position of the main panel (0-based)."""
        return self.top_panel_height()

    def bottom_panel_y(self) -> int:
        """Return Y position of the bottom panel (0-based)."""
        return self.top_panel_height() + self.main_panel_height()

    def resize(self, width: int, height: int) -> None:
        """Update layout after terminal resize.

        Args:
            width: New terminal width in columns
            height: New terminal height in rows
        """
        self.width = width
        self.height = height
        self._validate_size()

    def is_valid_size(self, width: int, height: int) -> bool:
        """Check if given dimensions meet minimum size requirements.

        Args:
            width: Terminal width in columns
            height: Terminal height in rows

        Returns:
            True if dimensions are valid, False otherwise
        """
        return width >= MIN_TERMINAL_WIDTH and height >= MIN_TERMINAL_HEIGHT
