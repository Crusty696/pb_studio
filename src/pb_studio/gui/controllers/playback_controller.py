"""
Playback Controller - Handles audio playback control.

Extracted from MainWindow God Object (P1.6).
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..main_window import MainWindow

logger = logging.getLogger(__name__)


class PlaybackController:
    """
    Manages audio playback operations.

    Responsibilities:
    - Start/stop/toggle playback
    - Update playback state
    - Coordinate with timeline and preview widgets
    """

    def __init__(self, main_window: "MainWindow"):
        """
        Initialize playback controller.

        Args:
            main_window: Reference to main window
        """
        self.main_window = main_window
        self.is_playing = False

    def toggle_playback(self):
        """Toggle playback state."""
        if self.is_playing:
            self.stop_playback()
        else:
            self.start_playback()

    def start_playback(self):
        """Start playback."""
        logger.info("Starting playback")
        self.is_playing = True
        self.main_window.play_button.setText("⏸ Pause")
        self.main_window.stop_button.setEnabled(True)

        # Start timeline playback
        if self.main_window.timeline_widget:
            self.main_window.timeline_widget.start_playback()

        # Start preview playback
        if (
            self.main_window.preview_widget
            and self.main_window.preview_widget.cap
            and not self.main_window.preview_widget.is_playing
        ):
            self.main_window.preview_widget._toggle_playback()
            logger.debug("Started preview widget playback")

        self.main_window.playback_started.emit()

    def stop_playback(self):
        """Stop playback."""
        logger.info("Stopping playback")
        self.is_playing = False
        self.main_window.play_button.setText("▶ Play")
        self.main_window.stop_button.setEnabled(False)

        # Stop timeline playback
        if self.main_window.timeline_widget:
            self.main_window.timeline_widget.stop_playback()

        # Stop preview playback
        if self.main_window.preview_widget and self.main_window.preview_widget.is_playing:
            self.main_window.preview_widget._toggle_playback()
            logger.debug("Stopped preview widget playback")

        self.main_window.playback_stopped.emit()

    def on_playback_started(self):
        """Handle playback started."""
        self.main_window.update_status("Playing...")

    def on_playback_stopped(self):
        """Handle playback stopped."""
        self.main_window.update_status("Stopped")
