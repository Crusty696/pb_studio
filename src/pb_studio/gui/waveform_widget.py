"""
Waveform Widget für PB_studio

Visualisiert Audio-Waveform mit Zoom und Playhead.

Author: PB_studio Development Team
"""

import logging

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget

logger = logging.getLogger(__name__)


class WaveformWidget(QWidget):
    """
    Widget for displaying audio waveform.

    Features:
        - Render waveform with peak min/max
        - Show playhead position
        - Display beat markers
        - Mouse wheel zoom
        - Click to seek

    Signals:
        playhead_moved: Emitted when playhead position changes (float: position in seconds)
        zoom_changed: Emitted when zoom level changes (float: zoom level)
    """

    playhead_moved = pyqtSignal(float)  # position in seconds
    zoom_changed = pyqtSignal(float)  # zoom level

    def __init__(self, parent=None):
        """Initialize Waveform Widget."""
        super().__init__(parent)

        self.peaks_min: np.ndarray | None = None
        self.peaks_max: np.ndarray | None = None
        self.sample_rate: int = 22050
        self.duration: float = 0.0

        self.playhead_position: float = 0.0  # seconds
        self.zoom_level: float = 1.0
        self.scroll_offset: float = 0.0

        self.beat_markers: list[float] = []  # beat positions in seconds

        self.setMinimumHeight(100)
        self.setMouseTracking(True)

        # Colors
        self.bg_color = QColor("#2C3E50")
        self.waveform_color = QColor("#3498DB")
        self.playhead_color = QColor("#F39C12")
        self.beat_color = QColor("#E74C3C")

        logger.debug("WaveformWidget initialized")

    def set_waveform_data(
        self, peaks_min: np.ndarray, peaks_max: np.ndarray, sample_rate: int, duration: float
    ) -> None:
        """
        Set waveform data for display.

        Args:
            peaks_min: Minimum peak values
            peaks_max: Maximum peak values
            sample_rate: Audio sample rate
            duration: Audio duration in seconds
        """
        self.peaks_min = peaks_min
        self.peaks_max = peaks_max
        self.sample_rate = sample_rate
        self.duration = duration
        self.update()
        logger.info(f"Waveform data set: {len(peaks_min)} peaks, {duration:.2f}s")

    def set_playhead_position(self, position: float) -> None:
        """
        Set playhead position in seconds.

        Args:
            position: Position in seconds
        """
        self.playhead_position = max(0.0, min(position, self.duration))
        self.update()

    def set_beat_markers(self, beat_times: list[float]) -> None:
        """
        Set beat marker positions.

        Args:
            beat_times: List of beat times in seconds
        """
        self.beat_markers = beat_times
        self.update()
        logger.debug(f"Beat markers set: {len(beat_times)} beats")

    def paintEvent(self, event):
        """Paint waveform."""
        if self.peaks_min is None or self.peaks_max is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        painter.fillRect(self.rect(), self.bg_color)

        # Calculate visible region
        width = self.width()
        height = self.height()
        center_y = height // 2

        # Draw waveform
        self._draw_waveform(painter, width, height, center_y)

        # Draw beat markers
        self._draw_beat_markers(painter, width, height)

        # Draw playhead
        self._draw_playhead(painter, width, height)

    def _draw_waveform(self, painter: QPainter, width: int, height: int, center_y: int):
        """Draw waveform peaks."""
        if len(self.peaks_min) == 0:
            return

        # Waveform color
        painter.setPen(QPen(self.waveform_color, 1))
        painter.setBrush(QBrush(self.waveform_color))

        # Scale to fit widget
        samples_count = len(self.peaks_min)
        pixels_per_sample = width / samples_count * self.zoom_level

        for i in range(min(samples_count, width)):
            x = int(i * pixels_per_sample - self.scroll_offset)

            if x < 0 or x >= width:
                continue

            # Convert sample values (-1 to 1) to pixel coordinates
            y_min = int(center_y + self.peaks_min[i] * (height / 2) * 0.9)
            y_max = int(center_y + self.peaks_max[i] * (height / 2) * 0.9)

            painter.drawLine(x, y_min, x, y_max)

    def _draw_beat_markers(self, painter: QPainter, width: int, height: int):
        """Draw beat markers."""
        if self.duration == 0:
            return

        painter.setPen(QPen(self.beat_color, 1, Qt.PenStyle.DashLine))

        for beat_time in self.beat_markers:
            x = int((beat_time / self.duration) * width)
            if 0 <= x < width:
                painter.drawLine(x, 0, x, height)

    def _draw_playhead(self, painter: QPainter, width: int, height: int):
        """Draw playhead line."""
        if self.duration == 0:
            return

        painter.setPen(QPen(self.playhead_color, 2))

        x = int((self.playhead_position / self.duration) * width)
        if 0 <= x < width:
            painter.drawLine(x, 0, x, height)

    def wheelEvent(self, event):
        """Handle mouse wheel for zoom."""
        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 0.9

        self.zoom_level *= zoom_factor
        self.zoom_level = max(1.0, min(self.zoom_level, 10.0))

        self.zoom_changed.emit(self.zoom_level)
        self.update()

    def mousePressEvent(self, event):
        """Handle mouse click to seek."""
        if event.button() == Qt.MouseButton.LeftButton and self.duration > 0:
            x = event.position().x()
            position = (x / self.width()) * self.duration
            position = max(0.0, min(position, self.duration))
            self.playhead_moved.emit(position)

    def set_theme_colors(self, bg: str, waveform: str, playhead: str, beat: str):
        """
        Set theme colors.

        Args:
            bg: Background color (hex)
            waveform: Waveform color (hex)
            playhead: Playhead color (hex)
            beat: Beat marker color (hex)
        """
        self.bg_color = QColor(bg)
        self.waveform_color = QColor(waveform)
        self.playhead_color = QColor(playhead)
        self.beat_color = QColor(beat)
        self.update()
