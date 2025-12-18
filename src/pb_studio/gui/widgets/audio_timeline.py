"""
Audio Timeline Widget for PB_studio

Production-ready PyQt6 widget for audio waveform visualization with beat markers.

Features:
- High-performance waveform rendering
- Beat/onset marker display
- Zoom and pan controls
- Playback position indicator
- Click-to-seek functionality
- Optimized drawing with caching

Dependencies:
- PyQt6
- numpy
- librosa (for waveform downsampling)

Usage:
    timeline = AudioTimelineWidget()
    timeline.load_audio(audio_samples, sample_rate)
    timeline.set_beats([1.5, 2.0, 2.5, 3.0])  # Beat times in seconds
"""

import numpy as np
from numpy.typing import NDArray
from PyQt6.QtCore import QPoint, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QPixmap,
    QResizeEvent,
    QWheelEvent,
)
from PyQt6.QtWidgets import QWidget


class AudioTimelineWidget(QWidget):
    """
    High-performance audio waveform timeline with beat markers.

    Signals:
        position_changed(float): Emitted when user clicks to seek (time in seconds)
        zoom_changed(float): Emitted when zoom level changes (pixels per second)
    """

    # Signals
    position_changed = pyqtSignal(float)
    zoom_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        """
        Initialize the AudioTimelineWidget.

        Args:
            parent: Parent widget (optional)
        """
        super().__init__(parent)

        # Audio data
        self.audio_samples: NDArray[np.floating] | None = None
        self.sample_rate: int = 44100
        self.duration: float = 0.0

        # Downsampled waveform for display
        self.waveform_peaks: NDArray[np.floating] | None = None
        self.waveform_cache: QPixmap | None = None
        self.cache_valid: bool = False

        # Beat markers
        self.beat_times: list[float] = []
        self.onset_times: list[float] = []

        # Playback state
        self.playback_position: float = 0.0  # Current position in seconds
        self.is_playing: bool = False

        # View state
        self.zoom_level: float = 100.0  # Pixels per second
        self.offset: float = 0.0  # Horizontal offset in seconds

        # Interaction state
        self.is_dragging: bool = False
        self.drag_start_x: int = 0
        self.drag_start_offset: float = 0.0

        # Styling
        self.background_color = QColor(30, 30, 30)
        self.waveform_color = QColor(100, 180, 255)
        self.beat_color = QColor(255, 100, 100, 180)
        self.onset_color = QColor(100, 255, 100, 120)
        self.playhead_color = QColor(255, 200, 50)
        self.grid_color = QColor(60, 60, 60)

        # Widget setup
        self.setMinimumHeight(100)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Update timer for playback
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start(33)  # ~30 FPS

    def load_audio(self, audio_samples: NDArray[np.floating], sample_rate: int) -> None:
        """
        Load audio data into the timeline.

        Args:
            audio_samples: Audio waveform data (mono)
            sample_rate: Sample rate in Hz
        """
        self.audio_samples = audio_samples
        self.sample_rate = sample_rate
        self.duration = len(audio_samples) / sample_rate

        # Downsample for display performance
        self._compute_waveform_peaks()

        # Invalidate cache
        self.cache_valid = False
        self.update()

    def _compute_waveform_peaks(self) -> None:
        """
        Compute downsampled waveform peaks for efficient rendering.

        Uses max/min downsampling for accurate peak representation.
        """
        if self.audio_samples is None:
            return

        # Target: ~2 samples per pixel at default zoom
        target_samples = int(self.width() * 2)
        if target_samples == 0:
            target_samples = 1000

        # Calculate downsample factor
        downsample_factor = max(1, len(self.audio_samples) // target_samples)

        # Reshape and compute min/max
        samples_trimmed = self.audio_samples[
            : len(self.audio_samples) // downsample_factor * downsample_factor
        ]
        samples_reshaped = samples_trimmed.reshape(-1, downsample_factor)

        peaks_max = np.max(samples_reshaped, axis=1)
        peaks_min = np.min(samples_reshaped, axis=1)

        # Interleave min/max for drawing
        self.waveform_peaks = np.empty(len(peaks_max) * 2, dtype=np.float32)
        self.waveform_peaks[0::2] = peaks_max
        self.waveform_peaks[1::2] = peaks_min

    def set_beats(self, beat_times: list[float]) -> None:
        """
        Set beat marker positions.

        Args:
            beat_times: List of beat times in seconds
        """
        self.beat_times = beat_times
        self.cache_valid = False
        self.update()

    def set_onsets(self, onset_times: list[float]) -> None:
        """
        Set onset marker positions.

        Args:
            onset_times: List of onset times in seconds
        """
        self.onset_times = onset_times
        self.cache_valid = False
        self.update()

    def set_playback_position(self, position: float) -> None:
        """
        Set current playback position.

        Args:
            position: Position in seconds
        """
        self.playback_position = max(0.0, min(position, self.duration))
        self.update()

    def set_playing(self, playing: bool) -> None:
        """
        Set playback state.

        Args:
            playing: True if playing, False if paused
        """
        self.is_playing = playing

    def zoom_in(self) -> None:
        """Zoom in (increase pixels per second)."""
        self.zoom_level = min(self.zoom_level * 1.2, 10000.0)
        self.cache_valid = False
        self.zoom_changed.emit(self.zoom_level)
        self.update()

    def zoom_out(self) -> None:
        """Zoom out (decrease pixels per second)."""
        self.zoom_level = max(self.zoom_level / 1.2, 10.0)
        self.cache_valid = False
        self.zoom_changed.emit(self.zoom_level)
        self.update()

    def fit_to_width(self) -> None:
        """Zoom to fit entire audio in widget width."""
        if self.duration > 0:
            self.zoom_level = self.width() / self.duration
            self.offset = 0.0
            self.cache_valid = False
            self.zoom_changed.emit(self.zoom_level)
            self.update()

    def _time_to_x(self, time: float) -> int:
        """Convert time in seconds to pixel x-coordinate."""
        return int((time - self.offset) * self.zoom_level)

    def _x_to_time(self, x: int) -> float:
        """Convert pixel x-coordinate to time in seconds."""
        return (x / self.zoom_level) + self.offset

    def paintEvent(self, event: QPaintEvent) -> None:
        """Paint the timeline."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw background
        painter.fillRect(self.rect(), self.background_color)

        if self.audio_samples is None:
            painter.end()
            return

        # Draw cached waveform if valid
        if self.cache_valid and self.waveform_cache is not None:
            painter.drawPixmap(0, 0, self.waveform_cache)
        else:
            # Render to cache
            self._render_to_cache()
            if self.waveform_cache is not None:
                painter.drawPixmap(0, 0, self.waveform_cache)

        # Draw playback position (always on top, not cached)
        self._draw_playhead(painter)

        painter.end()

    def _render_to_cache(self) -> None:
        """Render waveform, beats, and grid to cache pixmap."""
        # Create cache pixmap
        self.waveform_cache = QPixmap(self.size())
        self.waveform_cache.fill(Qt.GlobalColor.transparent)

        cache_painter = QPainter(self.waveform_cache)
        cache_painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw time grid
        self._draw_time_grid(cache_painter)

        # Draw onset markers (background)
        self._draw_onset_markers(cache_painter)

        # Draw beat markers
        self._draw_beat_markers(cache_painter)

        # Draw waveform
        self._draw_waveform(cache_painter)

        cache_painter.end()
        self.cache_valid = True

    def _draw_time_grid(self, painter: QPainter) -> None:
        """Draw time grid lines."""
        painter.setPen(QPen(self.grid_color, 1))

        # Calculate grid interval (1, 5, 10, 30, 60 seconds)
        visible_duration = self.width() / self.zoom_level
        if visible_duration < 10:
            interval = 1.0
        elif visible_duration < 50:
            interval = 5.0
        elif visible_duration < 120:
            interval = 10.0
        elif visible_duration < 600:
            interval = 30.0
        else:
            interval = 60.0

        # Draw grid lines
        start_time = int(self.offset / interval) * interval
        time = start_time
        while time < self.offset + visible_duration:
            x = self._time_to_x(time)
            if 0 <= x < self.width():
                painter.drawLine(x, 0, x, self.height())
            time += interval

    def _draw_waveform(self, painter: QPainter) -> None:
        """Draw audio waveform."""
        if self.waveform_peaks is None or len(self.waveform_peaks) == 0:
            return

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self.waveform_color))

        # Calculate visible range
        height = self.height()
        center_y = height / 2

        # Create path for waveform
        path = QPainterPath()

        # Calculate sample indices for visible range
        samples_per_pixel = 1.0 / self.zoom_level * self.sample_rate
        start_sample = int(
            self.offset
            * self.sample_rate
            / (len(self.audio_samples) / len(self.waveform_peaks) * 2)
        )
        start_sample = max(0, start_sample)

        # Draw waveform peaks
        for i in range(self.width()):
            sample_idx = start_sample + int(
                i * samples_per_pixel / (len(self.audio_samples) / len(self.waveform_peaks) * 2)
            )

            if sample_idx * 2 + 1 >= len(self.waveform_peaks):
                break

            peak_max = self.waveform_peaks[sample_idx * 2]
            peak_min = self.waveform_peaks[sample_idx * 2 + 1]

            y_max = int(center_y - peak_max * center_y * 0.9)
            y_min = int(center_y - peak_min * center_y * 0.9)

            # Draw vertical line for this pixel
            painter.drawLine(i, y_max, i, y_min)

    def _draw_beat_markers(self, painter: QPainter) -> None:
        """Draw beat markers."""
        painter.setPen(QPen(self.beat_color, 2))

        for beat_time in self.beat_times:
            x = self._time_to_x(beat_time)
            if 0 <= x < self.width():
                painter.drawLine(x, 0, x, self.height())

    def _draw_onset_markers(self, painter: QPainter) -> None:
        """Draw onset markers (lighter, background)."""
        painter.setPen(QPen(self.onset_color, 1))

        for onset_time in self.onset_times:
            x = self._time_to_x(onset_time)
            if 0 <= x < self.width():
                painter.drawLine(x, 0, x, self.height())

    def _draw_playhead(self, painter: QPainter) -> None:
        """Draw playback position indicator."""
        x = self._time_to_x(self.playback_position)

        if 0 <= x < self.width():
            painter.setPen(QPen(self.playhead_color, 2))
            painter.drawLine(x, 0, x, self.height())

            # Draw triangle at top
            triangle = [QPoint(x, 0), QPoint(x - 5, 10), QPoint(x + 5, 10)]
            painter.setBrush(QBrush(self.playhead_color))
            painter.drawPolygon(triangle)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Seek to clicked position
            time = self._x_to_time(event.position().x())
            self.position_changed.emit(max(0.0, min(time, self.duration)))

        elif event.button() == Qt.MouseButton.MiddleButton:
            # Start panning
            self.is_dragging = True
            self.drag_start_x = int(event.position().x())
            self.drag_start_offset = self.offset
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move."""
        if self.is_dragging:
            # Pan view
            dx = int(event.position().x()) - self.drag_start_x
            self.offset = max(0.0, self.drag_start_offset - dx / self.zoom_level)
            self.cache_valid = False
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self.is_dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handle mouse wheel for zooming."""
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Handle widget resize."""
        super().resizeEvent(event)
        self._compute_waveform_peaks()
        self.cache_valid = False

    def keyPressEvent(self, event) -> None:
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key.Key_Plus:
            self.zoom_in()
        elif event.key() == Qt.Key.Key_Minus:
            self.zoom_out()
        elif event.key() == Qt.Key.Key_F:
            self.fit_to_width()
        elif event.key() == Qt.Key.Key_Home:
            self.offset = 0.0
            self.cache_valid = False
            self.update()
        elif event.key() == Qt.Key.Key_End:
            self.offset = max(0.0, self.duration - self.width() / self.zoom_level)
            self.cache_valid = False
            self.update()
        else:
            super().keyPressEvent(event)
