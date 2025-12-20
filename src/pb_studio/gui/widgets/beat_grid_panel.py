"""
Beat Grid Panel Widget

BPM-Anzeige und Beat-Grid Visualisierung.

Author: PB_studio Development Team
"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ...utils.logger import get_logger

logger = get_logger(__name__)


class BeatGridVisualizerWidget(QWidget):
    """Widget for visualizing beat grid with vertical markers."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(60)
        self.setMaximumHeight(80)

        self.beat_times: list[float] = []
        self.total_duration: float = 0.0

    def set_beat_grid(self, beat_times: list[float], duration: float = None):
        """
        Set beat grid data.

        Args:
          beat_times: List of beat timestamps in seconds
          duration: Total audio duration (optional, auto-calculated if not provided)
        """
        self.beat_times = beat_times
        if duration:
            self.total_duration = duration
        elif beat_times:
            self.total_duration = max(beat_times) * 1.1
        else:
            self.total_duration = 0.0
        self.update()

    def paintEvent(self, event):
        """Custom paint event for beat grid."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        bg_color = self.palette().color(self.backgroundRole()).darker(120)
        painter.fillRect(self.rect(), QBrush(bg_color))

        painter.setPen(QPen(QColor(80, 80, 80), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

        if not self.beat_times or self.total_duration == 0:
            painter.setPen(QPen(self.palette().text().color()))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No beats detected")
            return

        width = self.rect().width()
        height = self.rect().height()

        painter.setPen(QPen(QColor(0, 255, 100), 2))

        for beat_time in self.beat_times:
            x = int((beat_time / self.total_duration) * width)
            painter.drawLine(x, 0, x, height)

        painter.setPen(QPen(self.palette().text().color()))
        info_text = f"{len(self.beat_times)} beats"
        painter.drawText(5, height - 5, info_text)


class BeatGridPanel(QWidget):
    """BPM und Beat-Grid Visualisierung."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_bpm: float = 120.0
        self.beat_times: list[float] = []

        self.bpm_label: QLabel | None = None
        self.beatgrid_status: QLabel | None = None
        self.beatgrid_visualizer: BeatGridVisualizerWidget | None = None

        self._setup_ui()

    def _setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout()

        # BPM Display
        bpm_layout = QHBoxLayout()
        bpm_layout.addWidget(QLabel("BPM:"))
        self.bpm_label = QLabel("120.0")
        self.bpm_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #00ff00;")
        bpm_layout.addWidget(self.bpm_label)
        bpm_layout.addStretch()
        layout.addLayout(bpm_layout)

        # Beat Grid Status
        beatgrid_layout = QHBoxLayout()
        beatgrid_layout.addWidget(QLabel("Beat Grid:"))
        self.beatgrid_status = QLabel("Not Loaded")
        self.beatgrid_status.setStyleSheet("color: #ff8800;")
        beatgrid_layout.addWidget(self.beatgrid_status)
        beatgrid_layout.addStretch()
        layout.addLayout(beatgrid_layout)

        # Beat Grid Visualizer
        self.beatgrid_visualizer = BeatGridVisualizerWidget()
        layout.addWidget(self.beatgrid_visualizer)

        self.setLayout(layout)

    def set_bpm(self, bpm: float):
        """Update BPM display."""
        self.current_bpm = bpm
        if self.bpm_label:
            self.bpm_label.setText(f"{bpm:.1f}")
        logger.debug(f"BPM set to: {bpm:.1f}")

    def set_beatgrid(self, beat_times: list[float]):
        """Update beat grid visualization."""
        self.beat_times = beat_times
        has_beatgrid = bool(beat_times)

        if self.beatgrid_status:
            if has_beatgrid:
                self.beatgrid_status.setText("✓ Loaded")
                self.beatgrid_status.setStyleSheet("color: #00ff00;")
            else:
                self.beatgrid_status.setText("Not Loaded")
                self.beatgrid_status.setStyleSheet("color: #ff8800;")

        if self.beatgrid_visualizer and has_beatgrid:
            self.beatgrid_visualizer.set_beat_grid(beat_times)

        logger.info(f"Beatgrid updated: {len(beat_times)} beats")
