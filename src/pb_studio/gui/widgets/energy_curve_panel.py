"""
Energy Curve Panel Widget

Energy-Kurve Anzeige und Visualisierung.

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


class EnergyCurveWidget(QWidget):
    """Widget for visualizing energy curve."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.setMaximumHeight(120)

        self.energy_levels: list[float] = [0.5] * 100

    def set_energy_curve(self, energy_levels: list[float]):
        """
        Set energy curve data.

        Args:
          energy_levels: List of energy values (0.0 to 1.0)
        """
        self.energy_levels = energy_levels
        self.update()

    def paintEvent(self, event):
        """Custom paint event for energy curve."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        bg_color = self.palette().color(self.backgroundRole()).darker(120)
        painter.fillRect(self.rect(), QBrush(bg_color))

        painter.setPen(QPen(QColor(80, 80, 80), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

        if not self.energy_levels:
            return

        # Draw grid lines
        painter.setPen(QPen(QColor(80, 80, 80), 1, Qt.PenStyle.DotLine))
        height = self.rect().height()
        for i in range(5):
            y = height * i / 4
            painter.drawLine(0, int(y), self.rect().width(), int(y))

        # Draw energy curve
        width = self.rect().width()
        painter.setPen(QPen(QColor(0, 255, 100), 2))

        points_per_pixel = max(1, len(self.energy_levels) / width)

        prev_x = 0
        prev_y = height - (self.energy_levels[0] * height)

        for x in range(1, width):
            idx = min(int(x * points_per_pixel), len(self.energy_levels) - 1)
            energy = self.energy_levels[idx]

            y = height - (energy * height)

            painter.drawLine(int(prev_x), int(prev_y), int(x), int(y))

            prev_x = x
            prev_y = y


class EnergyCurvePanel(QWidget):
    """Energy-Kurve Anzeige."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_energy: float = 0.5

        self.energy_curve_widget: EnergyCurveWidget | None = None
        self.energy_level_label: QLabel | None = None

        self._setup_ui()

    def _setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout()

        # Energy Curve Widget
        self.energy_curve_widget = EnergyCurveWidget()
        layout.addWidget(self.energy_curve_widget)

        # Energy Level Display
        energy_level_layout = QHBoxLayout()
        energy_level_layout.addWidget(QLabel("Current Energy:"))
        self.energy_level_label = QLabel("50%")
        self.energy_level_label.setStyleSheet("font-weight: bold;")
        energy_level_layout.addWidget(self.energy_level_label)
        energy_level_layout.addStretch()
        layout.addLayout(energy_level_layout)

        self.setLayout(layout)

    def set_energy_curve(self, energy_levels: list[float]):
        """
        Update energy curve visualization.

        Args:
          energy_levels: List of energy values (0.0 to 1.0)
        """
        if self.energy_curve_widget:
            self.energy_curve_widget.set_energy_curve(energy_levels)

        if energy_levels and self.energy_level_label:
            avg_energy = sum(energy_levels) / len(energy_levels)
            self.current_energy = avg_energy
            self.energy_level_label.setText(f"{int(avg_energy * 100)}%")
