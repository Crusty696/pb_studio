"""
Cut Parameters Panel

Reusable panel for cut/clip duration and tempo settings.
Extracts cut parameters UI from ParameterDashboardWidget.
"""


from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class CutParametersPanel(QGroupBox):
    """
    Panel for configuring cut parameters.

    Contains:
    - Average cut duration slider (3-12 seconds)
    - Pacing tempo slider (slow to fast)

    Signals:
        duration_changed(float): New average cut duration in seconds
        tempo_changed(int): New tempo value (0-100)
    """

    duration_changed = pyqtSignal(float)
    tempo_changed = pyqtSignal(int)

    # Tempo labels
    TEMPO_LABELS = ["Very Slow", "Slow", "Normal", "Fast", "Very Fast"]

    def __init__(self, parent: QWidget | None = None):
        super().__init__("Cut Parameters", parent)
        self.setToolTip("Grundeinstellungen für die Schnittlänge und das Tempo")
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # === Cut Duration ===
        duration_layout = QVBoxLayout()

        # Header with label and value
        duration_header = QHBoxLayout()
        duration_label = QLabel("Average Cut Duration:")
        duration_label.setToolTip(
            "Durchschnittliche Dauer eines Clips im fertigen Video.\n\n"
            "Kurz (3-4s) = Schnell geschnittenes, dynamisches Video\n"
            "Mittel (4-6s) = Ausgewogenes Tempo für Music Videos\n"
            "Lang (6-12s) = Ruhiger, filmischer Stil\n\n"
            "Hinweis: Die tatsächlichen Schnitte werden an die Musik angepasst."
        )
        duration_header.addWidget(duration_label)

        self.duration_value_label = QLabel("4.0s")
        self.duration_value_label.setStyleSheet("font-weight: bold;")
        duration_header.addWidget(self.duration_value_label)
        duration_header.addStretch()
        duration_layout.addLayout(duration_header)

        # Slider
        self.duration_slider = QSlider(Qt.Orientation.Horizontal)
        self.duration_slider.setMinimum(30)  # 3.0s
        self.duration_slider.setMaximum(120)  # 12.0s
        self.duration_slider.setValue(40)  # 4.0s default
        self.duration_slider.setToolTip(
            "Durchschnittliche Clip-Dauer:\n"
            "← Links = Kürzere Clips (3s, schneller)\n"
            "→ Rechts = Längere Clips (12s, langsamer)"
        )
        self.duration_slider.valueChanged.connect(self._on_duration_changed)
        duration_layout.addWidget(self.duration_slider)

        layout.addLayout(duration_layout)

        # === Tempo/Pace ===
        tempo_layout = QVBoxLayout()

        # Header
        tempo_header = QHBoxLayout()
        tempo_label = QLabel("Pacing Tempo:")
        tempo_label.setToolTip(
            "Allgemeines Schnitttempo des Videos.\n\n"
            "Very Slow = Ruhiger, meditativer Stil\n"
            "Slow = Entspanntes Tempo\n"
            "Normal = Standard Music Video Tempo\n"
            "Fast = Energetisches Pacing\n"
            "Very Fast = Aggressives, schnelles Pacing"
        )
        tempo_header.addWidget(tempo_label)

        self.tempo_value_label = QLabel("Normal")
        self.tempo_value_label.setStyleSheet("font-weight: bold;")
        tempo_header.addWidget(self.tempo_value_label)
        tempo_header.addStretch()
        tempo_layout.addLayout(tempo_header)

        # Slider
        self.tempo_slider = QSlider(Qt.Orientation.Horizontal)
        self.tempo_slider.setMinimum(0)
        self.tempo_slider.setMaximum(100)
        self.tempo_slider.setValue(50)  # Normal
        self.tempo_slider.setToolTip(
            "Schnitttempo:\n" "← Links = Langsamer\n" "→ Rechts = Schneller"
        )
        self.tempo_slider.valueChanged.connect(self._on_tempo_changed)
        tempo_layout.addWidget(self.tempo_slider)

        layout.addLayout(tempo_layout)

    def _on_duration_changed(self, value: int):
        """Handle duration slider change."""
        duration = value / 10.0  # Convert to seconds
        self.duration_value_label.setText(f"{duration:.1f}s")
        self.duration_changed.emit(duration)

    def _on_tempo_changed(self, value: int):
        """Handle tempo slider change."""
        # Map 0-100 to tempo labels
        index = min(value // 25, len(self.TEMPO_LABELS) - 1)
        label = self.TEMPO_LABELS[index]
        self.tempo_value_label.setText(label)
        self.tempo_changed.emit(value)

    def get_duration(self) -> float:
        """Get current cut duration in seconds."""
        return self.duration_slider.value() / 10.0

    def set_duration(self, seconds: float):
        """Set cut duration in seconds."""
        self.duration_slider.setValue(int(seconds * 10))

    def get_tempo(self) -> int:
        """Get current tempo value (0-100)."""
        return self.tempo_slider.value()

    def set_tempo(self, value: int):
        """Set tempo value (0-100)."""
        self.tempo_slider.setValue(value)

    def get_tempo_label(self) -> str:
        """Get human-readable tempo label."""
        return self.tempo_value_label.text()
