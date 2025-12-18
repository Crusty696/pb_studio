"""
Cut Parameter Panel Widget

Cut-Dauer, Tempo und Pacing-Mode Einstellungen.

Author: PB_studio Development Team
"""


from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ...utils.logger import get_logger

logger = get_logger(__name__)


class CutParameterPanel(QWidget):
    """Cut-Dauer, Tempo Sliders und Pacing Mode."""

    parameter_changed = pyqtSignal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.cut_duration_slider: QSlider | None = None
        self.cut_duration_value_label: QLabel | None = None
        self.tempo_slider: QSlider | None = None
        self.tempo_value_label: QLabel | None = None
        self.pacing_mode_combo: QComboBox | None = None
        self.pacing_mode_description: QLabel | None = None
        self.motion_matching_checkbox: QCheckBox | None = None
        self.structure_awareness_checkbox: QCheckBox | None = None
        self.continuity_slider: QSlider | None = None
        self.continuity_value_label: QLabel | None = None

        self._setup_ui()

    def _setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout()

        # Pacing Mode Section
        pacing_mode_layout = QVBoxLayout()

        mode_selector_layout = QHBoxLayout()
        mode_selector_layout.addWidget(QLabel("Pacing Modus:"))

        self.pacing_mode_combo = QComboBox()
        self.pacing_mode_combo.addItem("Beat-Sync Mode", "BEAT_SYNC")
        self.pacing_mode_combo.addItem("Adaptive Flow Mode", "ADAPTIVE_FLOW")
        self.pacing_mode_combo.setCurrentIndex(0)
        self.pacing_mode_combo.setToolTip(
            "Wähle den Pacing-Modus:\n\n"
            "Beat-Sync: Feste Schnittlängen (~0.6s), maximale Beat-Synchronisation\n"
            "Adaptive Flow: Variable Clip-Längen (4-10s), emotionaler Fluss"
        )
        self.pacing_mode_combo.currentIndexChanged.connect(self._on_pacing_mode_changed)
        mode_selector_layout.addWidget(self.pacing_mode_combo)
        mode_selector_layout.addStretch()

        pacing_mode_layout.addLayout(mode_selector_layout)

        self.pacing_mode_description = QLabel(
            "Beat-Sync: Feste 0.6s Cuts, maximale Synchronisation"
        )
        self.pacing_mode_description.setWordWrap(True)
        pacing_mode_layout.addWidget(self.pacing_mode_description)

        layout.addLayout(pacing_mode_layout)

        # Cut Duration
        cut_duration_layout = QVBoxLayout()
        cut_duration_header = QHBoxLayout()
        cut_duration_label = QLabel("Average Cut Duration:")
        cut_duration_label.setToolTip(
            "Durchschnittliche Dauer eines Clips im fertigen Video.\n\n"
            "Kurz (3-4s) = Schnell geschnittenes, dynamisches Video\n"
            "Mittel (4-6s) = Ausgewogenes Tempo für Music Videos\n"
            "Lang (6-12s) = Ruhiger, filmischer Stil\n\n"
            "Hinweis: Die tatsächlichen Schnitte werden an die Musik angepasst."
        )
        cut_duration_header.addWidget(cut_duration_label)
        self.cut_duration_value_label = QLabel("4.0s")
        self.cut_duration_value_label.setStyleSheet("font-weight: bold;")
        cut_duration_header.addWidget(self.cut_duration_value_label)
        cut_duration_header.addStretch()
        cut_duration_layout.addLayout(cut_duration_header)

        self.cut_duration_slider = QSlider(Qt.Orientation.Horizontal)
        self.cut_duration_slider.setMinimum(30)
        self.cut_duration_slider.setMaximum(120)
        self.cut_duration_slider.setValue(40)
        self.cut_duration_slider.setToolTip(
            "Durchschnittliche Clip-Dauer:\n"
            "← Links = Kürzere Clips (3s, schneller)\n"
            "→ Rechts = Längere Clips (12s, langsamer)"
        )
        self.cut_duration_slider.valueChanged.connect(self._on_cut_duration_changed)
        cut_duration_layout.addWidget(self.cut_duration_slider)

        layout.addLayout(cut_duration_layout)

        # Tempo/Pace
        tempo_layout = QVBoxLayout()
        tempo_header = QHBoxLayout()
        tempo_label = QLabel("Pacing Tempo:")
        tempo_label.setToolTip(
            "Allgemeines Schnitttempo des Videos.\n\n"
            "Very Slow = Ruhiger, meditativer Stil\n"
            "Slow = Entspanntes Tempo\n"
            "Normal = Standard Music Video Tempo\n"
            "Fast = Energetisches Pacing\n"
            "Very Fast = Aggressives, schnelles Pacing\n\n"
            "Wird mit den Trigger-Einstellungen kombiniert."
        )
        tempo_header.addWidget(tempo_label)
        self.tempo_value_label = QLabel("Normal")
        self.tempo_value_label.setStyleSheet("font-weight: bold;")
        tempo_header.addWidget(self.tempo_value_label)
        tempo_header.addStretch()
        tempo_layout.addLayout(tempo_header)

        self.tempo_slider = QSlider(Qt.Orientation.Horizontal)
        self.tempo_slider.setMinimum(0)
        self.tempo_slider.setMaximum(100)
        self.tempo_slider.setValue(50)
        self.tempo_slider.setToolTip(
            "Schnitttempo:\n"
            "← Links = Langsamer (Very Slow bis Slow)\n"
            "→ Rechts = Schneller (Fast bis Very Fast)"
        )
        self.tempo_slider.valueChanged.connect(self._on_tempo_changed)
        tempo_layout.addWidget(self.tempo_slider)

        layout.addLayout(tempo_layout)

        # Visual Continuity
        continuity_layout = QVBoxLayout()
        continuity_header = QHBoxLayout()
        continuity_label = QLabel("Visual Continuity:")
        continuity_label.setToolTip(
            "Steuert den visuellen Fluss ('Roter Faden').\n\n"
            "0% = Zufällige Auswahl (hohe Varianz)\n"
            "100% = Maximale Ähnlichkeit zum vorherigen Clip\n"
            "Empfohlen: 30-60% für ausgewogenen Flow"
        )
        continuity_header.addWidget(continuity_label)
        self.continuity_value_label = QLabel("40%")
        self.continuity_value_label.setStyleSheet("font-weight: bold;")
        continuity_header.addWidget(self.continuity_value_label)
        continuity_header.addStretch()
        continuity_layout.addLayout(continuity_header)

        self.continuity_slider = QSlider(Qt.Orientation.Horizontal)
        self.continuity_slider.setMinimum(0)
        self.continuity_slider.setMaximum(100)
        self.continuity_slider.setValue(40)  # Default 0.4
        self.continuity_slider.setToolTip(
            "Visuelle Kontinuität:\n"
            "← Links = Mehr Abwechslung\n"
            "→ Rechts = Mehr visueller Fluss"
        )
        self.continuity_slider.valueChanged.connect(self._on_continuity_changed)
        continuity_layout.addWidget(self.continuity_slider)

        layout.addLayout(continuity_layout)

        # Motion-Energy-Matching
        self.motion_matching_checkbox = QCheckBox("Motion-Energy-Matching aktivieren")
        self.motion_matching_checkbox.setChecked(True)
        self.motion_matching_checkbox.setToolTip(
            "Wählt Clips intelligent basierend auf Video-Motion und Audio-Energy"
        )
        self.motion_matching_checkbox.stateChanged.connect(
            lambda state: self.parameter_changed.emit(
                "motion_matching_enabled", state == Qt.CheckState.Checked.value
            )
        )
        layout.addWidget(self.motion_matching_checkbox)

        # Structure Awareness
        self.structure_awareness_checkbox = QCheckBox("Structure Awareness aktivieren")
        self.structure_awareness_checkbox.setChecked(False)
        self.structure_awareness_checkbox.setToolTip(
            "Analysiert Song-Struktur (Intro, Verse, Chorus, Drop, Outro) für intelligentere Schnitte"
        )
        self.structure_awareness_checkbox.stateChanged.connect(
            lambda state: self.parameter_changed.emit(
                "structure_awareness_enabled", state == Qt.CheckState.Checked.value
            )
        )
        layout.addWidget(self.structure_awareness_checkbox)

        layout.addStretch()
        self.setLayout(layout)

    def _on_cut_duration_changed(self, value: int):
        """Handle cut duration slider change."""
        duration = value / 10.0
        if self.cut_duration_value_label:
            self.cut_duration_value_label.setText(f"{duration:.1f}s")
        self.parameter_changed.emit("cut_duration", duration)
        logger.debug(f"Cut duration changed: {duration:.1f}s")

    def _on_tempo_changed(self, value: int):
        """Handle tempo slider change."""
        tempo_labels = ["Very Slow", "Slow", "Normal", "Fast", "Very Fast"]
        tempo_idx = min(value // 20, len(tempo_labels) - 1)
        tempo_label = tempo_labels[tempo_idx]

        if self.tempo_value_label:
            self.tempo_value_label.setText(tempo_label)
        self.parameter_changed.emit("tempo", value / 100.0)
        logger.debug(f"Tempo changed: {tempo_label} ({value})")

    def _on_continuity_changed(self, value: int):
        """Handle continuity slider change."""
        if self.continuity_value_label:
            self.continuity_value_label.setText(f"{value}%")

        weight = value / 100.0
        self.parameter_changed.emit("continuity_weight", weight)
        logger.debug(f"Continuity weight changed: {weight:.2f}")

    def _on_pacing_mode_changed(self, index: int):
        """Handle pacing mode change."""
        if not self.pacing_mode_combo:
            return

        mode_value = self.pacing_mode_combo.itemData(index)
        mode_name = self.pacing_mode_combo.itemText(index)

        if mode_value == "BEAT_SYNC":
            description = "Beat-Sync: Feste 0.6s Cuts, maximale Synchronisation"
        else:
            description = "Adaptive Flow: Variable 4-10s Clips, emotionaler Fluss"

        if self.pacing_mode_description:
            self.pacing_mode_description.setText(description)

        self.parameter_changed.emit("pacing_mode", mode_value)
        logger.info(f"Pacing mode changed: {mode_name} ({mode_value})")

    def get_cut_parameters(self) -> dict:
        """Get current cut parameter values."""
        params = {}

        if self.cut_duration_slider:
            params["cut_duration"] = self.cut_duration_slider.value() / 10.0

        if self.tempo_slider:
            params["tempo"] = self.tempo_slider.value() / 100.0

        if self.pacing_mode_combo:
            params["pacing_mode"] = self.pacing_mode_combo.currentData()

        if self.motion_matching_checkbox:
            params["motion_matching_enabled"] = self.motion_matching_checkbox.isChecked()

        if self.structure_awareness_checkbox:
            params["structure_awareness_enabled"] = self.structure_awareness_checkbox.isChecked()

        if self.continuity_slider:
            params["continuity_weight"] = self.continuity_slider.value() / 100.0

        return params
