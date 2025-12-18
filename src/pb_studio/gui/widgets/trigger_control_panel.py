"""
Trigger Control Panel Widget

Trigger-Settings für Cuts (Beat, Onset, Kick, Snare, HiHat, Energy).

Author: PB_studio Development Team
"""


from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ...utils.logger import get_logger

logger = get_logger(__name__)


class TriggerControlPanel(QWidget):
    """Trigger-Settings für Cuts."""

    parameter_changed = pyqtSignal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Trigger Controls (enabled checkboxes, intensity sliders, threshold sliders)
        self.beat_enabled: QCheckBox | None = None
        self.beat_intensity_slider: QSlider | None = None
        self.beat_threshold_slider: QSlider | None = None

        self.onset_enabled: QCheckBox | None = None
        self.onset_intensity_slider: QSlider | None = None
        self.onset_threshold_slider: QSlider | None = None

        self.kick_enabled: QCheckBox | None = None
        self.kick_intensity_slider: QSlider | None = None
        self.kick_threshold_slider: QSlider | None = None

        self.snare_enabled: QCheckBox | None = None
        self.snare_intensity_slider: QSlider | None = None
        self.snare_threshold_slider: QSlider | None = None

        self.hihat_enabled: QCheckBox | None = None
        self.hihat_intensity_slider: QSlider | None = None
        self.hihat_threshold_slider: QSlider | None = None

        self.energy_enabled: QCheckBox | None = None
        self.energy_intensity_slider: QSlider | None = None
        self.energy_threshold_slider: QSlider | None = None

        self._setup_ui()

    def _setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout()

        # Trigger-Beschreibungen
        trigger_descriptions = {
            "Beat": {
                "description": "Reagiert auf erkannte Beats/Taktschläge im Audio.",
                "intensity": "Wie stark beeinflusst dieser Trigger die Schnitte?\n"
                "Hoch = Mehr Schnitte auf Beats\n"
                "Niedrig = Weniger Einfluss auf Schnittplatzierung",
                "threshold": "Ab welcher Beat-Stärke wird ein Schnitt ausgelöst?\n"
                "Hoch = Nur auf starke Beats schneiden\n"
                "Niedrig = Auch auf schwächere Beats reagieren",
            },
            "Onset": {
                "description": "Reagiert auf plötzliche Klangänderungen (z.B. Instrumenteneinsatz).",
                "intensity": "Wie stark beeinflusst dieser Trigger die Schnitte?\n"
                "Hoch = Viele Schnitte bei Klangeinsätzen\n"
                "Niedrig = Weniger reaktiv auf Klangeinsätze",
                "threshold": "Ab welcher Lautstärke-Änderung wird geschnitten?\n"
                "Hoch = Nur bei starken Einsätzen\n"
                "Niedrig = Auch bei subtilen Änderungen",
            },
            "Kick": {
                "description": "Reagiert speziell auf Kick-Drums/Bassdrums.",
                "intensity": "Wie stark beeinflusst die Kick die Schnitte?\n"
                "Hoch = Schnitte folgen dem Kick-Rhythmus\n"
                "Niedrig = Kick wird weniger beachtet",
                "threshold": "Ab welcher Kick-Lautstärke wird geschnitten?\n"
                "Hoch = Nur bei lauten Kicks\n"
                "Niedrig = Auch bei leisen Kicks",
            },
            "Snare": {
                "description": "Reagiert speziell auf Snare-Drums (oft auf 2 und 4).",
                "intensity": "Wie stark beeinflusst die Snare die Schnitte?\n"
                "Hoch = Schnitte betonen Snare-Schläge\n"
                "Niedrig = Snare wird weniger beachtet",
                "threshold": "Ab welcher Snare-Lautstärke wird geschnitten?\n"
                "Hoch = Nur bei lauten Snares\n"
                "Niedrig = Auch bei leisen/Ghost-Notes",
            },
            "HiHat": {
                "description": "Reagiert auf Hi-Hat Patterns (schnelles Taktgefühl).",
                "intensity": "Wie stark beeinflusst die Hi-Hat die Schnitte?\n"
                "Hoch = Schnelle Schnitte im Hi-Hat-Rhythmus\n"
                "Niedrig = Hi-Hat wird ignoriert",
                "threshold": "Ab welcher Hi-Hat-Lautstärke wird reagiert?\n"
                "Hoch = Nur bei offenen/lauten Hi-Hats\n"
                "Niedrig = Auch bei geschlossenen Hi-Hats",
            },
            "Energy": {
                "description": "Reagiert auf Energie-Level der Musik (laut/leise, intensiv/ruhig).",
                "intensity": "Wie stark beeinflusst die Energie das Schnitttempo?\n"
                "Hoch = Schnellere Schnitte bei hoher Energie\n"
                "Niedrig = Konstantes Tempo unabhängig von Energie",
                "threshold": "Ab welchem Energie-Level ändert sich das Verhalten?\n"
                "Hoch = Nur bei sehr hoher Energie reagieren\n"
                "Niedrig = Schon bei mittlerer Energie reagieren",
            },
        }

        # Beat Trigger
        (
            beat_group,
            self.beat_enabled,
            self.beat_intensity_slider,
            self.beat_threshold_slider,
        ) = self._create_trigger_controls("Beat", trigger_descriptions["Beat"], 80, 30)
        layout.addWidget(beat_group)

        # Onset Trigger
        (
            onset_group,
            self.onset_enabled,
            self.onset_intensity_slider,
            self.onset_threshold_slider,
        ) = self._create_trigger_controls("Onset", trigger_descriptions["Onset"], 70, 40)
        layout.addWidget(onset_group)

        # Kick Trigger
        (
            kick_group,
            self.kick_enabled,
            self.kick_intensity_slider,
            self.kick_threshold_slider,
        ) = self._create_trigger_controls("Kick", trigger_descriptions["Kick"], 85, 35)
        layout.addWidget(kick_group)

        # Snare Trigger
        (
            snare_group,
            self.snare_enabled,
            self.snare_intensity_slider,
            self.snare_threshold_slider,
        ) = self._create_trigger_controls("Snare", trigger_descriptions["Snare"], 85, 35)
        layout.addWidget(snare_group)

        # HiHat Trigger
        (
            hihat_group,
            self.hihat_enabled,
            self.hihat_intensity_slider,
            self.hihat_threshold_slider,
        ) = self._create_trigger_controls("HiHat", trigger_descriptions["HiHat"], 75, 40)
        layout.addWidget(hihat_group)

        # Energy Trigger
        (
            energy_group,
            self.energy_enabled,
            self.energy_intensity_slider,
            self.energy_threshold_slider,
        ) = self._create_trigger_controls("Energy", trigger_descriptions["Energy"], 90, 25)
        layout.addWidget(energy_group)

        layout.addStretch()
        self.setLayout(layout)

    def _create_trigger_controls(
        self,
        trigger_name: str,
        desc: dict[str, str],
        default_intensity: int = 80,
        default_threshold: int = 30,
    ) -> tuple[QGroupBox, QCheckBox, QSlider, QSlider]:
        """Create trigger control group."""
        group = QGroupBox(f"{trigger_name} Trigger")
        group.setToolTip(desc["description"])
        group_layout = QVBoxLayout()

        # Enabled checkbox
        enabled_cb = QCheckBox("Aktiviert")
        enabled_cb.setChecked(True)
        enabled_cb.setToolTip(
            f"Aktiviert/deaktiviert den {trigger_name}-Trigger.\n{desc['description']}"
        )
        enabled_cb.stateChanged.connect(
            lambda state, name=trigger_name.lower(): self.parameter_changed.emit(
                f"{name}_enabled", state == Qt.CheckState.Checked.value
            )
        )
        group_layout.addWidget(enabled_cb)

        # Intensity slider
        intensity_layout = QVBoxLayout()
        intensity_header = QHBoxLayout()
        intensity_label = QLabel("Intensity:")
        intensity_label.setToolTip(desc["intensity"])
        intensity_header.addWidget(intensity_label)
        intensity_value = QLabel(f"{default_intensity}%")
        intensity_value.setStyleSheet("font-weight: bold;")
        intensity_header.addWidget(intensity_value)
        intensity_header.addStretch()
        intensity_layout.addLayout(intensity_header)

        intensity_slider = QSlider(Qt.Orientation.Horizontal)
        intensity_slider.setRange(0, 100)
        intensity_slider.setValue(default_intensity)
        intensity_slider.setToolTip(desc["intensity"])
        intensity_slider.valueChanged.connect(
            lambda value, label=intensity_value, name=trigger_name.lower(): (
                label.setText(f"{value}%"),
                self.parameter_changed.emit(f"{name}_intensity", value),
            )
        )
        intensity_layout.addWidget(intensity_slider)
        group_layout.addLayout(intensity_layout)

        # Threshold slider
        threshold_layout = QVBoxLayout()
        threshold_header = QHBoxLayout()
        threshold_label = QLabel("Threshold:")
        threshold_label.setToolTip(desc["threshold"])
        threshold_header.addWidget(threshold_label)
        threshold_value = QLabel(f"{default_threshold}%")
        threshold_value.setStyleSheet("font-weight: bold;")
        threshold_header.addWidget(threshold_value)
        threshold_header.addStretch()
        threshold_layout.addLayout(threshold_header)

        threshold_slider = QSlider(Qt.Orientation.Horizontal)
        threshold_slider.setRange(0, 100)
        threshold_slider.setValue(default_threshold)
        threshold_slider.setToolTip(desc["threshold"])
        threshold_slider.valueChanged.connect(
            lambda value, label=threshold_value, name=trigger_name.lower(): (
                label.setText(f"{value}%"),
                self.parameter_changed.emit(f"{name}_threshold", value),
            )
        )
        threshold_layout.addWidget(threshold_slider)
        group_layout.addLayout(threshold_layout)

        group.setLayout(group_layout)
        return group, enabled_cb, intensity_slider, threshold_slider

    def get_trigger_values(self) -> dict[str, object]:
        """Get current trigger parameter values."""
        params = {}

        if self.beat_enabled:
            params["beat_enabled"] = self.beat_enabled.isChecked()
            params["beat_intensity"] = self.beat_intensity_slider.value()
            params["beat_threshold"] = self.beat_threshold_slider.value()

        if self.onset_enabled:
            params["onset_enabled"] = self.onset_enabled.isChecked()
            params["onset_intensity"] = self.onset_intensity_slider.value()
            params["onset_threshold"] = self.onset_threshold_slider.value()

        if self.kick_enabled:
            params["kick_enabled"] = self.kick_enabled.isChecked()
            params["kick_intensity"] = self.kick_intensity_slider.value()
            params["kick_threshold"] = self.kick_threshold_slider.value()

        if self.snare_enabled:
            params["snare_enabled"] = self.snare_enabled.isChecked()
            params["snare_intensity"] = self.snare_intensity_slider.value()
            params["snare_threshold"] = self.snare_threshold_slider.value()

        if self.hihat_enabled:
            params["hihat_enabled"] = self.hihat_enabled.isChecked()
            params["hihat_intensity"] = self.hihat_intensity_slider.value()
            params["hihat_threshold"] = self.hihat_threshold_slider.value()

        if self.energy_enabled:
            params["energy_enabled"] = self.energy_enabled.isChecked()
            params["energy_intensity"] = self.energy_intensity_slider.value()
            params["energy_threshold"] = self.energy_threshold_slider.value()

        return params
