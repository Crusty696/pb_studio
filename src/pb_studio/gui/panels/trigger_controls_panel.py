"""
Trigger Controls Panel

Reusable panel for audio trigger configuration.
Extracts repeated trigger UI code from ParameterDashboardWidget.
"""

from dataclasses import dataclass

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


@dataclass
class TriggerConfig:
    """Configuration for a single trigger type."""

    name: str
    display_name: str
    tooltip: str
    default_enabled: bool = True
    default_intensity: int = 50
    default_threshold: int = 50


# Predefined trigger configurations
TRIGGER_CONFIGS = {
    "beat": TriggerConfig(
        name="beat",
        display_name="Beat Triggers",
        tooltip="Triggers on detected beats in the music",
        default_enabled=True,
        default_intensity=70,
        default_threshold=30,
    ),
    "onset": TriggerConfig(
        name="onset",
        display_name="Onset Triggers",
        tooltip="Triggers on sudden sound changes (transients)",
        default_enabled=True,
        default_intensity=50,
        default_threshold=40,
    ),
    "kick": TriggerConfig(
        name="kick",
        display_name="Kick Drum",
        tooltip="Triggers specifically on kick drum hits",
        default_enabled=False,
        default_intensity=60,
        default_threshold=50,
    ),
    "snare": TriggerConfig(
        name="snare",
        display_name="Snare Drum",
        tooltip="Triggers specifically on snare drum hits",
        default_enabled=False,
        default_intensity=60,
        default_threshold=50,
    ),
    "hihat": TriggerConfig(
        name="hihat",
        display_name="Hi-Hat",
        tooltip="Triggers on hi-hat cymbals",
        default_enabled=False,
        default_intensity=40,
        default_threshold=60,
    ),
    "energy": TriggerConfig(
        name="energy",
        display_name="Energy Triggers",
        tooltip="Triggers based on overall energy level changes",
        default_enabled=True,
        default_intensity=50,
        default_threshold=50,
    ),
}


class SingleTriggerControl(QWidget):
    """
    Control widget for a single trigger type.

    Contains:
    - Enable checkbox
    - Intensity slider
    - Threshold slider

    Signals:
        enabled_changed(str, bool): Trigger name and new enabled state
        intensity_changed(str, int): Trigger name and new intensity (0-100)
        threshold_changed(str, int): Trigger name and new threshold (0-100)
    """

    enabled_changed = pyqtSignal(str, bool)
    intensity_changed = pyqtSignal(str, int)
    threshold_changed = pyqtSignal(str, int)

    def __init__(self, config: TriggerConfig, parent: QWidget | None = None):
        super().__init__(parent)
        self.config = config
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Enable checkbox
        self.enabled_checkbox = QCheckBox(self.config.display_name)
        self.enabled_checkbox.setChecked(self.config.default_enabled)
        self.enabled_checkbox.setToolTip(self.config.tooltip)
        self.enabled_checkbox.toggled.connect(self._on_enabled_changed)
        layout.addWidget(self.enabled_checkbox)

        # Sliders container (only visible when enabled)
        self.sliders_widget = QWidget()
        sliders_layout = QVBoxLayout(self.sliders_widget)
        sliders_layout.setContentsMargins(20, 0, 0, 0)  # Indent

        # Intensity slider
        intensity_layout = QHBoxLayout()
        intensity_label = QLabel("Intensity:")
        intensity_label.setFixedWidth(70)
        intensity_layout.addWidget(intensity_label)

        self.intensity_slider = QSlider(Qt.Orientation.Horizontal)
        self.intensity_slider.setRange(0, 100)
        self.intensity_slider.setValue(self.config.default_intensity)
        self.intensity_slider.valueChanged.connect(self._on_intensity_changed)
        intensity_layout.addWidget(self.intensity_slider)

        self.intensity_value = QLabel(f"{self.config.default_intensity}%")
        self.intensity_value.setFixedWidth(40)
        intensity_layout.addWidget(self.intensity_value)
        sliders_layout.addLayout(intensity_layout)

        # Threshold slider
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("Threshold:")
        threshold_label.setFixedWidth(70)
        threshold_layout.addWidget(threshold_label)

        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(self.config.default_threshold)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        threshold_layout.addWidget(self.threshold_slider)

        self.threshold_value = QLabel(f"{self.config.default_threshold}%")
        self.threshold_value.setFixedWidth(40)
        threshold_layout.addWidget(self.threshold_value)
        sliders_layout.addLayout(threshold_layout)

        layout.addWidget(self.sliders_widget)

        # Set initial visibility
        self.sliders_widget.setVisible(self.config.default_enabled)

    def _on_enabled_changed(self, enabled: bool):
        self.sliders_widget.setVisible(enabled)
        self.enabled_changed.emit(self.config.name, enabled)

    def _on_intensity_changed(self, value: int):
        self.intensity_value.setText(f"{value}%")
        self.intensity_changed.emit(self.config.name, value)

    def _on_threshold_changed(self, value: int):
        self.threshold_value.setText(f"{value}%")
        self.threshold_changed.emit(self.config.name, value)

    def get_values(self) -> dict:
        """Get current trigger values."""
        return {
            "enabled": self.enabled_checkbox.isChecked(),
            "intensity": self.intensity_slider.value(),
            "threshold": self.threshold_slider.value(),
        }

    def set_values(self, enabled: bool, intensity: int, threshold: int):
        """Set trigger values programmatically."""
        self.enabled_checkbox.setChecked(enabled)
        self.intensity_slider.setValue(intensity)
        self.threshold_slider.setValue(threshold)


class TriggerControlsPanel(QGroupBox):
    """
    Panel containing all trigger controls.

    Signals:
        trigger_changed(str, str, object): trigger_name, param_name, value
    """

    trigger_changed = pyqtSignal(str, str, object)

    def __init__(self, triggers: list | None = None, parent: QWidget | None = None):
        """
        Initialize trigger controls panel.

        Args:
            triggers: List of trigger names to include (default: all)
            parent: Parent widget
        """
        super().__init__("Trigger Settings", parent)
        self.triggers = triggers or list(TRIGGER_CONFIGS.keys())
        self._controls: dict[str, SingleTriggerControl] = {}
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        for trigger_name in self.triggers:
            if trigger_name not in TRIGGER_CONFIGS:
                continue

            config = TRIGGER_CONFIGS[trigger_name]
            control = SingleTriggerControl(config)

            # Connect signals
            control.enabled_changed.connect(lambda n, v: self.trigger_changed.emit(n, "enabled", v))
            control.intensity_changed.connect(
                lambda n, v: self.trigger_changed.emit(n, "intensity", v)
            )
            control.threshold_changed.connect(
                lambda n, v: self.trigger_changed.emit(n, "threshold", v)
            )

            self._controls[trigger_name] = control
            layout.addWidget(control)

        layout.addStretch()

    def get_all_values(self) -> dict[str, dict]:
        """Get values for all triggers."""
        return {name: control.get_values() for name, control in self._controls.items()}

    def set_values(self, trigger_name: str, **kwargs):
        """Set values for a specific trigger."""
        if trigger_name in self._controls:
            control = self._controls[trigger_name]
            control.set_values(
                enabled=kwargs.get("enabled", True),
                intensity=kwargs.get("intensity", 50),
                threshold=kwargs.get("threshold", 50),
            )
