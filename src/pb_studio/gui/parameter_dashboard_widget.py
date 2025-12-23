"""
Parameter Dashboard Widget für PB_studio

Steuerzentrale für Pacing-Parameter und Regel-Konfiguration.
Ermöglicht Echtzeit-Anpassungen der Schnitt-Parameter.

Features:
- BPM und Beat-Grid Anzeige
- Energie-Kurven Visualisierung
- Regel-basierte Pacing-Controls
- Cut-Parameter (Dauer, Tempo, etc.)
- Preset-Management
- Real-time Parameter-Updates

Author: PB_studio Development Team
"""

from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..utils.logger import get_logger
from .widgets.beat_grid_panel import BeatGridPanel
from .widgets.cut_parameter_panel import CutParameterPanel
from .widgets.energy_curve_panel import EnergyCurvePanel
from .widgets.preset_panel import PresetPanel
from .widgets.trigger_control_panel import TriggerControlPanel

logger = get_logger(__name__)


class ParameterDashboardWidget(QWidget):
    """
    Parameter dashboard for pacing control.

    Signals:
      parameter_changed: Emitted when a parameter changes (param_name, value)
      rule_toggled: Emitted when a rule is enabled/disabled (rule_name, enabled)
      preset_loaded: Emitted when a preset is loaded (preset_name)
    """

    parameter_changed = pyqtSignal(str, object)
    rule_toggled = pyqtSignal(str, bool)
    preset_loaded = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        logger.info("Initializing ParameterDashboardWidget")

        # State
        self.current_bpm: float = 120.0
        self.current_energy: float = 0.5
        self.active_rules: dict[str, bool] = {}

        # Sub-panels
        self.beat_grid_panel: BeatGridPanel | None = None
        self.energy_curve_panel: EnergyCurvePanel | None = None
        self.trigger_control_panel: TriggerControlPanel | None = None
        self.cut_parameter_panel: CutParameterPanel | None = None
        self.preset_panel: PresetPanel | None = None

        self._init_ui()
        logger.info("ParameterDashboardWidget initialization complete")

    def _init_ui(self):
        """Initialize UI components."""
        main_layout = QVBoxLayout()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        tab_widget = QTabWidget()

        # Tab 1: Audio & Energy
        tab_widget.addTab(self._create_audio_energy_tab(), "Audio & Energy")

        # Tab 2: Cut Parameters
        tab_widget.addTab(self._create_cut_params_tab(), "Cut Parameters")

        # Tab 3: Triggers
        tab_widget.addTab(self._create_triggers_tab(), "Triggers")

        # Tab 4: Presets
        tab_widget.addTab(self._create_presets_tab(), "Presets")

        scroll_content = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(tab_widget)
        scroll_content.setLayout(layout)
        scroll_area.setWidget(scroll_content)

        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)
        logger.debug("Parameter dashboard UI initialized")

    def _create_audio_energy_tab(self) -> QWidget:
        """Create Audio & Energy tab."""
        tab = QWidget()
        layout = QVBoxLayout()

        beat_grid_group = QGroupBox("Audio Information")
        beat_grid_layout = QVBoxLayout()
        self.beat_grid_panel = BeatGridPanel()
        beat_grid_layout.addWidget(self.beat_grid_panel)
        beat_grid_group.setLayout(beat_grid_layout)
        layout.addWidget(beat_grid_group)

        energy_curve_group = QGroupBox("Energy Curve")
        energy_curve_layout = QVBoxLayout()
        self.energy_curve_panel = EnergyCurvePanel()
        energy_curve_layout.addWidget(self.energy_curve_panel)
        energy_curve_group.setLayout(energy_curve_layout)
        layout.addWidget(energy_curve_group)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def _create_cut_params_tab(self) -> QWidget:
        """Create Cut Parameters tab."""
        tab = QWidget()
        layout = QVBoxLayout()

        cut_params_group = QGroupBox("Cut Parameters")
        cut_params_layout = QVBoxLayout()
        self.cut_parameter_panel = CutParameterPanel()
        self.cut_parameter_panel.parameter_changed.connect(self._forward_parameter_changed)
        cut_params_layout.addWidget(self.cut_parameter_panel)
        cut_params_group.setLayout(cut_params_layout)
        layout.addWidget(cut_params_group)

        rules_group = QGroupBox("Pacing Rules")
        rules_layout = QVBoxLayout()
        for rule_name in [
            "Beat Synchronization",
            "Energy Following",
            "Phrase Alignment",
            "Duration Constraints",
            "Variety Enforcement",
        ]:
            checkbox = QCheckBox(rule_name)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(
                lambda state, name=rule_name: self._on_rule_toggled(
                    name, state == Qt.CheckState.Checked.value
                )
            )
            rules_layout.addWidget(checkbox)
            self.active_rules[rule_name] = True
        rules_group.setLayout(rules_layout)
        layout.addWidget(rules_group)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def _create_triggers_tab(self) -> QWidget:
        """Create Triggers tab."""
        tab = QWidget()
        layout = QVBoxLayout()

        triggers_group = QGroupBox("Trigger Controls")
        triggers_layout = QVBoxLayout()
        self.trigger_control_panel = TriggerControlPanel()
        self.trigger_control_panel.parameter_changed.connect(self._forward_parameter_changed)
        triggers_layout.addWidget(self.trigger_control_panel)
        triggers_group.setLayout(triggers_layout)
        layout.addWidget(triggers_group)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def _create_presets_tab(self) -> QWidget:
        """Create Presets tab."""
        tab = QWidget()
        layout = QVBoxLayout()

        presets_group = QGroupBox("Presets")
        presets_layout = QVBoxLayout()
        self.preset_panel = PresetPanel()
        self.preset_panel.preset_selected.connect(self._on_preset_loaded)
        self.preset_panel.set_callbacks(self._get_preset_data, self._apply_preset_data)
        presets_layout.addWidget(self.preset_panel)
        presets_group.setLayout(presets_layout)
        layout.addWidget(presets_group)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def _forward_parameter_changed(self, param_name: str, value: object):
        self.parameter_changed.emit(param_name, value)

    def _on_rule_toggled(self, rule_name: str, enabled: bool):
        self.active_rules[rule_name] = enabled
        self.rule_toggled.emit(rule_name, enabled)
        logger.info(f"Rule {rule_name}: {'enabled' if enabled else 'disabled'}")

    def _on_preset_loaded(self, preset_name: str):
        self.preset_loaded.emit(preset_name)

    def _get_preset_data(self) -> dict:
        return self.cut_parameter_panel.get_cut_parameters() if self.cut_parameter_panel else {}

    def _apply_preset_data(self, params: dict):
        if not self.cut_parameter_panel:
            return

        # Helper to safely apply scaler value
        def apply_scaled_value(slider, value, scale_factor):
            if isinstance(value, float):
                # Custom presets (floats) -> scale to int
                target = int(round(value * scale_factor))
            elif isinstance(value, int):
                # Built-in presets (ints) -> use directly
                target = value
            else:
                return

            slider.setValue(target)

        if "cut_duration" in params and self.cut_parameter_panel.cut_duration_slider:
            apply_scaled_value(
                self.cut_parameter_panel.cut_duration_slider,
                params["cut_duration"],
                10.0
            )

        if "tempo" in params and self.cut_parameter_panel.tempo_slider:
            apply_scaled_value(
                self.cut_parameter_panel.tempo_slider,
                params["tempo"],
                100.0
            )

    def set_bpm(self, bpm: float):
        self.current_bpm = bpm
        if self.beat_grid_panel:
            self.beat_grid_panel.set_bpm(bpm)

    def update_bpm(self, bpm: float):
        self.set_bpm(bpm)

    def update_beatgrid(self, beat_times: list[float]):
        if self.beat_grid_panel:
            self.beat_grid_panel.set_beatgrid(beat_times)

    def set_energy_curve(self, energy_levels: list[float]):
        if self.energy_curve_panel:
            self.energy_curve_panel.set_energy_curve(energy_levels)
            if energy_levels:
                self.current_energy = sum(energy_levels) / len(energy_levels)

    def update_energy_curve(self, energy_levels: list[float]):
        self.set_energy_curve(energy_levels)

    def get_parameters(self) -> dict[str, Any]:
        """Get current parameter values."""
        params = {
            "bpm": self.current_bpm,
            "active_rules": self.active_rules.copy(),
        }

        if self.cut_parameter_panel:
            params.update(self.cut_parameter_panel.get_cut_parameters())

        if self.trigger_control_panel:
            params.update(self.trigger_control_panel.get_trigger_values())

        return params

    def get_parameter_value(self, name: str) -> Any | None:
        """Get single parameter value (for Undo/Redo)."""
        return self.get_parameters().get(name)

    def set_parameter_value(self, name: str, value: object, emit: bool = True) -> bool:
        """Set parameter value programmatically (Undo/Redo support)."""
        try:
            if name == "bpm":
                self.set_bpm(float(value))
                return True

            # Delegate to cut parameter panel
            if self.cut_parameter_panel:
                if name in [
                    "cut_duration",
                    "tempo",
                    "pacing_mode",
                    "motion_matching_enabled",
                    "structure_awareness_enabled",
                ]:
                    return self._set_cut_parameter(name, value, emit)

            # Delegate to trigger panel
            if self.trigger_control_panel:
                if any(
                    name.startswith(prefix)
                    for prefix in ["beat_", "onset_", "kick_", "snare_", "hihat_", "energy_"]
                ):
                    return self._set_trigger_parameter(name, value, emit)

            return False
        except Exception as exc:
            logger.warning(f"set_parameter_value failed ({name}): {exc}")
            return False

    def _set_cut_parameter(self, name: str, value: object, emit: bool) -> bool:
        """Set cut parameter value."""
        panel = self.cut_parameter_panel
        if not panel:
            return False

        if name == "cut_duration" and panel.cut_duration_slider:
            target = int(round(float(value) * 10.0))
            prev = panel.cut_duration_slider.blockSignals(not emit)
            panel.cut_duration_slider.setValue(target)
            panel.cut_duration_slider.blockSignals(prev)
            return True

        if name == "tempo" and panel.tempo_slider:
            target = int(round(float(value) * 100.0))
            prev = panel.tempo_slider.blockSignals(not emit)
            panel.tempo_slider.setValue(target)
            panel.tempo_slider.blockSignals(prev)
            return True

        return False

    def _set_trigger_parameter(self, name: str, value: object, emit: bool) -> bool:
        """Set trigger parameter value."""
        # Implementation delegated to trigger panel
        return False
