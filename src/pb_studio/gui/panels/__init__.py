"""
GUI Panel Components Package

Extracted panel components from ParameterDashboardWidget for better
code organization and reduced complexity.

Components:
- TriggerControlsPanel: Controls for beat/onset/energy triggers
- CutParametersPanel: Cut duration and tempo settings

Note: BeatGridVisualizerWidget and EnergyCurveWidget remain in
parameter_dashboard_widget.py as they are visualization components
tightly coupled to their parent.
"""

from .cut_parameters_panel import CutParametersPanel
from .trigger_controls_panel import (
    TRIGGER_CONFIGS,
    SingleTriggerControl,
    TriggerConfig,
    TriggerControlsPanel,
)

__all__ = [
    "TriggerControlsPanel",
    "SingleTriggerControl",
    "TriggerConfig",
    "TRIGGER_CONFIGS",
    "CutParametersPanel",
]
