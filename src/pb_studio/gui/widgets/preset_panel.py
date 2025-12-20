"""
Preset Panel Widget

Preset Save/Load Funktionalität.

Author: PB_studio Development Team
"""

import json
from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...utils.logger import get_logger

logger = get_logger(__name__)


class PresetPanel(QWidget):
    """Preset Save/Load."""

    preset_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.preset_combo: QComboBox | None = None
        self.get_parameters_callback: Callable | None = None
        self.apply_parameters_callback: Callable | None = None

        self._setup_ui()
        self._load_custom_presets()

    def _setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout()

        # Preset Selector
        preset_selector_layout = QHBoxLayout()
        preset_selector_layout.addWidget(QLabel("Select Preset:"))

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(
            [
                "Default",
                "High Energy EDM",
                "Cinematic Slow",
                "Music Video",
                "Dance Performance",
            ]
        )
        self.preset_combo.currentTextChanged.connect(self._on_preset_selected)
        preset_selector_layout.addWidget(self.preset_combo)

        layout.addLayout(preset_selector_layout)

        # Preset action buttons
        preset_buttons_layout = QHBoxLayout()
        save_preset_btn = QPushButton("Save Preset")
        save_preset_btn.clicked.connect(self._save_preset)
        preset_buttons_layout.addWidget(save_preset_btn)

        reset_btn = QPushButton("Reset to Default")
        reset_btn.clicked.connect(self._reset_to_default)
        preset_buttons_layout.addWidget(reset_btn)

        layout.addLayout(preset_buttons_layout)

        self.setLayout(layout)

    def _on_preset_selected(self, preset_name: str):
        """Handle preset selection."""
        if not preset_name or not preset_name.strip():
            return

        logger.info(f"Preset selected: {preset_name}")
        self._apply_preset(preset_name)
        self.preset_selected.emit(preset_name)

    def _apply_preset(self, preset_name: str):
        """Apply preset parameters (built-in or custom)."""
        builtin_presets = {
            "Default": {
                "cut_duration": 40,
                "tempo": 50,
            },
            "High Energy EDM": {
                "cut_duration": 45,
                "tempo": 70,
            },
            "Cinematic Slow": {
                "cut_duration": 60,
                "tempo": 20,
            },
            "Music Video": {
                "cut_duration": 40,
                "tempo": 60,
            },
            "Dance Performance": {
                "cut_duration": 35,
                "tempo": 65,
            },
        }

        if preset_name in builtin_presets:
            params = builtin_presets[preset_name]
            if self.apply_parameters_callback:
                self.apply_parameters_callback(params)
            logger.info(f"Applied built-in preset: {preset_name}")
        else:
            if self._apply_custom_preset(preset_name):
                logger.info(f"Applied custom preset: {preset_name}")
            else:
                logger.warning(f"Preset not found: {preset_name}")

    def _save_preset(self):
        """Save current parameters as custom preset."""
        try:
            name, ok = QInputDialog.getText(
                self, "Preset speichern", "Preset-Name:", text="Mein Preset"
            )

            if not ok or not name.strip():
                logger.debug("Preset save cancelled")
                return

            name = name.strip()

            builtin_presets = [
                "Default",
                "High Energy EDM",
                "Cinematic Slow",
                "Music Video",
                "Dance Performance",
            ]

            if name in builtin_presets:
                QMessageBox.warning(
                    self,
                    "Ungültiger Name",
                    f"'{name}' ist ein vordefiniertes Preset.\nBitte wähle einen anderen Namen.",
                )
                return

            if not self.get_parameters_callback:
                logger.error("No get_parameters_callback set")
                return

            preset_data = self.get_parameters_callback()

            preset_path = self._get_preset_path(name)
            preset_path.parent.mkdir(parents=True, exist_ok=True)

            with open(preset_path, "w", encoding="utf-8") as f:
                json.dump(preset_data, f, indent=2)

            logger.info(f"Preset '{name}' saved to {preset_path}")

            self._load_custom_presets()
            if self.preset_combo:
                self.preset_combo.setCurrentText(name)

            QMessageBox.information(
                self, "Preset gespeichert", f"Preset '{name}' wurde erfolgreich gespeichert."
            )

        except Exception as e:
            logger.error(f"Failed to save preset: {e}", exc_info=True)
            QMessageBox.critical(self, "Fehler", f"Preset konnte nicht gespeichert werden:\n{e}")

    def _reset_to_default(self):
        """Reset all parameters to default values."""
        if self.preset_combo:
            self.preset_combo.setCurrentText("Default")
        logger.info("Reset to default preset")

    def _get_presets_dir(self) -> Path:
        """Get presets directory path."""
        import os

        if os.name == "nt":
            base_dir = Path.home() / "AppData" / "Roaming" / "PB_studio"
        else:
            base_dir = Path.home() / ".pb_studio"

        return base_dir / "presets"

    def _get_preset_path(self, preset_name: str) -> Path:
        """Get file path for a preset."""
        if not preset_name or not isinstance(preset_name, str):
            raise ValueError("Invalid preset name: must be a non-empty string")

        if len(preset_name) > 100:
            raise ValueError("Invalid preset name: exceeds maximum length (100 chars)")

        safe_name = "".join(c for c in preset_name if c.isalnum() or c in (" ", "_", "-"))
        safe_name = safe_name.strip()

        if ".." in preset_name or "/" in preset_name or "\\" in preset_name:
            raise ValueError("Invalid preset name: path traversal not allowed")

        if not safe_name:
            safe_name = "preset"

        preset_path = self._get_presets_dir() / f"{safe_name}.json"
        try:
            preset_path.resolve().relative_to(self._get_presets_dir().resolve())
        except ValueError:
            raise ValueError("Invalid preset name: path traversal detected")

        return preset_path

    def _load_custom_presets(self):
        """Load custom presets from disk and add them to combo box."""
        try:
            presets_dir = self._get_presets_dir()

            if not presets_dir.exists():
                logger.debug("Presets directory does not exist yet")
                return

            builtin_presets = [
                "Default",
                "High Energy EDM",
                "Cinematic Slow",
                "Music Video",
                "Dance Performance",
            ]

            current_selection = self.preset_combo.currentText() if self.preset_combo else ""

            if self.preset_combo:
                self.preset_combo.clear()
                self.preset_combo.addItems(builtin_presets)
                self.preset_combo.insertSeparator(len(builtin_presets))

            preset_files = sorted(presets_dir.glob("*.json"))
            custom_preset_names = []

            for preset_file in preset_files:
                try:
                    preset_name = preset_file.stem

                    with open(preset_file, encoding="utf-8") as f:
                        data = json.load(f)

                    if "cut_duration" in data and "tempo" in data:
                        custom_preset_names.append(preset_name)
                    else:
                        logger.warning(f"Invalid preset file (missing fields): {preset_file}")

                except Exception as e:
                    logger.warning(f"Failed to load preset {preset_file}: {e}")

            if custom_preset_names and self.preset_combo:
                self.preset_combo.addItems(custom_preset_names)
                logger.info(f"Loaded {len(custom_preset_names)} custom presets")

            if self.preset_combo and current_selection:
                idx = self.preset_combo.findText(current_selection)
                if idx >= 0:
                    self.preset_combo.setCurrentIndex(idx)

        except Exception as e:
            logger.error(f"Failed to load custom presets: {e}", exc_info=True)

    def _apply_custom_preset(self, preset_name: str) -> bool:
        """Apply a custom preset from disk."""
        try:
            preset_path = self._get_preset_path(preset_name)

            if not preset_path.exists():
                logger.error(f"Preset file not found: {preset_path}")
                return False

            with open(preset_path, encoding="utf-8") as f:
                data = json.load(f)

            if self.apply_parameters_callback:
                self.apply_parameters_callback(data)

            logger.info(f"Applied custom preset: {preset_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to apply custom preset: {e}", exc_info=True)
            return False

    def set_callbacks(self, get_params: Callable, apply_params: Callable):
        """Set callbacks for getting/applying parameters."""
        self.get_parameters_callback = get_params
        self.apply_parameters_callback = apply_params
