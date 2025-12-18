"""
Theme Manager für PB_studio GUI

Verwaltung von Dark/Light-Themes mit Qt StyleSheets.

Unterstützt:
- Dark Theme (dunkle Farben, hoher Kontrast)
- Light Theme (helle Farben, klassisch)
- System-Theme (folgt OS-Einstellung)
- Persistente Speicherung der Theme-Wahl

Technische Details:
- PyQt6 StyleSheet-basiert
- JSON-Config für Persistenz
- Automatische Widget-Aktualisierung
- Event-System für Theme-Änderungen

Author: PB_studio Development Team
"""

import json
import logging
from enum import Enum
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

# Import template-based stylesheets (eliminates 500+ lines of duplicated code)
from .theme_colors import DARK_STYLESHEET, LIGHT_STYLESHEET

logger = logging.getLogger(__name__)


class ThemeMode(Enum):
    """Available theme modes."""

    DARK = "dark"
    LIGHT = "light"
    SYSTEM = "system"  # Folgt OS-Einstellung


class ThemeManager(QObject):
    """
    Verwaltet Themes für die GUI-Anwendung.

    Features:
    - Dark/Light-Mode mit StyleSheets
    - Persistente Speicherung in JSON
    - Signal-basierte Theme-Updates
    - System-Theme-Detection

    Signals:
        theme_changed: Wird ausgelöst bei Theme-Wechsel (ThemeMode)

    Example:
        >>> theme_manager = ThemeManager()
        >>> theme_manager.theme_changed.connect(on_theme_changed)
        >>> theme_manager.set_theme(ThemeMode.DARK)
    """

    # Signal für Theme-Änderungen
    theme_changed = pyqtSignal(object)  # ThemeMode

    def __init__(self, config_file: Path | None = None):
        """
        Initialisiert Theme-Manager.

        Args:
            config_file: Pfad zur Theme-Config (default: .pb_studio_theme.json)
        """
        super().__init__()

        if config_file is None:
            config_file = Path.home() / ".pb_studio_theme.json"

        self.config_file = config_file
        self.current_theme = ThemeMode.DARK  # Default to Dark theme

        # Theme-Definitions (using template-based stylesheets)
        # This eliminates 500+ lines of duplicated stylesheet code
        self._dark_stylesheet = DARK_STYLESHEET
        self._light_stylesheet = LIGHT_STYLESHEET

        # Load saved theme
        self._load_theme_config()

        logger.info(f"ThemeManager initialisiert: {self.current_theme.value}")

    # NOTE: Stylesheet generation moved to theme_colors.py (DRY principle)
    # The _generate_dark_stylesheet and _generate_light_stylesheet methods
    # were removed as dead code - stylesheets now imported from theme_colors

    def set_theme(self, theme_mode: ThemeMode):
        """
        Setzt Theme-Modus.

        Args:
            theme_mode: Gewünschter Theme-Modus
        """
        self.current_theme = theme_mode

        # Apply stylesheet
        app = QApplication.instance()
        if not app:
            logger.warning("QApplication nicht gefunden, Theme nicht angewendet")
            return

        if theme_mode == ThemeMode.DARK:
            app.setStyleSheet(self._dark_stylesheet)
            logger.info("Dark Theme aktiviert")
        elif theme_mode == ThemeMode.LIGHT:
            app.setStyleSheet(self._light_stylesheet)
            logger.info("Light Theme aktiviert")
        elif theme_mode == ThemeMode.SYSTEM:
            # System-Theme (würde OS-Einstellung erfordern)
            # Fallback zu Light
            app.setStyleSheet(self._light_stylesheet)
            logger.info("System Theme aktiviert (Fallback: Light)")

        # Save theme config
        self._save_theme_config()

        # Emit signal
        self.theme_changed.emit(theme_mode)

    def toggle_theme(self):
        """
        Wechselt zwischen Dark und Light Theme.
        """
        if self.current_theme == ThemeMode.DARK:
            self.set_theme(ThemeMode.LIGHT)
        else:
            self.set_theme(ThemeMode.DARK)

    def get_current_theme(self) -> ThemeMode:
        """
        Gibt aktuellen Theme-Modus zurück.

        Returns:
            ThemeMode
        """
        return self.current_theme

    def _load_theme_config(self):
        """Lädt gespeicherte Theme-Config."""
        if not self.config_file.exists():
            return

        try:
            data = json.loads(self.config_file.read_text(encoding="utf-8"))
            theme_str = data.get("theme", "dark")
            self.current_theme = ThemeMode(theme_str)
            logger.debug(f"Theme-Config geladen: {theme_str}")

            # Apply theme
            self.set_theme(self.current_theme)

        except Exception as e:
            logger.warning(f"Theme-Config konnte nicht geladen werden: {e}")

    def _save_theme_config(self):
        """Speichert Theme-Config."""
        try:
            data = {"theme": self.current_theme.value}
            self.config_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            logger.debug(f"Theme-Config gespeichert: {self.current_theme.value}")
        except Exception as e:
            logger.warning(f"Theme-Config konnte nicht gespeichert werden: {e}")
