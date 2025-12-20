"""
Zentrales Konfigurations-Management für PB_studio

Verwendet configparser für .ini-Dateien.
Automatische Erstellung von Standardkonfiguration.
"""

import configparser
from pathlib import Path
from typing import Any

from ..utils.logger import get_logger

logger = get_logger()


class Config:
    """Zentrale Konfigurationsklasse für PB_studio."""

    def __init__(self, config_file: str = "config.ini"):
        """
        Initialisiert die Konfiguration.

        Args:
            config_file: Pfad zur Konfigurationsdatei
        """
        self.config_file = Path(config_file)
        self.config = configparser.ConfigParser()
        self.load()

    def load(self) -> None:
        """Lädt die Konfiguration aus der Datei oder erstellt Standardkonfiguration."""
        if self.config_file.exists():
            self.config.read(self.config_file, encoding="utf-8")
            logger.info(f"Konfiguration aus {self.config_file} geladen")
        else:
            logger.info(
                f"Konfigurationsdatei {self.config_file} nicht gefunden. "
                "Erstelle Standardkonfiguration."
            )
            self._create_default_config()
            self.save()

    def _create_default_config(self) -> None:
        """Erstellt die Standardkonfiguration."""
        # Paths
        self.config["Paths"] = {
            "audio_dir": "audio",
            "video_dir": "video",
            "export_dir": "exports",
            "temp_dir": "temp",
            "cache_dir": "cache",
        }

        # Database
        self.config["Database"] = {
            "sqlite_path": "data/project.db",
            "duckdb_path": "data/analytics.duckdb",
        }

        # Hardware
        self.config["Hardware"] = {
            "compute_device": "cpu",  # cpu, cuda, cuda:0, etc.
            "use_gpu_rendering": "true",
            "gpu_memory_reserve": "0.20",  # 20% Reserve für System
        }

        # Audio Analysis
        self.config["Audio"] = {
            "sample_rate": "22050",
            "hop_length": "512",
            "cache_analysis": "true",
        }

        # Video Processing
        self.config["Video"] = {
            "thumbnail_size": "320x180",
            "preview_quality": "medium",
        }

        # GUI
        self.config["GUI"] = {
            "theme": "dark",
            "timeline_height": "200",
            "waveform_color": "#00FF00",
        }

        # Logging
        self.config["Logging"] = {
            "console_level": "INFO",
            "file_level": "DEBUG",
            "log_file": "app.log",
        }

    def save(self) -> None:
        """Speichert die Konfiguration in die Datei."""
        # Erstelle Verzeichnis falls nötig
        self.config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.config_file, "w", encoding="utf-8") as f:
            self.config.write(f)
        logger.info(f"Konfiguration in {self.config_file} gespeichert")

    def get(self, section: str, option: str, default: Any | None = None) -> str | None:
        """
        Holt einen Konfigurationswert.

        Args:
            section: Section-Name
            option: Option-Name
            default: Default-Wert falls nicht gefunden

        Returns:
            Konfigurationswert oder default
        """
        try:
            return self.config.get(section, option)
        except (configparser.NoSectionError, configparser.NoOptionError):
            logger.warning(
                f"Konfigurationswert [{section}] {option} nicht gefunden. "
                f"Verwende Default: {default}"
            )
            return default

    def get_int(self, section: str, option: str, default: int | None = None) -> int | None:
        """
        Holt einen Integer-Konfigurationswert.

        Args:
            section: Section-Name
            option: Option-Name
            default: Default-Wert

        Returns:
            Integer-Wert oder default
        """
        try:
            return self.config.getint(section, option)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return default

    def get_float(self, section: str, option: str, default: float | None = None) -> float | None:
        """
        Holt einen Float-Konfigurationswert.

        Args:
            section: Section-Name
            option: Option-Name
            default: Default-Wert

        Returns:
            Float-Wert oder default
        """
        try:
            return self.config.getfloat(section, option)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return default

    def get_bool(self, section: str, option: str, default: bool | None = None) -> bool | None:
        """
        Holt einen Boolean-Konfigurationswert.

        Args:
            section: Section-Name
            option: Option-Name
            default: Default-Wert

        Returns:
            Boolean-Wert oder default
        """
        try:
            return self.config.getboolean(section, option)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return default

    def set(self, section: str, option: str, value: Any) -> None:
        """
        Setzt einen Konfigurationswert.

        Args:
            section: Section-Name
            option: Option-Name
            value: Zu setzender Wert
        """
        if not self.config.has_section(section):
            self.config.add_section(section)
        self.config.set(section, option, str(value))
        logger.debug(f"Konfiguration gesetzt: [{section}] {option} = {value}")

    def __repr__(self) -> str:
        """String-Repräsentation."""
        sections = ", ".join(self.config.sections())
        return f"Config(file='{self.config_file}', sections=[{sections}])"


# Globale Konfigurationsinstanz
_config = None


def get_config(config_file: str = "config.ini") -> Config:
    """
    Gibt die globale Konfigurationsinstanz zurück.

    Args:
        config_file: Pfad zur Konfigurationsdatei

    Returns:
        Config-Instanz
    """
    global _config
    if _config is None:
        _config = Config(config_file)
    return _config


# Alias for backwards compatibility
load_config = get_config
