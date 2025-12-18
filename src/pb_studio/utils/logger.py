"""
Logging-System für PB_studio

Basierend auf Python logging mit Console- und File-Handler.
Automatische Log-Rotation und verschiedene Log-Level.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(
    log_file: str = "app.log",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_dir: str = "logs",
) -> logging.Logger:
    """
    Richtet das Logging-System für die Anwendung ein.

    Args:
        log_file: Name der Log-Datei
        console_level: Log-Level für Console-Ausgabe
        file_level: Log-Level für File-Ausgabe
        log_dir: Verzeichnis für Log-Dateien

    Returns:
        Konfigurierter Logger
    """
    # Erstelle Logger
    logger = logging.getLogger("pb_studio")
    logger.setLevel(logging.DEBUG)  # Erfasse alle Levels

    # Entferne bestehende Handler
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console-Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Log-Verzeichnis erstellen
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Datei-Handler mit Rotation (10MB pro Datei, 5 Backups)
    log_filepath = log_path / log_file
    file_handler = RotatingFileHandler(
        log_filepath, maxBytes=10485760, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(file_level)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logging-System initialisiert. Log-Datei: {log_filepath}")
    return logger


# Globaler Logger für die Anwendung
_logger = None


def get_logger(name: str = None) -> logging.Logger:
    """
    Gibt einen Logger zurück (entweder modul-spezifisch oder global).
    Falls noch nicht initialisiert, wird setup_logging() aufgerufen.

    Args:
        name: Optional logger name (typically __name__ from calling module)

    Returns:
        Application logger
    """
    global _logger
    if _logger is None:
        _logger = setup_logging()

    # If name provided, return child logger for that module
    if name:
        return logging.getLogger(f"pb_studio.{name}")

    return _logger
