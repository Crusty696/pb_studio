"""
First-Run Check - Prueft Hardware und Konfiguration beim ersten Start.
"""

import json
import time
from pathlib import Path
from typing import Any

from ..utils.logger import get_logger
from .gpu_backend import get_backend_info, print_backend_info

logger = get_logger(__name__)

FIRST_RUN_FILE = Path.home() / ".pb_studio" / "first_run_complete.json"
LOCK_FILE = Path.home() / ".pb_studio" / ".first_run.lock"

# LOW-06: Schema-Version fuer Validierung
CURRENT_SCHEMA_VERSION = "1.0"
REQUIRED_HARDWARE_KEYS = {"backend", "vendor", "device", "is_gpu"}


def _acquire_lock(timeout: float = 10.0) -> bool:
    """
    MEDIUM-09 FIX: Plattform-unabhaengiges File-Locking.

    Args:
        timeout: Maximale Wartezeit in Sekunden

    Returns:
        True wenn Lock erworben wurde
    """
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            # Versuche Lock-File exklusiv zu erstellen
            LOCK_FILE.touch(exist_ok=False)
            return True
        except FileExistsError:
            # Lock existiert bereits, warte kurz
            time.sleep(0.1)

    logger.warning(f"Konnte Lock nicht erwerben nach {timeout}s")
    return False


def _release_lock():
    """Gibt File-Lock frei."""
    try:
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()
    except Exception as e:
        logger.debug(f"Lock-File konnte nicht geloescht werden: {e}")


def _validate_config(config: dict[str, Any]) -> bool:
    """
    LOW-06 FIX: Validiert Konfigurationsstruktur.

    Args:
        config: Geladene Konfiguration

    Returns:
        True wenn gueltig
    """
    if not isinstance(config, dict):
        return False

    if config.get("version") != CURRENT_SCHEMA_VERSION:
        logger.warning(
            f"Schema-Version mismatch: {config.get('version')} != {CURRENT_SCHEMA_VERSION}"
        )
        return False

    hardware = config.get("hardware", {})
    if not isinstance(hardware, dict):
        return False

    missing_keys = REQUIRED_HARDWARE_KEYS - set(hardware.keys())
    if missing_keys:
        logger.warning(f"Fehlende Hardware-Keys: {missing_keys}")
        return False

    return True


def is_first_run() -> bool:
    """Prueft ob dies der erste Start ist."""
    return not FIRST_RUN_FILE.exists()


def run_first_time_setup() -> dict[str, Any]:
    """
    Fuehrt First-Run Setup durch:
    1. Hardware-Erkennung
    2. Speichert Konfiguration
    3. Zeigt Willkommensnachricht

    Returns:
        Dict mit Hardware-Infos
    """
    logger.info("=" * 60)
    logger.info("PB Studio - First Run Setup")
    logger.info("=" * 60)

    # Hardware erkennen
    hw_info = get_backend_info()

    # Info anzeigen
    print_backend_info()

    # MEDIUM-09 FIX: Nutze File-Locking beim Schreiben
    if not _acquire_lock():
        logger.warning("Setup laeuft bereits in anderem Prozess")
        return hw_info

    try:
        # Konfiguration speichern
        FIRST_RUN_FILE.parent.mkdir(parents=True, exist_ok=True)

        config = {
            "version": CURRENT_SCHEMA_VERSION,
            "hardware": hw_info,
        }

        with open(FIRST_RUN_FILE, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Konfiguration gespeichert: {FIRST_RUN_FILE}")
        logger.info("First Run Setup abgeschlossen")

    finally:
        _release_lock()

    return hw_info


def check_and_setup() -> dict[str, Any]:
    """
    Prueft ob First-Run noetig ist und fuehrt ihn ggf. durch.

    Returns:
        Hardware-Info Dict
    """
    if is_first_run():
        logger.info("Erster Start erkannt - fuehre Setup durch...")
        return run_first_time_setup()
    else:
        # Lade gespeicherte Config
        try:
            with open(FIRST_RUN_FILE) as f:
                config = json.load(f)

            # LOW-06 FIX: Validiere Konfiguration
            if not _validate_config(config):
                logger.warning("Ungueltige Konfiguration - fuehre Setup erneut durch")
                return run_first_time_setup()

            logger.info(
                f"Vorhandene Konfiguration geladen: {config.get('hardware', {}).get('backend', 'unknown')}"
            )
            return config.get("hardware", get_backend_info())
        except json.JSONDecodeError as e:
            logger.warning(f"Konfiguration korrupt: {e}")
            return run_first_time_setup()
        except Exception as e:
            logger.warning(f"Konnte Konfiguration nicht laden: {e}")
            return get_backend_info()


def reset_first_run():
    """Setzt First-Run Status zurueck (fuer Debugging)."""
    if FIRST_RUN_FILE.exists():
        FIRST_RUN_FILE.unlink()
        logger.info("First-Run Status zurueckgesetzt")
