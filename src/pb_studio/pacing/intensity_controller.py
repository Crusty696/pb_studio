"""
Intensity Controller für PB_studio Pacing Engine

Verwaltet Trigger-Intensitäten und wendet exponentielles Scaling an.
Basiert auf Audio-Plugin-Design-Prinzipien für natürliche menschliche Wahrnehmung.

Hauptkomponenten:
- TriggerConfig: Konfiguration pro Trigger-Typ
- TriggerIntensitySettings: Verwaltung aller Trigger-Konfigurationen
- IntensityController: Exponentielles Scaling und Evaluation
"""

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TriggerConfig:
    """
    Konfiguration für einen einzelnen Trigger-Typ.

    Attributes:
        enabled: Trigger aktiviert/deaktiviert
        intensity: Intensität 0-100% (wie stark wirkt der Trigger)
        threshold: Schwellwert 0-100% (ab welcher Stärke wird getriggert)
        weight: Gewichtung bei Multi-Trigger-Kombination
    """

    enabled: bool = True
    intensity: int = 80  # 0-100%
    threshold: int = 30  # 0-100%
    weight: float = 1.0


class TriggerIntensitySettings:
    """
    Verwaltet Intensitäts-Einstellungen für alle Trigger-Typen.

    Standard-Trigger:
    - beat: Regelmäßige Beats (BPM-Grid)
    - onset: Alle Transienten/Hits
    - kick: Nur Kicks (Bass-Drums)
    - snare: Nur Snares
    - hihat: Nur Hi-Hats
    - energy: Energiespitzen
    """

    def __init__(self):
        """Initialisiert mit Standard-Einstellungen (ALLE Trigger aktiviert)."""
        # FIX: All triggers enabled by default to match GUI checkboxes
        # For EDM/Psy-Trance, kick and snare are especially important for cuts!
        self.triggers: dict[str, TriggerConfig] = {
            "beat": TriggerConfig(enabled=True, intensity=80, threshold=30, weight=1.0),
            "onset": TriggerConfig(enabled=True, intensity=60, threshold=40, weight=0.5),
            "kick": TriggerConfig(
                enabled=True,  # FIX: Was False - now enabled for EDM!
                intensity=85,
                threshold=35,
                weight=1.2,  # Kicks are important in EDM!
            ),
            "snare": TriggerConfig(
                enabled=True,  # FIX: Was False - now enabled for EDM!
                intensity=85,
                threshold=35,
                weight=1.0,
            ),
            "hihat": TriggerConfig(
                enabled=True,  # FIX: Was False - now enabled
                intensity=50,
                threshold=45,
                weight=0.3,  # HiHats are lighter triggers
            ),
            "energy": TriggerConfig(enabled=True, intensity=100, threshold=50, weight=1.5),
        }
        logger.info("TriggerIntensitySettings initialisiert: alle Trigger aktiviert")

    def get_config(self, trigger_type: str) -> TriggerConfig:
        """
        Gibt Konfiguration für Trigger-Typ zurück.

        Args:
            trigger_type: Name des Trigger-Typs

        Returns:
            TriggerConfig-Objekt (Standard-Config wenn nicht gefunden)
        """
        if trigger_type not in self.triggers:
            logger.warning(f"Trigger-Typ '{trigger_type}' nicht gefunden, verwende Standard-Config")
            return TriggerConfig()
        return self.triggers[trigger_type]

    def set_config(
        self,
        trigger_type: str,
        enabled: bool = None,
        intensity: int = None,
        threshold: int = None,
        weight: float = None,
    ):
        """
        Setzt Konfiguration für Trigger-Typ.

        Args:
            trigger_type: Name des Trigger-Typs
            enabled: Trigger aktivieren/deaktivieren (optional)
            intensity: Intensität 0-100% (optional)
            threshold: Schwellwert 0-100% (optional)
            weight: Gewichtung (optional)
        """
        if trigger_type not in self.triggers:
            logger.info(f"Erstelle neuen Trigger-Typ: {trigger_type}")
            self.triggers[trigger_type] = TriggerConfig()

        config = self.triggers[trigger_type]

        if enabled is not None:
            config.enabled = enabled
        if intensity is not None:
            config.intensity = max(0, min(100, intensity))
        if threshold is not None:
            config.threshold = max(0, min(100, threshold))
        if weight is not None:
            config.weight = max(0.0, weight)

        logger.debug(
            f"Trigger-Config aktualisiert: {trigger_type} -> "
            f"enabled={config.enabled}, intensity={config.intensity}%, "
            f"threshold={config.threshold}%, weight={config.weight}"
        )


class IntensityController:
    """
    Steuert Trigger-Intensitäten mit exponentiellem Scaling.

    Verwendet Audio-Plugin-Design-Prinzipien:
    - Exponentielles Scaling für natürliche Wahrnehmung
    - Schwellwert-basierte Triggerung
    - Gewichtete Kombination mehrerer Trigger
    """

    @staticmethod
    def apply_intensity(strength: float, intensity_percent: int) -> float:
        """
        Wendet exponentielles Intensity-Scaling an.

        Basiert auf Audio-Plugin-Design-Prinzipien für natürliche
        menschliche Wahrnehmung. Bei 100% Intensität bleibt die
        Trigger-Stärke unverändert, bei niedrigeren Werten wird
        sie exponentiell gedämpft.

        Args:
            strength: Gemessene Trigger-Stärke (0.0-1.0)
            intensity_percent: User-Intensität (0-100%)

        Returns:
            Skalierte Stärke (0.0-1.0)

        Examples:
            >>> IntensityController.apply_intensity(0.8, 100)
            0.8  # Volle Stärke (exponent=1.0)
            >>> IntensityController.apply_intensity(0.8, 50)
            0.64  # Gedämpft (exponent=1.5)
            >>> IntensityController.apply_intensity(0.8, 0)
            0.64  # Stark gedämpft (exponent=2.0)
        """
        # BUG-08 FIX: More informative logging with original and clamped values
        # FIX #15: NaN/Infinity-Validierung vor Clamping (max/min propagieren NaN!)
        if not math.isfinite(strength):
            logger.warning(f"Strength is {strength}, using default 0.5")
            strength = 0.5
        if not math.isfinite(intensity_percent):
            logger.warning(f"Intensity is {intensity_percent}, using default 50")
            intensity_percent = 50

        if not 0.0 <= strength <= 1.0:
            original_strength = strength
            strength = max(0.0, min(1.0, strength))
            logger.warning(
                f"Strength outside valid range [0.0, 1.0]: {original_strength:.3f} "
                f"clamped to {strength:.3f}. This indicates invalid trigger data."
            )

        if not 0 <= intensity_percent <= 100:
            original_intensity = intensity_percent
            intensity_percent = max(0, min(100, intensity_percent))
            logger.warning(
                f"Intensity outside valid range [0, 100]: {original_intensity} "
                f"clamped to {intensity_percent}. Check trigger configuration."
            )

        # Normalisiere auf 0.0-1.0
        intensity = intensity_percent / 100.0

        # Exponentielles Scaling
        # Bei 100% Intensity: exponent = 1.0 (keine Änderung)
        # Bei 50% Intensity: exponent = 1.5 (moderate Dämpfung)
        # Bei 0% Intensity: exponent = 2.0 (starke Dämpfung)
        exponent = 2.0 - intensity
        scaled_strength = strength**exponent
        # Note: Debug logging disabled for performance (2500+ calls/preview)
        return scaled_strength

    @staticmethod
    def check_threshold(strength: float, threshold_percent: int) -> bool:
        """
        Prüft ob Trigger-Stärke Schwellwert überschreitet.

        Args:
            strength: Gemessene Trigger-Stärke (0.0-1.0)
            threshold_percent: Minimum-Schwellwert (0-100%)

        Returns:
            True wenn Stärke >= Schwellwert, False sonst

        Examples:
            >>> IntensityController.check_threshold(0.8, 50)
            True  # 0.8 >= 0.5
            >>> IntensityController.check_threshold(0.3, 50)
            False  # 0.3 < 0.5
        """
        # BUG-08 FIX: More informative logging with original and clamped values
        if not 0.0 <= strength <= 1.0:
            original_strength = strength
            strength = max(0.0, min(1.0, strength))
            logger.warning(
                f"Strength outside valid range [0.0, 1.0]: {original_strength:.3f} "
                f"clamped to {strength:.3f}. This indicates invalid trigger data."
            )

        if not 0 <= threshold_percent <= 100:
            original_threshold = threshold_percent
            threshold_percent = max(0, min(100, threshold_percent))
            logger.warning(
                f"Threshold outside valid range [0, 100]: {original_threshold} "
                f"clamped to {threshold_percent}. Check trigger configuration."
            )

        threshold = threshold_percent / 100.0
        passes = strength >= threshold
        return passes

    @staticmethod
    def evaluate_trigger(
        trigger_type: str, strength: float, settings: TriggerIntensitySettings
    ) -> tuple[bool, float]:
        """
        Evaluiert ob Trigger ausgelöst werden soll.

        Kombiniert drei Checks:
        1. Ist der Trigger aktiviert?
        2. Überschreitet die Stärke den Schwellwert?
        3. Wie stark ist der skalierte Trigger?

        Args:
            trigger_type: Typ des Triggers ('beat', 'onset', etc.)
            strength: Gemessene Trigger-Stärke (0.0-1.0)
            settings: TriggerIntensitySettings-Objekt

        Returns:
            Tuple (should_trigger: bool, scaled_strength: float)
            - should_trigger: True wenn Trigger ausgelöst werden soll
            - scaled_strength: Skalierte Trigger-Stärke (0.0 wenn nicht ausgelöst)

        Examples:
            >>> settings = TriggerIntensitySettings()
            >>> IntensityController.evaluate_trigger('beat', 0.8, settings)
            (True, 0.725...)  # Trigger aktiv, skalierte Stärke
            >>> IntensityController.evaluate_trigger('beat', 0.2, settings)
            (False, 0.0)  # Unter Schwellwert
        """
        config = settings.get_config(trigger_type)

        # Check 1: Ist Trigger aktiviert?
        if not config.enabled:
            return False, 0.0

        # Check 2: Schwellwert-Check
        if not IntensityController.check_threshold(strength, config.threshold):
            return False, 0.0

        # Check 3: Intensity-Scaling anwenden
        scaled_strength = IntensityController.apply_intensity(strength, config.intensity)
        return True, scaled_strength
