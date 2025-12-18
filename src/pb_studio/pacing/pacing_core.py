"""
Core Beat-Sync Engine für PB_studio

CRITICAL-02 FIX: Extrahiert aus advanced_pacing_engine.py für bessere Wartbarkeit

Enthält die Kernlogik für:
- Beat-Sync Cut-Generierung
- Trigger-Evaluation und -Filterung
- Min-Cut-Interval Enforcement
- Trigger-Statistiken

Diese Funktionen sind die Basis aller Pacing-Modi.
"""

import logging

from .intensity_controller import IntensityController, TriggerIntensitySettings
from .pacing_config import PacingConfig
from .pacing_models import PacingCut
from .trigger_system import TriggerSystem

logger = logging.getLogger(__name__)


class CorePacingEngine:
    """
    Kern-Engine für Beat-Sync Pacing.

    Verantwortlich für:
    - Cut-Generierung aus Triggern
    - Trigger-Evaluation
    - Min-Interval-Filterung
    """

    def __init__(
        self,
        config: PacingConfig | None = None,
        trigger_settings: TriggerIntensitySettings | None = None,
        trigger_system: TriggerSystem | None = None,
    ):
        """
        Initialisiert Core Pacing Engine.

        Args:
            config: PacingConfig (optional)
            trigger_settings: TriggerIntensitySettings (optional)
            trigger_system: TriggerSystem (optional)
        """
        self.config = config or PacingConfig.create_default()
        self.trigger_settings = trigger_settings or TriggerIntensitySettings()
        self.trigger_system = trigger_system or TriggerSystem()

        logger.debug("CorePacingEngine initialisiert")

    def evaluate_triggers(
        self,
        trigger_type: str,
        times: list[float],
        base_strength: float,
        start_time: float | None = None,
        end_time: float | None = None,
        filter_downbeats: bool = False,
    ) -> list[PacingCut]:
        """
        Evaluiert Trigger und erstellt PacingCut-Objekte.

        Args:
            trigger_type: Typ des Triggers ('beat', 'onset', etc.)
            times: Liste der Trigger-Zeitpunkte
            base_strength: Basis-Stärke des Trigger-Typs (0.0-1.0)
            start_time: Optional start time (filters triggers before this)
            end_time: Optional end time (filters triggers after this)
            filter_downbeats: If True, keep only triggers near downbeat

        Returns:
            Liste von PacingCut-Objekten (nur aktive Trigger)
        """
        cuts: list[PacingCut] = []

        # BPM für Phrase Alignment (wenn verfügbar)
        bpm = 120.0  # Default fallback
        beat_offset = 0.0

        # Versuch, BPM aus letzter Analyse zu holen
        if hasattr(self.trigger_system, "last_analysis") and self.trigger_system.last_analysis:
            bpm = self.trigger_system.last_analysis.bpm
            if self.trigger_system.last_analysis.beat_times:
                beat_offset = self.trigger_system.last_analysis.beat_times[0]

        beat_duration = 60.0 / bpm if bpm > 0 else 0.5
        bar_duration = beat_duration * 4  # Annahme 4/4 Takt

        for time in times:
            # Filter by time window
            if start_time is not None and time < start_time:
                continue
            if end_time is not None and time >= end_time:
                continue

            # Phrase Alignment Logic
            if filter_downbeats:
                rel_time = time - beat_offset
                tolerance = beat_duration * 0.15

                dist_to_bar = rel_time % bar_duration
                is_downbeat = (dist_to_bar < tolerance) or (
                    dist_to_bar > (bar_duration - tolerance)
                )

                if not is_downbeat:
                    continue

            # Trigger evaluieren (Enabled + Threshold + Intensity)
            should_trigger, scaled_strength = IntensityController.evaluate_trigger(
                trigger_type=trigger_type, strength=base_strength, settings=self.trigger_settings
            )

            if should_trigger:
                # Gewichtung anwenden
                config = self.trigger_settings.get_config(trigger_type)
                final_strength = scaled_strength * config.weight

                cuts.append(
                    PacingCut(
                        time=time,
                        trigger_type=trigger_type,
                        strength=final_strength,
                        raw_strength=base_strength,
                    )
                )

        logger.debug(f"Evaluated {len(times)} {trigger_type} triggers → {len(cuts)} active cuts")

        return cuts

    def remove_close_cuts(
        self, cuts: list[PacingCut], min_interval: float | None = None
    ) -> list[PacingCut]:
        """
        Entfernt Cuts die zu nah beieinander liegen.

        Behält jeweils den stärkeren Cut bei Konflikten.

        Args:
            cuts: Liste von PacingCut-Objekten (sortiert nach Zeit)
            min_interval: Minimaler Abstand in Sekunden (verwendet config wenn None)

        Returns:
            Gefilterte Liste von PacingCut-Objekten
        """
        if not cuts:
            return []

        min_interval = min_interval or self.config.min_cut_interval

        # Sort by time (should already be sorted, but ensure)
        sorted_cuts = sorted(cuts, key=lambda c: c.time)

        filtered: list[PacingCut] = []
        last_cut_time = -999.0  # Sehr weit in der Vergangenheit

        for cut in sorted_cuts:
            time_since_last = cut.time - last_cut_time

            if time_since_last >= min_interval:
                # Genug Abstand → behalten
                filtered.append(cut)
                last_cut_time = cut.time
            else:
                # Zu nah → vergleiche Stärken
                if filtered and cut.strength > filtered[-1].strength:
                    # Aktueller Cut ist stärker → ersetze letzten
                    filtered[-1] = cut
                    last_cut_time = cut.time
                # Sonst: ignoriere schwächeren Cut

        removed_count = len(cuts) - len(filtered)
        if removed_count > 0:
            logger.debug(f"Removed {removed_count} cuts (min_interval={min_interval}s)")

        return filtered

    def get_cut_times(self, cuts: list[PacingCut]) -> list[float]:
        """
        Extrahiert Cut-Zeiten aus PacingCut-Liste.

        Args:
            cuts: Liste von PacingCut-Objekten

        Returns:
            Sortierte Liste von Cut-Zeiten in Sekunden
        """
        times = [cut.time for cut in cuts if cut.active]
        return sorted(times)

    def get_trigger_statistics(self, cuts: list[PacingCut]) -> dict[str, int]:
        """
        Berechnet Trigger-Statistiken.

        Args:
            cuts: Liste von PacingCut-Objekten

        Returns:
            Dictionary mit Trigger-Type → Count
        """
        stats: dict[str, int] = {}

        for cut in cuts:
            trigger_type = cut.trigger_type
            stats[trigger_type] = stats.get(trigger_type, 0) + 1

        return stats

    def log_cut_statistics(self, cuts: list[PacingCut]):
        """
        Loggt Cut-Statistiken.

        Args:
            cuts: Liste von PacingCut-Objekten
        """
        if not cuts:
            logger.warning("No cuts generated!")
            return

        stats = self.get_trigger_statistics(cuts)
        total = len(cuts)

        logger.info(f"Generated {total} cuts:")
        for trigger_type, count in sorted(stats.items()):
            percentage = (count / total * 100) if total > 0 else 0
            logger.info(f"  {trigger_type}: {count} ({percentage:.1f}%)")

    def calculate_adaptive_cut_duration(
        self, cut: PacingCut, audio_energy: float, bpm: float
    ) -> float:
        """
        Berechnet adaptive Cut-Dauer basierend auf Trigger-Type und Energy.

        Für ADAPTIVE_FLOW Pacing-Mode.

        Args:
            cut: PacingCut-Objekt
            audio_energy: Audio-Energy (0.0-1.0)
            bpm: BPM der Musik

        Returns:
            Clip-Dauer in Sekunden
        """
        # Base duration depends on trigger type
        if cut.trigger_type == "kick":
            base_duration = self.config.adaptive_kick_base_duration
        else:
            base_duration = self.config.adaptive_other_base_duration

        # Adjust by energy: Higher energy → Shorter clips
        energy_factor = 1.0 - (audio_energy * 0.5)  # 0.5-1.0 range
        duration = base_duration * energy_factor

        # Snap to beat grid (optional, macht Cuts musikalischer)
        beat_duration = 60.0 / bpm if bpm > 0 else 0.5
        # Round to nearest beat
        duration = round(duration / beat_duration) * beat_duration

        # Clamp to min/max
        duration = max(self.config.adaptive_min_duration, duration)

        return duration
