"""
Intensity & Energy Mapping für PB_studio Pacing Engine

CRITICAL-02 FIX: Extrahiert aus advanced_pacing_engine.py für bessere Wartbarkeit

Hauptfunktionen:
- Energy → Motion Mapping (STATIC, SLOW, MEDIUM, FAST, EXTREME)
- Energy → Mood Mapping (CALM, ENERGETIC, EUPHORIC, etc.)
- Trigger-Strength-Berechnung
- Variation/Diversity-Algorithmen

Reduziert Komplexität der Haupt-Engine von 1797 auf ~1200 Zeilen.
"""

import logging

from .pacing_config import MotionType, PacingConfig

logger = logging.getLogger(__name__)


class IntensityMapper:
    """
    Mapped Audio-Energy auf Video-Motion und Mood.

    Verwendet konfigurierbare Thresholds aus PacingConfig für:
    - Energy → Motion (5 Stufen: STATIC bis EXTREME)
    - Energy → Mood (15+ Moods mit Rotation für Variation)
    """

    def __init__(self, config: PacingConfig | None = None):
        """
        Initialisiert Intensity Mapper.

        Args:
            config: PacingConfig (optional, verwendet Defaults)
        """
        self.config = config or PacingConfig.create_default()
        logger.debug("IntensityMapper initialisiert")

    def energy_to_motion(self, energy: float) -> str:
        """
        Konvertiert Energie zu Motion-Type.

        Mapping (uses config thresholds):
        - Below energy_threshold_static (0.2): STATIC
        - Below energy_threshold_slow (0.4): SLOW
        - Below energy_threshold_medium (0.6): MEDIUM
        - Below energy_threshold_fast (0.8): FAST
        - Above: EXTREME

        Args:
            energy: Energy-Level (0.0-1.0)

        Returns:
            Motion-Type String (STATIC, SLOW, MEDIUM, FAST, EXTREME)
        """
        if energy < self.config.energy_threshold_static:
            return MotionType.STATIC.value
        elif energy < self.config.energy_threshold_slow:
            return MotionType.SLOW.value
        elif energy < self.config.energy_threshold_medium:
            return MotionType.MEDIUM.value
        elif energy < self.config.energy_threshold_fast:
            return MotionType.FAST.value
        else:
            return MotionType.EXTREME.value

    def energy_to_mood(self, energy: float, cut_idx: int = 0) -> str:
        """
        Konvertiert Energie zu Mood mit Abwechslung.

        IMPROVED: More mood variety for long DJ sets (no structure analysis).
        Uses energy level + cut index for variety within energy bands.

        Mapping:
        - Very low (<0.2): CALM, DREAMY, PEACEFUL (rotation)
        - Low (0.2-0.4): PEACEFUL, COOL, MELANCHOLIC (rotation)
        - Medium (0.4-0.6): CHEERFUL, WARM, BRIGHT (rotation)
        - High (0.6-0.8): ENERGETIC, EUPHORIC, CHEERFUL (rotation)
        - Very high (>0.8): ENERGETIC, EUPHORIC, AGGRESSIVE (rotation)

        Args:
            energy: Energy-Level (0.0-1.0)
            cut_idx: Cut-Index für Rotation innerhalb Energy-Band

        Returns:
            Mood-Type String
        """
        if energy < 0.2:
            moods = ["CALM", "DREAMY", "PEACEFUL"]
        elif energy < 0.4:
            moods = ["PEACEFUL", "COOL", "MELANCHOLIC"]
        elif energy < 0.6:
            moods = ["CHEERFUL", "WARM", "BRIGHT"]
        elif energy < 0.8:
            moods = ["ENERGETIC", "EUPHORIC", "CHEERFUL"]
        else:
            moods = ["ENERGETIC", "EUPHORIC", "AGGRESSIVE"]

        # Rotate through moods within energy band for variety
        return moods[cut_idx % len(moods)]

    def get_motion_from_energy(self, energy_level: float) -> str:
        """
        Wrapper für energy_to_motion (Backward Compatibility).

        Args:
            energy_level: Energy-Level (0.0-1.0)

        Returns:
            Motion-Type String
        """
        return self.energy_to_motion(energy_level)

    def get_emotion_from_audio(
        self, energy_level: float, cut_index: int = 0, segment_type: str | None = None
    ) -> str:
        """
        Extrahiert Emotion aus Audio-Features.

        Verwendet entweder:
        1. Segment-basiertes Mapping (wenn segment_type verfügbar)
        2. Energy-basiertes Mapping (Fallback)

        Args:
            energy_level: Energy-Level (0.0-1.0)
            cut_index: Cut-Index für Variation
            segment_type: Optional Segment-Type (intro, verse, chorus, drop, etc.)

        Returns:
            Mood/Emotion String
        """
        if segment_type and segment_type in self.config.segment_mood_mapping:
            # Segment-basiertes Mapping
            moods = self.config.segment_mood_mapping[segment_type]
            return moods[cut_index % len(moods)]
        else:
            # Energy-basiertes Mapping (Fallback)
            return self.energy_to_mood(energy_level, cut_index)

    def calculate_motion_variation(self, cut_idx: int, base_energy: float) -> float:
        """
        Berechnet Motion-Variation für Clip-Diversität.

        Verwendet Sinus-Wave-Pattern für natürliche Variation.
        Wichtig für lange Videos (verhindert monotone Clip-Auswahl).

        Formula:
        variation = sin(cut_idx / cycle_count * 2π) * step_size - offset

        Args:
            cut_idx: Cut-Index
            base_energy: Basis-Energy-Level

        Returns:
            Variierter Energy-Level
        """
        import math

        # Sinus-basierte Variation (natürliches auf/ab)
        cycle = self.config.variation_cycle_count
        variation = math.sin(cut_idx / cycle * 2 * math.pi) * self.config.variation_step_size

        # Apply variation mit Offset
        varied_energy = base_energy + variation - self.config.variation_motion_offset

        # Clamp auf [0.0, 1.0]
        return max(0.0, min(1.0, varied_energy))

    def calculate_energy_variation(self, cut_idx: int, base_energy: float) -> float:
        """
        Berechnet Energy-Variation für Clip-Diversität.

        Ähnlich wie motion_variation aber mit anderem Offset.

        Args:
            cut_idx: Cut-Index
            base_energy: Basis-Energy-Level

        Returns:
            Variierter Energy-Level
        """
        import math

        cycle = self.config.variation_cycle_count
        variation = math.sin(cut_idx / cycle * 2 * math.pi) * self.config.variation_step_size

        # Apply variation mit Energy-Offset
        varied_energy = base_energy + variation + self.config.variation_energy_offset

        # Clamp auf [0.0, 1.0]
        return max(0.0, min(1.0, varied_energy))

    def get_trigger_base_strength(self, trigger_type: str) -> float:
        """
        Gibt Basis-Stärke für Trigger-Typ zurück.

        Args:
            trigger_type: Trigger-Typ (beat, onset, kick, snare, etc.)

        Returns:
            Basis-Stärke (0.0-1.0)
        """
        trigger_strengths = {
            "beat": self.config.trigger_strength_beat,
            "onset": self.config.trigger_strength_onset,
            "kick": self.config.trigger_strength_percussion,
            "snare": self.config.trigger_strength_percussion,
            "hihat": self.config.trigger_strength_hihat,
            "energy": self.config.trigger_strength_energy_peak,
            "bass": self.config.trigger_strength_bass,
        }

        return trigger_strengths.get(trigger_type, 0.6)  # Default: 0.6
