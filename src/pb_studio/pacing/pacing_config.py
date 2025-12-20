"""
Pacing Engine Konfiguration

Zentrale Konfiguration für Advanced Pacing Engine.
Ersetzt hardcodierte Feature Flags und Magic Numbers durch
eine konfigurierbare Klasse.

CRITICAL-02 FIX: Refactoring für bessere Wartbarkeit
"""

from dataclasses import dataclass, field
from enum import Enum


class PacingMode(Enum):
    """Pacing modes for clip length control."""

    BEAT_SYNC = "BEAT_SYNC"  # Fixed-length cuts (0.6s), maximal beat synchronization
    ADAPTIVE_FLOW = "ADAPTIVE_FLOW"  # Variable-length clips (4-10s), emotion-based flow


class MotionType(Enum):
    """Motion intensity levels for clip matching."""

    STATIC = "STATIC"
    SLOW = "SLOW"
    MEDIUM = "MEDIUM"
    FAST = "FAST"
    EXTREME = "EXTREME"


class MoodType(Enum):
    """Mood types for clip emotional matching."""

    CALM = "CALM"
    PEACEFUL = "PEACEFUL"
    CHEERFUL = "CHEERFUL"
    ENERGETIC = "ENERGETIC"
    MELANCHOLIC = "MELANCHOLIC"
    EUPHORIC = "EUPHORIC"
    AGGRESSIVE = "AGGRESSIVE"
    MYSTERIOUS = "MYSTERIOUS"
    DREAMY = "DREAMY"
    COOL = "COOL"
    WARM = "WARM"
    NOSTALGIC = "NOSTALGIC"
    ROMANTIC = "ROMANTIC"
    TENSE = "TENSE"
    PLAYFUL = "PLAYFUL"


@dataclass
class PacingConfig:
    """
    Zentrale Konfiguration für Advanced Pacing Engine.

    Alle Feature Flags und Algorithmus-Parameter an einem Ort.
    Macht Testing und Tuning einfacher.
    """

    # =============================================================================
    # Feature Flags (enable/disable optional features)
    # =============================================================================
    enable_structure_analysis: bool = True  # Aktiv nach speichereffizienter Struktur-Analyse
    enable_clip_loop: bool = False  # War zu aggressiv (jedes Video geloopt)
    enable_vector_search: bool = True  # FAISS/Qdrant für Clip-Matching
    enable_semantic_search: bool = True  # Semantic Clip Matching
    enable_music_video_mapping: bool = True  # Audio-Visual Synchronization (Phase 2)
    enable_parallel_processing: bool = False  # Parallele Clip-Auswahl mit ProcessPoolExecutor

    # =============================================================================
    # Parallel Processing Settings
    # =============================================================================
    parallel_clip_threshold: int = 100  # Aktiviere automatisch bei >100 Clips
    parallel_cut_threshold: int = 500  # Aktiviere automatisch bei >500 Cuts
    parallel_max_workers: int | None = None  # None = auto-detect CPU cores

    # =============================================================================
    # Clip Diversity Settings
    # =============================================================================
    diversity_reset_threshold: float = 0.75  # Reset exclusion nach 75% Clip-Nutzung
    diversity_reset_threshold_alt: float = 0.80  # Alternative: 80%

    # =============================================================================
    # Clip-Auswahl Balance (Kontinuität vs. Diversität)
    # =============================================================================
    continuity_weight: float = 0.6
    """Balance zwischen Kontinuität und Diversität bei Clip-Auswahl.

    Steuert die Gewichtung im FAISS-Scoring:
    - 0.0 = Maximale Diversität (abwechslungsreich, aber sprunghaft)
    - 0.5 = Ausgewogen (gleiche Gewichtung Diversität und Ähnlichkeit)
    - 1.0 = Maximale Kontinuität (starker roter Faden, weniger Abwechslung)

    Default: 0.6 (leichter Fokus auf visuellen roten Faden)

    Auswirkung:
    - Höhere Werte (z.B. 0.8): Clips bleiben thematisch/visuell ähnlich
    - Niedrigere Werte (z.B. 0.3): Mehr Abwechslung, überraschende Übergänge
    """

    # =============================================================================
    # Motion Variation Settings (für Clip-Diversität in langen Videos)
    # =============================================================================
    variation_cycle_count: int = 20  # Cuts bis Variation-Pattern wiederholt
    variation_step_size: float = 0.12  # Step size pro Cut
    variation_motion_offset: float = 0.3  # Motion variation offset
    variation_energy_offset: float = 0.2  # Energy variation offset

    # =============================================================================
    # Energy-to-Motion Mapping Thresholds
    # =============================================================================
    energy_threshold_static: float = 0.2  # < 0.2: STATIC motion
    energy_threshold_slow: float = 0.4  # < 0.4: SLOW motion
    energy_threshold_medium: float = 0.6  # < 0.6: MEDIUM motion
    energy_threshold_fast: float = 0.8  # < 0.8: FAST, >= 0.8: EXTREME

    # =============================================================================
    # Energy-to-Mood Mapping Thresholds
    # =============================================================================
    mood_threshold_calm: float = 0.3  # < 0.3: CALM
    mood_threshold_peaceful: float = 0.45  # < 0.45: PEACEFUL
    mood_threshold_cheerful: float = 0.6  # < 0.6: CHEERFUL, >= 0.6: ENERGETIC

    # =============================================================================
    # Trigger Type Base Strengths
    # =============================================================================
    trigger_strength_beat: float = 0.8
    trigger_strength_onset: float = 0.6
    trigger_strength_percussion: float = 0.7
    trigger_strength_bass: float = 0.65
    trigger_strength_hihat: float = 0.5
    trigger_strength_energy_peak: float = 1.0

    # =============================================================================
    # Timing & Duration
    # =============================================================================
    min_cut_interval: float = 2.0  # Min. Abstand zwischen Cuts (Sekunden)

    # Adaptive Flow Mode
    adaptive_motion_threshold: float = 0.3  # Max Diff für perfektes Motion-Match
    adaptive_kick_base_duration: float = 6.0  # Base Duration Kick-Trigger
    adaptive_other_base_duration: float = 8.0  # Base Duration andere Trigger
    adaptive_min_duration: float = 2.0  # Min. Clip-Dauer
    adaptive_emotion_window: float = 4.0  # Emotion-Fenster-Größe

    # =============================================================================
    # Segment-Based Mood/Motion Mapping
    # =============================================================================
    segment_mood_mapping: dict[str, list[str]] = field(
        default_factory=lambda: {
            "intro": ["CALM", "PEACEFUL", "DREAMY", "MYSTERIOUS"],
            "verse": ["CALM", "PEACEFUL", "MELANCHOLIC", "COOL"],
            "chorus": ["ENERGETIC", "CHEERFUL", "BRIGHT", "WARM", "EUPHORIC"],
            "drop": ["ENERGETIC", "EUPHORIC", "AGGRESSIVE", "TENSE"],
            "bridge": ["DREAMY", "MYSTERIOUS", "MELANCHOLIC", "COOL"],
            "outro": ["CALM", "PEACEFUL", "DREAMY", "WARM"],
        }
    )

    segment_motion_mapping: dict[str, list[str]] = field(
        default_factory=lambda: {
            "intro": ["STATIC", "SLOW"],
            "verse": ["SLOW", "MEDIUM"],
            "chorus": ["FAST", "EXTREME"],
            "drop": ["EXTREME"],
            "bridge": ["MEDIUM", "SLOW"],
            "outro": ["SLOW", "STATIC"],
        }
    )

    def __post_init__(self):
        """Validierung und Normalisierung nach Initialisierung."""
        # Clamp continuity_weight auf [0.0, 1.0]
        self.continuity_weight = max(0.0, min(1.0, self.continuity_weight))

    @classmethod
    def create_default(cls) -> "PacingConfig":
        """Factory method für Standard-Konfiguration."""
        return cls()

    @classmethod
    def create_fast_cuts(cls) -> "PacingConfig":
        """Factory method für schnelle, hektische Cuts (Actionfilm-Stil)."""
        config = cls()
        config.min_cut_interval = 0.5  # Sehr schnell!
        config.variation_cycle_count = 10  # Mehr Variation
        return config

    @classmethod
    def create_smooth_flow(cls) -> "PacingConfig":
        """Factory method für ruhige, fließende Cuts (Doku-Stil)."""
        config = cls()
        config.min_cut_interval = 4.0  # Langsam
        config.variation_step_size = 0.05  # Weniger Variation
        config.adaptive_min_duration = 4.0  # Längere Clips
        return config


# Pre-computed lists for fast access
MOTION_TYPES_LIST = [m.value for m in MotionType]
MOODS_LIST = [m.value for m in MoodType]
