"""
Song-Struktur-Analyse für PB_studio Pacing Engine

CRITICAL-02 FIX: Extrahiert aus advanced_pacing_engine.py für bessere Wartbarkeit

Hauptfunktionen:
- Song-Struktur-Erkennung (Intro, Verse, Chorus, Drop, Bridge, Outro)
- Segment-basiertes Motion/Mood-Mapping
- Struktur-Awareness für intelligentere Clip-Auswahl

Verwendung:
    manager = StructureManager(config, structure_analyzer, intensity_mapper)
    manager.enable_structure_awareness()
    properties = manager.determine_target_properties(cut, energy, segment_type)

Reduziert Komplexität der Haupt-Engine weiter.
"""

import logging
from typing import TYPE_CHECKING, Optional

from .pacing_config import PacingConfig
from .structure_analyzer import StructureAnalysisResult, StructureAnalyzer

if TYPE_CHECKING:
    from .pacing_intensity import IntensityMapper
    from .pacing_models import PacingCut

logger = logging.getLogger(__name__)


class StructureManager:
    """
    Verwaltet Song-Struktur-Analyse und Segment-basiertes Mapping.

    Ermöglicht:
    - Struktur-basierte Clip-Auswahl (Verse → ruhig, Chorus → energetisch)
    - Segment-Type-Annotation in Cuts
    - Intelligentere Motion/Mood-Auswahl
    """

    def __init__(
        self,
        config: PacingConfig | None = None,
        structure_analyzer: StructureAnalyzer | None = None,
        intensity_mapper: Optional["IntensityMapper"] = None,
    ):
        """
        Initialisiert Structure Manager.

        Args:
            config: PacingConfig (optional, verwendet Defaults)
            structure_analyzer: StructureAnalyzer (optional, erstellt neues)
            intensity_mapper: IntensityMapper (optional, für Fallback-Mapping)
        """
        self.config = config or PacingConfig.create_default()
        self.structure_analyzer = structure_analyzer or StructureAnalyzer()
        self.intensity_mapper = intensity_mapper

        # State
        self.use_structure_awareness = False
        self.structure_result: StructureAnalysisResult | None = None

        logger.debug("StructureManager initialisiert")

    def enable_structure_awareness(self, enabled: bool = True):
        """
        Aktiviert/deaktiviert Song-Struktur-Awareness.

        Args:
            enabled: Ob Struktur-Awareness aktiviert werden soll
        """
        self.use_structure_awareness = enabled
        logger.info(f"Song-Struktur-Awareness: {'aktiviert' if enabled else 'deaktiviert'}")

    def analyze_song_structure(self, audio_path: str) -> StructureAnalysisResult:
        """
        Analysiert Song-Struktur.

        Args:
            audio_path: Pfad zur Audio-Datei

        Returns:
            StructureAnalysisResult mit erkannten Segmenten

        Raises:
            Exception: Wenn Struktur-Analyse fehlschlägt
        """
        self.structure_result = self.structure_analyzer.analyze_structure(audio_path)
        logger.info(
            f"🎼 Song structure analyzed: {len(self.structure_result.segments)} segments "
            f"({[s.segment_type for s in self.structure_result.segments]})"
        )
        return self.structure_result

    def get_segment_at_time(self, time: float) -> str | None:
        """
        Gibt Segment-Type zur angegebenen Zeit zurück.

        Args:
            time: Zeit in Sekunden

        Returns:
            Segment-Type (intro, verse, chorus, drop, bridge, outro) oder None
        """
        if not self.structure_result or not self.use_structure_awareness:
            return None

        for segment in self.structure_result.segments:
            if segment.start_time <= time < segment.end_time:
                return segment.segment_type.lower()

        return None

    def determine_target_properties(
        self, cut: "PacingCut", cut_idx: int, audio_energy: float, segment_type: str | None = None
    ) -> tuple[float, float, str, str]:
        """
        Bestimmt Target-Motion/Energy/Mood für einen Cut basierend auf Audio-Analyse.

        M-04 FIX: Extracted from advanced_pacing_engine for better readability.

        Verwendet:
        1. Segment-basiertes Mapping (wenn verfügbar)
        2. Energy-basiertes Mapping (Fallback)

        Args:
            cut: PacingCut-Objekt
            cut_idx: Index des Cuts (für Variation)
            audio_energy: Audio-Energy an diesem Cut-Punkt (0.0-1.0)
            segment_type: Song-Segment-Type (e.g., 'drop', 'verse', None)

        Returns:
            Tuple of (varied_energy, varied_motion, target_motion_type, target_mood)
        """
        # Small variation to prevent exact same clips
        variation = (cut_idx % 10) * 0.05  # ±5% variation
        varied_energy = max(0.0, min(1.0, audio_energy + variation - 0.025))
        varied_motion = max(0.0, min(1.0, audio_energy - variation + 0.025))

        # Select motion type based on SEGMENT and ENERGY
        if segment_type and segment_type in self.config.segment_motion_mapping:
            segment_motions = self.config.segment_motion_mapping[segment_type]

            # Energy-basierte Auswahl innerhalb Segment-Motions
            if audio_energy > 0.7 and "FAST" in segment_motions:
                target_motion_type = "FAST"
            elif audio_energy > 0.7 and "EXTREME" in segment_motions:
                target_motion_type = "EXTREME"
            elif audio_energy > 0.5 and "MEDIUM" in segment_motions:
                target_motion_type = "MEDIUM"
            elif audio_energy < 0.3 and "STATIC" in segment_motions:
                target_motion_type = "STATIC"
            else:
                # Rotation durch Segment-Motions
                target_motion_type = segment_motions[cut_idx % len(segment_motions)]
        else:
            # Fallback: Energy-basiert (wenn IntensityMapper verfügbar)
            if self.intensity_mapper:
                target_motion_type = self.intensity_mapper.energy_to_motion(audio_energy)
            else:
                # Default Fallback
                target_motion_type = "MEDIUM"

        # Select mood based on SEGMENT TYPE
        if segment_type and segment_type in self.config.segment_mood_mapping:
            segment_moods = self.config.segment_mood_mapping[segment_type]
            target_mood = segment_moods[cut_idx % len(segment_moods)]
        else:
            # Fallback: Energy-basiert
            if self.intensity_mapper:
                target_mood = self.intensity_mapper.energy_to_mood(audio_energy, cut_idx)
            else:
                # Default Fallback
                target_mood = "CHEERFUL"

        return varied_energy, varied_motion, target_motion_type, target_mood

    def should_analyze_structure(self, audio_duration: float) -> bool:
        """
        Entscheidet ob Struktur-Analyse durchgeführt werden soll.

        Struktur-Analyse ist O(n²) und kann bei langen Audio-Dateien
        zu Performance-Problemen führen (Self-Similarity Matrix).

        Args:
            audio_duration: Audio-Dauer in Sekunden

        Returns:
            True wenn Struktur-Analyse durchgeführt werden soll
        """
        # Check Config Flag
        if not self.config.enable_structure_analysis:
            logger.debug("Structure analysis disabled in config")
            return False

        # Check Audio-Dauer (Max 2 Minuten für Struktur-Analyse)
        MAX_DURATION_FOR_STRUCTURE = 120  # 2 minutes
        if audio_duration > MAX_DURATION_FOR_STRUCTURE:
            logger.warning(
                f"⚠️ Skipping structure analysis: audio too long "
                f"({audio_duration / 60:.1f} min > {MAX_DURATION_FOR_STRUCTURE / 60:.1f} min). "
                f"Using energy-based mood mapping instead."
            )
            return False

        return True

    def get_structure_summary(self) -> str:
        """
        Gibt Zusammenfassung der erkannten Struktur zurück.

        Returns:
            Human-readable Struktur-Zusammenfassung
        """
        if not self.structure_result:
            return "No structure analysis available"

        segments = self.structure_result.segments
        summary_parts = []

        for seg in segments:
            duration = seg.end_time - seg.start_time
            summary_parts.append(
                f"{seg.segment_type.upper()} ({seg.start_time:.1f}s-{seg.end_time:.1f}s, {duration:.1f}s)"
            )

        return " → ".join(summary_parts)
