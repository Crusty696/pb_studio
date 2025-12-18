"""
Advanced Pacing Engine für PB_studio

Kombiniert Trigger-System und Intensity-Controller zu einer intelligenten
Pacing-Engine, die Schnittlisten basierend auf Audio-Triggern generiert.

Hauptkomponenten:
- TriggerSystem: Extrahiert Trigger aus Audio
- IntensityController: Skaliert Trigger-Stärken
- AdvancedPacingEngine: Generiert Schnittlisten

Workflow:
1. Audio analysieren → Trigger extrahieren
2. Trigger nach Intensität evaluieren
3. Trigger nach Stärke gewichten
4. Schnittliste generieren
"""

import logging
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

# Type alias für Progress-Callback: (current, total, message) -> None
PacingProgressCallback = Callable[[int, int, str], None]

# Zentrale Konfiguration + Modularisierung
from .clip_selection import (
    ClipSelectionStrategy,
    DiversityManager,
    FAISSStrategy,
    RoundRobinStrategy,
    SmartStrategy,
)
from .energy_curve import EnergyAnalyzer, EnergyCurveData
from .intensity_controller import TriggerIntensitySettings
from .motion_analyzer import MotionAnalyzer
from .pacing_config import PacingConfig, PacingMode
from .pacing_core import CorePacingEngine
from .pacing_intensity import IntensityMapper
from .pacing_models import PacingCut
from .pacing_structure import StructureManager
from .structure_analyzer import StructureAnalysisResult, StructureAnalyzer
from .trigger_system import TriggerSystem

# Parallel Processing Support (Phase 5)
try:
    from .parallel_pacing_engine import ClipSelectionResult, ClipSelectionTask, ParallelPacingEngine

    PARALLEL_PROCESSING_AVAILABLE = True
except ImportError:
    PARALLEL_PROCESSING_AVAILABLE = False
    ParallelPacingEngine = None

# Phase 2: Music-Video Mapping (Audio-Visual Synchronization)
# Use lazy imports to avoid circular dependency
MUSIC_VIDEO_MAPPING_AVAILABLE = False
_audio_visual_mapper_module = None
_motion_music_sync_module = None


def _get_audio_visual_mapper():
    """Lazy import of AudioVisualMapper to avoid circular imports."""
    global _audio_visual_mapper_module, MUSIC_VIDEO_MAPPING_AVAILABLE
    if _audio_visual_mapper_module is None:
        try:
            from . import audio_visual_mapper as _avm

            _audio_visual_mapper_module = _avm
            MUSIC_VIDEO_MAPPING_AVAILABLE = True
        except ImportError:
            pass
    return _audio_visual_mapper_module


def _get_motion_music_sync():
    """Lazy import of MotionMusicSynchronizer to avoid circular imports."""
    global _motion_music_sync_module
    if _motion_music_sync_module is None:
        try:
            from . import motion_music_sync as _mms

            _motion_music_sync_module = _mms
        except ImportError:
            pass
    return _motion_music_sync_module


# For type checking only (no runtime import)
if TYPE_CHECKING:
    from .audio_visual_mapper import AudioDerivedPalette
    from .motion_music_sync import ClipBeatAlignment

# Phase 3: Multi-Platform Vector Search (FAISS / Qdrant)
try:
    from .clip_matcher_factory import create_clip_matcher, get_available_backends

    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False
    create_clip_matcher = None

# Backward compatibility
FAISS_AVAILABLE = VECTOR_SEARCH_AVAILABLE

try:
    from .semantic_matcher import SemanticClipMatcher

    SEMANTIC_SEARCH_AVAILABLE = True
except Exception as exc:
    SEMANTIC_SEARCH_AVAILABLE = False
    SemanticClipMatcher = None
    logging.getLogger(__name__).warning(
        "Semantic search disabled (dependency import failed): %s", exc
    )

logger = logging.getLogger(__name__)

# CRITICAL-02 FIX: Enums und Lists jetzt in pacing_config.py
# (importiert oben, keine Duplikation mehr)

# Phase 3: Smart Matcher Integration
try:
    from .clip_matcher import SmartMatcher

    SMART_MATCHING_AVAILABLE = True
except ImportError:
    SMART_MATCHING_AVAILABLE = False
    SmartMatcher = None


# =============================================================================
# Pacing Algorithm Constants (extracted magic numbers)
# =============================================================================

# Clip Diversity Settings
# Reset exclusion list after using X% of available clips
CLIP_DIVERSITY_RESET_THRESHOLD = 0.75  # 75% for generate_cut_list_with_clips
CLIP_DIVERSITY_RESET_THRESHOLD_ALT = 0.80  # 80% for _select_clip_for_cut

# Motion Variation Settings (for clip diversity in long videos)
VARIATION_CYCLE_COUNT = 20  # Number of cuts before variation pattern repeats
VARIATION_STEP_SIZE = 0.12  # Step size for motion variation per cut
VARIATION_MOTION_OFFSET = 0.3  # Offset subtracted from motion variation
VARIATION_ENERGY_OFFSET = 0.2  # Offset added to energy variation

# Energy-to-Motion Mapping Thresholds
ENERGY_THRESHOLD_STATIC = 0.2  # Below: STATIC motion
ENERGY_THRESHOLD_SLOW = 0.4  # Below: SLOW motion
ENERGY_THRESHOLD_MEDIUM = 0.6  # Below: MEDIUM motion
ENERGY_THRESHOLD_FAST = 0.8  # Below: FAST motion, Above: EXTREME

# Energy-to-Mood Mapping Thresholds
MOOD_THRESHOLD_CALM = 0.3  # Below: CALM mood
MOOD_THRESHOLD_PEACEFUL = 0.45  # Below: PEACEFUL mood
MOOD_THRESHOLD_CHEERFUL = 0.6  # Below: CHEERFUL mood, Above: ENERGETIC

# Trigger Type Base Strengths
TRIGGER_STRENGTH_BEAT = 0.8  # Beats are generally strong
TRIGGER_STRENGTH_ONSET = 0.6  # Onsets are medium strength
TRIGGER_STRENGTH_PERCUSSION = 0.7  # Percussion hits
TRIGGER_STRENGTH_BASS = 0.65  # Bass hits
TRIGGER_STRENGTH_HIHAT = 0.5  # HiHats are lighter
TRIGGER_STRENGTH_ENERGY_PEAK = 1.0  # Energy peaks are very strong

# Minimum Cut Interval Default
# FIX: Increased from 0.1 to 2.0 seconds to reduce hectic cuts
# 0.1s was way too fast - resulted in nervous/hectic video
DEFAULT_MIN_CUT_INTERVAL = 2.0  # seconds (was 0.1 - too hectic!)

# =============================================================================
# LONG-AUDIO-FIX: Settings for audio files > 30 minutes
# Problem: 62 min audio = 7800+ beats = UI freeze during cut-list generation
# Solution: Reduce triggers + batch processing + UI updates
# =============================================================================
LONG_AUDIO_THRESHOLD_MINUTES = 30.0  # Files longer than this get optimized
LONG_AUDIO_TRIGGER_REDUCE_FACTOR = 4  # Use every Nth trigger (reduces 7800 to ~1950)
LONG_AUDIO_BATCH_SIZE = 100  # Process cuts in batches of 100
LONG_AUDIO_UI_UPDATE_INTERVAL = 10  # Update UI every N cuts (within batch)

# Adaptive Flow Mode Constants
ADAPTIVE_MOTION_THRESHOLD = 0.3  # Max difference for perfect motion match
ADAPTIVE_KICK_BASE_DURATION = 6.0  # Base duration for kick triggers (s)
ADAPTIVE_OTHER_BASE_DURATION = 8.0  # Base duration for other triggers (s)
ADAPTIVE_MIN_DURATION = 2.0  # Minimum clip duration (s)
ADAPTIVE_EMOTION_WINDOW = 4.0  # Emotion window size (s)

# =============================================================================
# Feature Flags: CRITICAL-02 FIX - jetzt in PacingConfig
# =============================================================================
# Feature Flags wurden nach pacing_config.py verschoben für bessere Wartbarkeit

# =============================================================================
# Structure-Based Mood Mapping (for intelligent clip selection)
# =============================================================================
# Maps song segment types to appropriate moods for better music-video sync
SEGMENT_MOOD_MAPPING = {
    "intro": ["CALM", "PEACEFUL", "DREAMY", "MYSTERIOUS"],
    "verse": ["CALM", "PEACEFUL", "MELANCHOLIC", "COOL"],
    "chorus": ["ENERGETIC", "CHEERFUL", "BRIGHT", "WARM", "EUPHORIC"],
    "drop": ["ENERGETIC", "EUPHORIC", "AGGRESSIVE", "TENSE"],
    "bridge": ["DREAMY", "MYSTERIOUS", "MELANCHOLIC", "COOL"],
    "outro": ["CALM", "PEACEFUL", "DREAMY", "WARM"],
}

# Motion types for different song segments
SEGMENT_MOTION_MAPPING = {
    "intro": ["STATIC", "SLOW"],
    "verse": ["SLOW", "MEDIUM"],
    "chorus": ["MEDIUM", "FAST"],
    "drop": ["FAST", "EXTREME"],
    "bridge": ["SLOW", "MEDIUM"],
    "outro": ["SLOW", "STATIC"],
}

# Semantic text queries for different song segments (Vibe-Matching)
SEGMENT_TEXT_MAPPING = {
    "intro": "abstract, dark, cinematic, atmospheric, slow motion, texture",
    "verse": "portrait, performance, stylish, medium pace, cool, neon",
    "chorus": "party, dancing, crowd, energetic, lights, concert, stage",
    "drop": "explosion, strobe, lasers, fast motion, intense, energy",
    "bridge": "slow motion, emotional, close up, nature, dream",
    "outro": "fade out, abstract, logo, crowd cheering, calm",
}


class AdvancedPacingEngine:
    """
    Advanced Pacing Engine mit Trigger-basierter Schnittlistengenerierung.

    Kombiniert:
    - TriggerSystem für Audio-Analyse
    - IntensityController für Trigger-Scaling
    - Intelligente Schnittlisten-Generierung

    Unterstützt:
    - Multi-Trigger-Kombination mit Gewichtung
    - Intensity-basiertes Scaling
    - Threshold-basierte Filterung
    - Adaptive Schnitt-Dichte
    """

    def __init__(
        self,
        trigger_settings: TriggerIntensitySettings | None = None,
        trigger_system: TriggerSystem | None = None,
        motion_analyzer: MotionAnalyzer | None = None,
        structure_analyzer: StructureAnalyzer | None = None,
        pacing_mode: PacingMode = PacingMode.BEAT_SYNC,
        config: PacingConfig | None = None,
    ):
        """
        Initialisiert die Advanced Pacing Engine.

        Args:
            trigger_settings: TriggerIntensitySettings (optional, verwendet Defaults)
            trigger_system: TriggerSystem (optional, erstellt neues mit Defaults)
            motion_analyzer: MotionAnalyzer (optional, erstellt neues mit Defaults)
            structure_analyzer: StructureAnalyzer (optional, erstellt neues mit Defaults)
            pacing_mode: Pacing-Modus (BEAT_SYNC oder ADAPTIVE_FLOW)
            config: PacingConfig (optional, verwendet Defaults)
        """
        # CRITICAL-02 FIX: Zentrale Konfiguration
        self.config = config or PacingConfig.create_default()

        self.trigger_settings = trigger_settings or TriggerIntensitySettings()
        self.trigger_system = trigger_system or TriggerSystem()
        self.pacing_mode = pacing_mode

        # CRITICAL-02 FIX: Modularisierte Sub-Engines
        self.core_engine = CorePacingEngine(self.config, self.trigger_settings, self.trigger_system)
        self.intensity_mapper = IntensityMapper(self.config)

        # Motion-Energy-Matching Support (Phase 2)
        self.motion_analyzer = motion_analyzer or MotionAnalyzer()
        self.use_motion_matching = False  # Flag: Motion-Matching aktiviert?

        # Smart Matcher (Unified Logic)
        self.smart_matcher = SmartMatcher() if SMART_MATCHING_AVAILABLE else None
        self.use_smart_matching = True  # Enable by default if available

        # Song-Struktur-Analyse Support (Phase 3)
        self.structure_analyzer = structure_analyzer or StructureAnalyzer()
        self.structure_manager = StructureManager(
            self.config, self.structure_analyzer, self.intensity_mapper
        )
        self.use_structure_awareness = False  # Flag: Struktur-Awareness aktiviert?
        self.structure_result: StructureAnalysisResult | None = None

        # Energy Curve Analysis (Phase 4 - Improved Music Matching)
        self.energy_analyzer = EnergyAnalyzer()
        self.energy_curve: EnergyCurveData | None = None
        self.use_energy_matching = True  # Flag: Direct energy matching aktiviert?

        # Vector Search Motion Matching Support (Phase 3)
        # Supports: FAISS (NVIDIA GPU), Qdrant (AMD GPU / optimized CPU)
        self.clip_matcher = None  # FAISSClipMatcher or QdrantClipMatcher
        self.use_faiss = False  # Flag: Vector Search aktiviert? (kept for backward compat)
        self.use_ai_embeddings = False  # Flag: KI-Embeddings (X-CLIP 512D) aktiviert?

        # Backward compatibility alias (will be set when matcher is created)
        self.faiss_matcher = None

        # Semantic Matching (Phase 4 Extension)
        self.semantic_matcher: SemanticClipMatcher | None = None
        self.use_semantic_matching = False

        # Phase 2: Music-Video Mapping Support
        self.audio_visual_mapper = None
        self.motion_sync = None
        self.use_music_video_mapping = False  # Flag: Enable audio-visual sync?
        self._audio_derived_palette = None  # Type: AudioDerivedPalette | None
        self._beat_times: list[float] = []  # Audio beat times for sync scoring

        # Initialize Phase 2 modules using lazy imports (avoids circular dependency)
        avm_module = _get_audio_visual_mapper()
        mms_module = _get_motion_music_sync()
        if avm_module and mms_module:
            self.audio_visual_mapper = avm_module.AudioVisualMapper()
            self.motion_sync = mms_module.MotionMusicSynchronizer()
            logger.info("Phase 2: Music-Video Mapping modules initialized")

        # QUICK WIN #2: Cache clips_by_id dictionary (5-8% speedup)
        self._clips_by_id_cache: dict[int, dict] | None = None
        self._clips_cache_hash: int | None = None
        # FIX #12: Thread-Safety für Cache-Zugriffe
        self._cache_lock = threading.Lock()

        # Phase 5: Parallel Processing Engine (für >100 Clips / >500 Cuts)
        self.parallel_engine: ParallelPacingEngine | None = None
        if PARALLEL_PROCESSING_AVAILABLE:
            self.parallel_engine = ParallelPacingEngine(
                max_workers=self.config.parallel_max_workers
            )

        # Refactoring Phase 2: Strategy Pattern Initialization
        self.diversity_manager: DiversityManager | None = None
        self.strategies: list[ClipSelectionStrategy] = []
        # Strategies will be populated in _ensure_strategies_initialized()

        logger.info("AdvancedPacingEngine initialisiert")

    def _ensure_strategies_initialized(self, available_clips: list):
        """Build strategy chain if not already initialized."""
        if self.strategies:
            return

        # 1. FAISS Strategy (Motion/Energy Matching + Vector Search)
        # Note: FAISSStrategy handles its own index building via faiss_matcher
        self.strategies.append(FAISSStrategy(self.faiss_matcher, self._clips_by_id_cache))

        # 2. Smart Strategy (Unified Logic)
        if self.smart_matcher and self.use_smart_matching:
            self.strategies.append(SmartStrategy(self.smart_matcher))

        # 3. Legacy Selector Strategy (ClipSelector)
        # Removed in favor of SmartStrategy

        # 4. Round Robin Fallback (Always available)
        self.strategies.append(RoundRobinStrategy())

        logger.info(
            f"Initialized {len(self.strategies)} clip selection strategies: {[s.name for s in self.strategies]}"
        )

    def _get_clips_by_id(self, available_clips: list[dict]) -> dict[int, dict]:
        """
        QUICK WIN #2: Returns cached clips_by_id dictionary (5-8% speedup).

        Caches the dictionary and only rebuilds when clip list changes.
        Uses hash of clip IDs to detect changes efficiently.

        FIX #12: Thread-safe mit Lock für Multi-Threading-Szenarien.

        Args:
            available_clips: List of clip dictionaries

        Returns:
            Dictionary mapping clip_id -> clip_dict
        """
        # Calculate hash of clip IDs (fast O(n) operation)
        current_hash = hash(tuple(c.get("id", 0) for c in available_clips))

        # FIX #12: Thread-safe Cache-Zugriff
        with self._cache_lock:
            # Check if cache is valid
            if self._clips_cache_hash == current_hash and self._clips_by_id_cache is not None:
                # Cache hit! Return cached dict (O(1))
                return self._clips_by_id_cache

            # Cache miss - rebuild
            self._clips_by_id_cache = {c["id"]: c for c in available_clips}
            self._clips_cache_hash = current_hash
            logger.debug(f"clips_by_id cache rebuilt ({len(self._clips_by_id_cache)} clips)")

            return self._clips_by_id_cache

    def _should_use_parallel(self, num_clips: int, num_cuts: int) -> bool:
        """
        Entscheidet ob parallele Verarbeitung verwendet werden soll.

        Kriterien:
        1. Config-Flag enable_parallel_processing
        2. Anzahl Clips > parallel_clip_threshold
        3. Anzahl Cuts > parallel_cut_threshold

        Args:
            num_clips: Anzahl verfügbarer Clips
            num_cuts: Anzahl zu generierender Cuts

        Returns:
            True wenn parallele Verarbeitung aktiviert werden soll
        """
        # Check if parallel engine available
        if not self.parallel_engine:
            return False

        # Explicit config flag
        if self.config.enable_parallel_processing:
            return True

        # Auto-activation based on thresholds
        if num_clips > self.config.parallel_clip_threshold:
            logger.info(
                f"Auto-enabling parallel processing ({num_clips} clips > {self.config.parallel_clip_threshold})"
            )
            return True

        if num_cuts > self.config.parallel_cut_threshold:
            logger.info(
                f"Auto-enabling parallel processing ({num_cuts} cuts > {self.config.parallel_cut_threshold})"
            )
            return True

        return False

    def _prepare_clips_for_parallel(self, clips: list[dict]) -> list[dict]:
        """
        Bereitet Clips für parallele Verarbeitung vor (picklable).

        Konvertiert Clip-Dictionaries in ein Format, das sicher zwischen
        Prozessen serialisiert werden kann (keine GUI-Objekte, nur Primitive).

        Args:
            clips: Liste von Clip-Dictionaries mit allen Metadaten

        Returns:
            Liste von picklable Dictionaries mit essentiellen Clip-Features
        """
        picklable_clips = []

        for clip in clips:
            # Extrahiere nur serialisierbare Daten
            # BUGFIX: Worker erwartet 'path', nicht 'file_path'!
            file_path = str(clip.get("file_path", ""))
            clip_data = {
                "id": clip.get("id"),
                "file_path": file_path,
                "path": file_path,  # Worker erwartet 'path'
                "duration": clip.get("duration", 0.0),
            }

            # Motion & Energy Scores aus Analysis-Daten
            # BUGFIX: Worker erwartet 'motion' und 'energy', nicht '_score' Suffix!
            analysis = clip.get("analysis", {})
            motion_data = analysis.get("motion", {})

            clip_data["motion"] = motion_data.get("motion_score", 0.5)
            clip_data["energy"] = analysis.get("energy_score", 0.5)
            # Backward compatibility
            clip_data["motion_score"] = clip_data["motion"]
            clip_data["energy_score"] = clip_data["energy"]

            # Optional: Mood/Tags
            clip_data["mood"] = clip.get("mood")
            clip_data["tags"] = clip.get("tags", [])

            # Optional: Embeddings (konvertiere zu List wenn NumPy Array)
            if "embedding" in clip:
                embedding = clip["embedding"]
                if hasattr(embedding, "tolist"):  # NumPy Array
                    clip_data["embedding"] = embedding.tolist()
                else:
                    clip_data["embedding"] = embedding

            picklable_clips.append(clip_data)

        return picklable_clips

    # =========================================================================
    # CRITICAL-02 FIX: _calculate_adaptive_clip_duration() → pacing_core.py
    # core_engine.calculate_adaptive_cut_duration() für ADAPTIVE_FLOW mode
    # =========================================================================

    def generate_cut_list(
        self,
        audio_path: str,
        expected_bpm: float | None = None,
        min_cut_interval: float = 0.1,
        start_time: float | None = None,
        end_time: float | None = None,
        phrase_alignment_mode: bool = False,
        progress_callback: PacingProgressCallback | None = None,
    ) -> list[PacingCut]:
        """
        Generiert Schnittliste basierend auf Audio-Triggern.

        Workflow:
        1. Audio analysieren → Trigger extrahieren
        2. Alle Trigger evaluieren (Intensity + Threshold)
        3. Trigger kombinieren und sortieren
        4. Duplikate entfernen (Min-Interval-Check)

        Args:
            audio_path: Pfad zur Audio-Datei
            expected_bpm: Erwartete BPM (optional, verbessert Beat-Tracking)
            min_cut_interval: Minimaler Abstand zwischen Schnitten in Sekunden
            start_time: Optional start time in seconds (for time-windowed analysis)
            end_time: Optional end time in seconds (for time-windowed analysis)
            progress_callback: Optional callback(current, total, message) für Fortschritt
                              Wird mit (step, total_steps, description) aufgerufen

        Returns:
            Liste von PacingCut-Objekten, sortiert nach Zeit

        Raises:
            FileNotFoundError: Wenn Audio-Datei nicht existiert
        """
        # Progress tracking: 8 steps total (1 analysis + 6 trigger types + 1 finalize)
        total_steps = 8
        current_step = 0

        def report_progress(message: str):
            nonlocal current_step
            current_step += 1
            if progress_callback:
                progress_callback(current_step, total_steps, message)

        if start_time is not None and end_time is not None:
            logger.info(
                f"Generiere Schnittliste für: {audio_path} "
                f"(Zeitfenster: {start_time:.1f}s - {end_time:.1f}s, "
                f"PhraseAlignment={'AN' if phrase_alignment_mode else 'AUS'})"
            )
        else:
            logger.info(
                f"Generiere Schnittliste für: {audio_path} "
                f"(PhraseAlignment={'AN' if phrase_alignment_mode else 'AUS'})"
            )

        # Wenn Phrase Alignment aktiv ist, TriggerSystem informieren (optional)
        # oder hier filtern. Wir filtern in _evaluate_triggers.

        # Schritt 1: Trigger extrahieren (mit Zeitfenster falls angegeben)
        # PERFORMANCE: Zeitfenster-Parameter werden jetzt an TriggerSystem weitergegeben
        report_progress("Analysiere Audio-Trigger...")
        analysis = self.trigger_system.analyze_triggers(
            audio_path, expected_bpm, start_time=start_time, end_time=end_time
        )
        logger.debug(f"Trigger-Analyse abgeschlossen: BPM={analysis.bpm:.1f}")
        logger.info(
            f"⏱️ Trigger counts: {len(analysis.beat_times)} beats, {len(analysis.onset_times)} onsets, "
            f"{len(analysis.kick_times)} kicks, {len(analysis.snare_times)} snares, "
            f"{len(analysis.hihat_times)} hihats, {len(analysis.energy_times)} energy peaks"
        )

        # =================================================================
        # LONG-AUDIO-FIX: Trigger-Reduktion für lange Dateien (>30 Min)
        # Problem: 62 Min Audio = 7800+ Beats = UI freeze
        # Lösung: Nur jeden Nten Trigger verwenden
        # =================================================================
        audio_duration_min = analysis.duration / 60.0 if hasattr(analysis, "duration") else 0
        if audio_duration_min == 0 and len(analysis.beat_times) > 0:
            # Schätze Dauer aus letztem Beat
            audio_duration_min = max(analysis.beat_times) / 60.0

        trigger_reduce_factor = 1  # Default: keine Reduktion
        if audio_duration_min > LONG_AUDIO_THRESHOLD_MINUTES:
            trigger_reduce_factor = LONG_AUDIO_TRIGGER_REDUCE_FACTOR
            original_beats = len(analysis.beat_times)
            original_onsets = len(analysis.onset_times)

            # Reduziere alle Trigger-Arrays (jeden Nten behalten)
            analysis.beat_times = analysis.beat_times[::trigger_reduce_factor]
            analysis.onset_times = analysis.onset_times[::trigger_reduce_factor]
            analysis.kick_times = analysis.kick_times[::trigger_reduce_factor]
            analysis.snare_times = analysis.snare_times[::trigger_reduce_factor]
            analysis.hihat_times = analysis.hihat_times[::trigger_reduce_factor]
            analysis.energy_times = analysis.energy_times[::trigger_reduce_factor]

            logger.warning(
                f"⚡ LONG-AUDIO-OPTIMIZATION: {audio_duration_min:.1f} min > {LONG_AUDIO_THRESHOLD_MINUTES} min. "
                f"Trigger-Reduktion aktiviert (Faktor {trigger_reduce_factor}x). "
                f"Beats: {original_beats} → {len(analysis.beat_times)}, "
                f"Onsets: {original_onsets} → {len(analysis.onset_times)}"
            )

        # Schritt 2: Trigger evaluieren und sammeln
        logger.info("📊 Starting trigger evaluation...")
        all_cuts: list[PacingCut] = []

        # FIX: Increased base_strength values to 1.0 so triggers pass threshold checks
        # The GUI sets thresholds (e.g. 70%), and with base_strength=0.6 triggers failed!
        # Now: base_strength=1.0 passes all thresholds, intensity scaling still applies
        # The threshold now acts as expected: lower threshold = more triggers, higher = fewer

        # Beat-Trigger
        report_progress(f"Evaluiere {len(analysis.beat_times)} Beat-Trigger...")
        logger.info(f"🎵 Evaluating {len(analysis.beat_times)} beat triggers...")
        logger.debug("Evaluating beat triggers...")
        beat_cuts = self.core_engine.evaluate_triggers(
            trigger_type="beat",
            times=analysis.beat_times,
            base_strength=1.0,  # FIX: Was 0.8, now 1.0 to pass threshold checks
            start_time=start_time,
            end_time=end_time,
            filter_downbeats=phrase_alignment_mode,
        )
        logger.info(
            f"  ✓ Beat evaluation complete: {len(beat_cuts)} cuts from {len(analysis.beat_times)} triggers"
        )
        all_cuts.extend(beat_cuts)
        logger.debug(f"Beat triggers evaluated: {len(all_cuts)} cuts so far")

        # Onset-Trigger
        report_progress(f"Evaluiere {len(analysis.onset_times)} Onset-Trigger...")
        logger.debug("Evaluating onset triggers...")
        all_cuts.extend(
            self.core_engine.evaluate_triggers(
                trigger_type="onset",
                times=analysis.onset_times,
                base_strength=1.0,  # FIX: Was 0.6, now 1.0 to pass threshold checks
                start_time=start_time,
                end_time=end_time,
            )
        )
        logger.debug(f"Onset triggers evaluated: {len(all_cuts)} cuts so far")

        # Kick-Trigger
        report_progress(f"Evaluiere {len(analysis.kick_times)} Kick-Trigger...")
        logger.debug("Evaluating kick triggers...")
        all_cuts.extend(
            self.core_engine.evaluate_triggers(
                trigger_type="kick",
                times=analysis.kick_times,
                base_strength=1.0,  # FIX: Was 0.7, now 1.0 to pass threshold checks
                start_time=start_time,
                end_time=end_time,
            )
        )
        logger.debug(f"Kick triggers evaluated: {len(all_cuts)} cuts so far")

        # Snare-Trigger
        report_progress(f"Evaluiere {len(analysis.snare_times)} Snare-Trigger...")
        logger.debug("Evaluating snare triggers...")
        all_cuts.extend(
            self.core_engine.evaluate_triggers(
                trigger_type="snare",
                times=analysis.snare_times,
                base_strength=1.0,  # FIX: Was 0.65, now 1.0 to pass threshold checks
                start_time=start_time,
                end_time=end_time,
            )
        )
        logger.debug(f"Snare triggers evaluated: {len(all_cuts)} cuts so far")

        # HiHat-Trigger
        report_progress(f"Evaluiere {len(analysis.hihat_times)} HiHat-Trigger...")
        logger.debug("Evaluating hihat triggers...")
        all_cuts.extend(
            self.core_engine.evaluate_triggers(
                trigger_type="hihat",
                times=analysis.hihat_times,
                base_strength=0.9,  # FIX: Was 0.5, now 0.9 (slightly lower for lighter triggers)
                start_time=start_time,
                end_time=end_time,
            )
        )
        logger.debug(f"HiHat triggers evaluated: {len(all_cuts)} cuts so far")

        # Energy-Trigger
        report_progress(f"Evaluiere {len(analysis.energy_times)} Energy-Trigger...")
        logger.debug("Evaluating energy triggers...")
        all_cuts.extend(
            self.core_engine.evaluate_triggers(
                trigger_type="energy",
                times=analysis.energy_times,
                base_strength=1.0,  # Energy-Peaks sind sehr stark (unchanged)
                start_time=start_time,
                end_time=end_time,
            )
        )

        logger.info(f"✅ All triggers evaluated: {len(all_cuts)} potential cuts")

        # Schritt 3: Nach Zeit sortieren und Duplikate entfernen
        report_progress(f"Finalisiere {len(all_cuts)} Schnitte...")
        logger.debug(f"Sorting {len(all_cuts)} cuts by time...")
        all_cuts.sort(key=lambda c: c.time)
        logger.debug("Cuts sorted")

        # Schritt 4: Duplikate entfernen (Min-Interval-Check)
        logger.debug(f"Removing close cuts (min_interval={min_cut_interval}s)...")
        filtered_cuts = self.core_engine.remove_close_cuts(all_cuts, min_cut_interval)
        logger.debug(f"Close cuts removed: {len(filtered_cuts)} remaining")

        logger.info(
            f"Schnittliste generiert: {len(filtered_cuts)} Schnitte "
            f"(von {len(all_cuts)} Kandidaten)"
        )

        return filtered_cuts

    # =========================================================================
    # CRITICAL-02 FIX: Alte Methoden entfernt (→ pacing_core.py)
    # _evaluate_triggers() → core_engine.evaluate_triggers()
    # _remove_close_cuts() → core_engine.remove_close_cuts()
    # =========================================================================

    def get_cut_times(
        self, audio_path: str, expected_bpm: float | None = None, min_cut_interval: float = 0.1
    ) -> list[float]:
        """
        Generiert Schnittliste und gibt nur die Zeitpunkte zurück.

        Convenience-Methode für einfache Integration.

        Args:
            audio_path: Pfad zur Audio-Datei
            expected_bpm: Erwartete BPM (optional)
            min_cut_interval: Minimaler Abstand zwischen Schnitten

        Returns:
            Liste von Schnitt-Zeitpunkten in Sekunden
        """
        cuts = self.generate_cut_list(audio_path, expected_bpm, min_cut_interval)
        return [cut.time for cut in cuts]

    def get_trigger_statistics(self, cuts: list[PacingCut]) -> dict[str, int]:
        """
        Berechnet Statistiken über verwendete Trigger-Typen.

        Args:
            cuts: Liste von PacingCut-Objekten

        Returns:
            Dict mit Trigger-Typ → Anzahl
        """
        stats: dict[str, int] = {}

        for cut in cuts:
            stats[cut.trigger_type] = stats.get(cut.trigger_type, 0) + 1

        return stats

    # =========================================================================
    # M-04 FIX: Extracted helper methods for better maintainability
    # =========================================================================

    # =========================================================================
    # CRITICAL-02 FIX: _determine_target_properties() → pacing_structure.py
    # structure_manager.determine_target_properties() für segment-aware matching
    # =========================================================================

    def enable_motion_matching(self, enabled: bool = True):
        """
        Aktiviert/deaktiviert Motion-Energy-Matching.

        Args:
            enabled: True = Motion-Matching aktivieren, False = deaktivieren
        """
        self.use_motion_matching = enabled
        logger.info(f"Motion-Energy-Matching: " f"{'aktiviert' if enabled else 'deaktiviert'}")

    def enable_faiss_matching(
        self,
        enabled: bool = True,
        use_gpu: bool = False,
        backend: str = "auto",
        use_ai_embeddings: bool = False,
    ) -> None:
        """
        Aktiviert Vector Search Motion Matching (Phase 3).

        Multi-Platform Support:
        - NVIDIA GPU → FAISS GPU (5-10x faster than CPU)
        - AMD GPU → Qdrant (1.5-3x faster than FAISS CPU)
        - CPU → Auto-selects best available (Qdrant or FAISS CPU)

        Performance:
        - O(log n) statt O(n) pro Query
        - 100-1000x schneller als Brute-Force
        - Automatic platform detection

        Args:
            enabled: Vector Search aktivieren
            use_gpu: GPU-Acceleration verwenden (wenn verfügbar)
            backend: Backend auswählen ('auto', 'faiss-cpu', 'faiss-gpu', 'qdrant')
                    'auto' = automatische Erkennung (empfohlen)
            use_ai_embeddings: KI-basierte 512D Embeddings (X-CLIP) statt manueller 27D Features

        Note:
            Install mindestens einen Backend:
            - CPU: pip install faiss-cpu
            - NVIDIA GPU: pip install faiss-gpu
            - AMD GPU / Fast CPU: pip install qdrant-client
            Oder: poetry install -E vector-all (installiert alle)
            Für KI-Embeddings: poetry install -E ai-video
        """
        if not VECTOR_SEARCH_AVAILABLE:
            logger.warning(
                "Vector search not available. Install at least one backend:\n"
                "  pip install faiss-cpu        # CPU (basic)\n"
                "  pip install faiss-gpu        # NVIDIA GPU\n"
                "  pip install qdrant-client    # AMD GPU / Optimized CPU\n"
                "Or use poetry: poetry install -E vector-all"
            )
            self.use_faiss = False
            return

        self.use_faiss = enabled
        self.use_ai_embeddings = use_ai_embeddings

        if enabled:
            if self.clip_matcher is None:
                # Auto-detect best backend
                self.clip_matcher = create_clip_matcher(
                    backend=backend, use_gpu=use_gpu, use_ai_embeddings=use_ai_embeddings
                )
                # Backward compatibility alias
                self.faiss_matcher = self.clip_matcher

                backend_info = self.clip_matcher.get_index_stats()
                embedding_type = backend_info.get("embedding_type", "27D Manual")
                logger.info(
                    f"Vector search enabled: {backend_info.get('backend', 'Unknown')} "
                    f"({embedding_type})"
                )
            else:
                logger.info("Vector search matcher already initialized")
        else:
            logger.info("Vector search matching disabled")

    def enable_semantic_matching(self, enabled: bool = True) -> None:
        """
        Aktiviert Semantic Vibe-Matching (CLIP-basiert).
        """
        if enabled and not SEMANTIC_SEARCH_AVAILABLE:
            logger.warning("Semantic search not available (missing dependencies).")
            return

        self.use_semantic_matching = enabled
        if enabled:
            if self.semantic_matcher is None:
                self.semantic_matcher = SemanticClipMatcher()
            logger.info("Semantic Vibe-Matching enabled")
        else:
            logger.info("Semantic Vibe-Matching disabled")

    def generate_cut_list_with_clips(
        self,
        audio_path: str,
        available_clips: list[dict],
        expected_bpm: float | None = None,
        min_cut_interval: float = 0.1,
        start_time: float | None = None,
        end_time: float | None = None,
        phrase_alignment_mode: bool = False,
        progress_callback: PacingProgressCallback | None = None,
        pacing_mode: PacingMode | None = None,
    ) -> list[tuple[PacingCut, dict]]:
        """
        Generiert Schnittliste MIT intelligenter Clip-Auswahl.

         Refactored to use Strategy Pattern (Phase 2).
        """
        # Use instance pacing_mode if not provided as parameter
        active_mode = pacing_mode if pacing_mode is not None else self.pacing_mode

        # Adaptive Flow Mode: Increase min_cut_interval for longer clips
        if active_mode == PacingMode.ADAPTIVE_FLOW:
            min_cut_interval = max(min_cut_interval, 4.0)
            logger.info(f"🎬 ADAPTIVE FLOW MODE: min_cut_interval set to {min_cut_interval}s")

        # Progress forwarding wrapper
        def phase1_callback(current: int, total: int, message: str):
            if progress_callback:
                progress_callback(current, total * 2, f"[1/2] {message}")

        # 1. Trigger extrahieren (Phase 1)
        pacing_cuts = self.generate_cut_list(
            audio_path,
            expected_bpm,
            min_cut_interval,
            start_time=start_time,
            end_time=end_time,
            phrase_alignment_mode=phrase_alignment_mode,
            progress_callback=phase1_callback,
        )

        logger.info(
            f"Generiere Clip-Auswahl für {len(pacing_cuts)} Schnitte "
            f"(FAISS={'aktiviert' if self.use_faiss else 'deaktiviert'}, "
            f"Motion-Matching={'aktiviert' if self.use_motion_matching else 'deaktiviert'})"
        )

        # Prepare Caches
        clips_by_id = self._get_clips_by_id(available_clips)

        # Initialize Strategies
        self._ensure_strategies_initialized(available_clips)

        # Initialize Diversity Manager
        # RESET_THRESHOLD: 80% (same as before)
        self.diversity_manager = DiversityManager(
            total_clips=len(available_clips),
            reset_threshold_percent=CLIP_DIVERSITY_RESET_THRESHOLD_ALT,
        )

        # ---------------------------------------------------------------------
        # Analysis Phase (Structure & Energy)
        # ---------------------------------------------------------------------
        if self.use_energy_matching:
            try:
                self.energy_curve = self.energy_analyzer.analyze_energy(audio_path)
            except Exception as e:
                logger.warning(f"Energy curve analysis failed: {e}")
                self.energy_curve = None

        MAX_DURATION_FOR_STRUCTURE = 120
        audio_duration = (
            self.energy_curve.duration
            if (self.energy_curve and hasattr(self.energy_curve, "duration"))
            else 0
        )

        if audio_duration > MAX_DURATION_FOR_STRUCTURE:
            logger.warning("Skipping structure analysis (audio too long)")
            self.structure_result = None
        elif self.use_structure_awareness and self.config.enable_structure_analysis:
            try:
                if self.structure_result is None or self.structure_result.audio_path != audio_path:
                    self.structure_result = self.structure_analyzer.analyze_structure(audio_path)
            except Exception as e:
                logger.warning(f"Structure analysis failed: {e}")
                self.structure_result = None

        # ---------------------------------------------------------------------
        # Strategy Preparation
        # ---------------------------------------------------------------------
        for strategy in self.strategies:
            if hasattr(strategy, "prepare"):
                strategy.prepare(available_clips)

        # ---------------------------------------------------------------------
        # Parallel Execution Path (Shortcut)
        # ---------------------------------------------------------------------
        use_parallel = self._should_use_parallel(len(available_clips), len(pacing_cuts))
        if use_parallel and self.parallel_engine:
            # ... Logic for parallel execution ...
            # Note: For now we delegate parallel completely if triggered
            # This duplicates some logic but parallel engine is separate anyway
            # In future we might want to strategy-fy parallel too
            pass  # Parallel logic is complex and specific, keeping original block structure might be cleaner OR copy-paste it here.
            # Given "Replace the >600 lines", I need to reimplement or copy the logic.
            # I will COPY the parallel block and then the sequential loop.

        # NOTE: To keep this clean, I will re-use the exact parallel block from original code,
        # but I'm replacing the whole method so I must include it.

        if use_parallel and self.parallel_engine:
            logger.info(f"Using PARALLEL processing ({len(pacing_cuts)} cuts)")
            # ... existing parallel implementation ...
            # Note: I am simplifying this for brevity in thought but code must be full.
            # I will copy the parallel implementation block.

            tasks = []
            for cut_idx, cut in enumerate(pacing_cuts):
                audio_energy = cut.strength
                if self.energy_curve:
                    try:
                        audio_energy = self.energy_curve.get_energy_at_time(cut.time)
                    except:
                        pass
                task = ClipSelectionTask(cut_idx, audio_energy, audio_energy, cut.time, cut.time)
                tasks.append(task)

            clips_features = self._prepare_clips_for_parallel(available_clips)

            def parallel_progress_callback(current: int, total: int, message: str):
                if progress_callback:
                    progress_callback(
                        len(pacing_cuts) + current, len(pacing_cuts) * 2, f"[2/2] {message}"
                    )

            results = self.parallel_engine.generate_cutlist_parallel(
                tasks=tasks,
                clips_features=clips_features,
                progress_callback=parallel_progress_callback,
            )

            final_result = []
            for res in results:
                cut = pacing_cuts[res.task_id]
                clip_dict = clips_by_id.get(res.clip_id)
                if not clip_dict:
                    # Fallback dict
                    clip_dict = {
                        "id": res.clip_id,
                        "file_path": res.clip_path,
                        "duration": res.duration,
                        "analysis": {
                            "energy_score": res.energy,
                            "motion": {"motion_score": res.motion},
                        },
                    }
                final_result.append((cut, clip_dict))
            return final_result

        # ---------------------------------------------------------------------
        # Sequential Execution Path (Strategy Loop)
        # ---------------------------------------------------------------------
        logger.info("Using SEQUENTIAL processing with Strategy Pattern")

        result: list[tuple[PacingCut, dict]] = []

        # Loop Vars
        last_clip_id = None
        last_clip_data = None
        last_clip_energy = None
        clip_repeat_count = 0
        total_repeats = 0

        MAX_CLIP_REPEATS = 3 if self.config.enable_clip_loop else 0
        ENERGY_SIMILARITY_THRESHOLD = 0.15 if self.config.enable_clip_loop else 0.0

        # UI Updates setup
        total_pacing_cuts = len(pacing_cuts)
        progress_interval = max(1, total_pacing_cuts // 100)
        batch_size = LONG_AUDIO_BATCH_SIZE
        ui_update_interval = LONG_AUDIO_UI_UPDATE_INTERVAL

        _qapp = None
        try:
            from PyQt6.QtWidgets import QApplication

            _qapp = QApplication.instance()
        except ImportError:
            pass

        for cut_idx, cut in enumerate(pacing_cuts):
            # UI Updates & Progress
            should_update_progress = progress_callback and (
                cut_idx % progress_interval == 0 or cut_idx % ui_update_interval == 0
            )
            if should_update_progress:
                progress_callback(
                    total_pacing_cuts + cut_idx,
                    total_pacing_cuts * 2,
                    f"[2/2] Clip-Auswahl: {cut_idx}/{total_pacing_cuts}",
                )

            if _qapp and (cut_idx % batch_size == 0 or cut_idx % ui_update_interval == 0):
                _qapp.processEvents()

            # Diversity Management
            if self.diversity_manager.check_and_reset():
                pass  # Already logged in manager

            # Determine Targets
            if self.energy_curve:
                try:
                    audio_energy = self.energy_curve.get_energy_at_time(cut.time)
                except:
                    audio_energy = cut.strength
            else:
                audio_energy = cut.strength

            segment_type = None
            if self.structure_result:
                segment = self.structure_analyzer.get_segment_at_time(
                    self.structure_result, cut.time
                )
                if segment:
                    segment_type = segment.segment_type

            (
                varied_energy,
                varied_motion,
                target_motion_type,
                target_mood,
            ) = self.structure_manager.determine_target_properties(
                cut, cut_idx, audio_energy, segment_type
            )

            # Clip Loop Logic (Repeat Check)
            should_repeat = False
            if last_clip_data and last_clip_energy is not None:
                energy_diff = abs(audio_energy - last_clip_energy)
                if (
                    energy_diff <= ENERGY_SIMILARITY_THRESHOLD
                    and clip_repeat_count < MAX_CLIP_REPEATS
                ):
                    should_repeat = True

            selected_clip = None
            strategy_used = "None"

            if should_repeat:
                selected_clip = last_clip_data
                clip_repeat_count += 1
                total_repeats += 1
                strategy_used = "Repeat"
                # logger.debug(f"Repeating clip {selected_clip.get('id')}")
            else:
                # Strategy Selection Loop
                excluded_clips = self.diversity_manager.excluded_clips

                for strategy in self.strategies:
                    if not strategy.is_available():
                        continue

                    # Skip FAISS if disabled (config override)
                    if strategy.name == "FAISS" and not self.use_faiss:
                        continue

                    # Execute Strategy
                    selection = strategy.select_clip(
                        target_energy=varied_energy,
                        target_motion=varied_motion,
                        target_motion_type=target_motion_type,
                        target_mood=target_mood,
                        excluded_clips=excluded_clips,
                        available_clips=available_clips,
                        min_duration=cut.time,  # Optional hint for SmartStrategy
                        previous_clip_id=last_clip_id,  # Hint for continuity
                        continuity_weight=self.config.continuity_weight,  # VISUAL CONTINUITY
                    )

                    if selection:
                        selected_clip = selection.clip_data
                        strategy_used = strategy.name

                        # Add match quality for analysis
                        if isinstance(selected_clip, dict):
                            selected_clip["match_quality"] = selection.distance
                            # bpm_match removed as it is not in ClipSelectionResult

                        break

                if not selected_clip:
                    # Absolute fallback if all strategies fail (should not happen due to RoundRobin)
                    if available_clips:
                        selected_clip = available_clips[cut_idx % len(available_clips)]
                        strategy_used = "UltimateFallback"
                    else:
                        logger.error("No clips available!")
                        continue

                # Update Loop Tracking
                last_clip_id = selected_clip.get("id")
                last_clip_data = selected_clip
                last_clip_energy = audio_energy
                clip_repeat_count = 0

                # Record Usage
                if last_clip_id is not None:
                    self.diversity_manager.record_used(last_clip_id)

            # Store Result
            result.append((cut, selected_clip))

            # Logging (reduced frequency)
            if cut_idx < 3 or (cut_idx % 500 == 0 and logger.isEnabledFor(logging.INFO)):
                logger.info(
                    f"Cut {cut_idx}: Selected via {strategy_used} (ID: {selected_clip.get('id')})"
                )

        total_time = 0  # Placeholder if needed
        # Log Summary
        stats = self.diversity_manager.get_statistics()
        logger.info(f"Selection complete. {len(result)} cuts. Diversity: {stats}")

        return result

    # =========================================================================
    # CRITICAL-02 FIX: Delegate Methods (Backward Compatibility)
    # Diese Methoden delegieren zu den neuen modularen Engines
    # =========================================================================

    def _energy_to_motion(self, energy: float) -> str:
        """Delegate zu IntensityMapper.energy_to_motion()."""
        return self.intensity_mapper.energy_to_motion(energy)

    def _energy_to_mood(self, energy: float, cut_idx: int = 0) -> str:
        """Delegate zu IntensityMapper.energy_to_mood()."""
        return self.intensity_mapper.energy_to_mood(energy, cut_idx)

    def enable_structure_awareness(self, enabled: bool = True):
        """Delegate zu StructureManager.enable_structure_awareness()."""
        self.use_structure_awareness = enabled
        self.structure_manager.enable_structure_awareness(enabled)

    def analyze_song_structure(self, audio_path: str) -> StructureAnalysisResult:
        """Delegate zu StructureManager.analyze_song_structure()."""
        self.structure_result = self.structure_manager.analyze_song_structure(audio_path)
        return self.structure_result

    def generate_cut_list_with_structure(
        self,
        audio_path: str,
        expected_bpm: float | None = None,
        min_cut_interval: float = 0.1,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[PacingCut]:
        """
        Generiert Schnittliste MIT Struktur-Awareness.

        Erweitert generate_cut_list() um:
        - Song-Struktur-Analyse
        - Segment-Type-Annotation in PacingCut
        - Segment-basierte Cut-Dichte-Anpassung

        Args:
            audio_path: Pfad zur Audio-Datei
            expected_bpm: Erwartete BPM (optional)
            min_cut_interval: Minimaler Abstand zwischen Schnitten
            start_time: Optional start time for time-window optimization
            end_time: Optional end time for time-window optimization

        Returns:
            Liste von PacingCut-Objekten mit segment_type
        """
        logger.info("Generiere Schnittliste mit Struktur-Awareness")

        # Schritt 1: Song-Struktur analysieren
        if not self.structure_result or self.structure_result.audio_path != audio_path:
            self.analyze_song_structure(audio_path)

        # Schritt 2: Normale Schnittliste generieren mit Time-Window-Optimization
        cuts = self.generate_cut_list(
            audio_path, expected_bpm, min_cut_interval, start_time=start_time, end_time=end_time
        )

        # Schritt 3: Segment-Type annotieren
        annotated_cuts = []
        for cut in cuts:
            segment = self.structure_analyzer.get_segment_at_time(self.structure_result, cut.time)

            # Neues PacingCut mit segment_type
            annotated_cut = PacingCut(
                time=cut.time,
                trigger_type=cut.trigger_type,
                strength=cut.strength,
                raw_strength=cut.raw_strength,
                segment_type=segment.segment_type if segment else None,
            )

            annotated_cuts.append(annotated_cut)

        logger.info(f"Struktur-Awareness abgeschlossen: {len(annotated_cuts)} annotierte Schnitte")

        return annotated_cuts

    # =========================================================================
    # Phase 2: Music-Video Mapping Integration
    # =========================================================================

    def enable_music_video_mapping(self, enabled: bool = True) -> bool:
        """
        Enable/disable Phase 2 Music-Video Mapping features.

        Features when enabled:
        - Audio-derived color palette matching (LAB/CIEDE2000)
        - Beat-level synchronization scoring
        - Emotion-based clip selection

        Args:
            enabled: Whether to enable music-video mapping

        Returns:
            True if successfully enabled, False if modules not available
        """
        if enabled and not MUSIC_VIDEO_MAPPING_AVAILABLE:
            logger.warning(
                "Music-Video Mapping modules not available. "
                "Check audio_visual_mapper.py and motion_music_sync.py"
            )
            return False

        self.use_music_video_mapping = enabled
        logger.info(f"Phase 2 Music-Video Mapping: {'enabled' if enabled else 'disabled'}")
        return True

    def analyze_audio_for_visual_mapping(
        self, audio_path: str, spectral_features: dict | None = None
    ) -> Optional["AudioDerivedPalette"]:
        """
        Analyze audio to generate color palette for visual matching.

        Uses Russell's Circumplex Model (valence/arousal) to map audio
        features to emotional color palettes in LAB color space.

        Args:
            audio_path: Path to audio file
            spectral_features: Optional pre-computed spectral features

        Returns:
            AudioDerivedPalette with colors matched to audio mood,
            or None if mapping unavailable
        """
        if not self.audio_visual_mapper:
            logger.warning("AudioVisualMapper not available")
            return None

        # Get spectral features if not provided
        if spectral_features is None:
            try:
                from ..audio.audio_analyzer import AudioAnalyzer

                analyzer = AudioAnalyzer()
                spectral_features = analyzer.extract_spectral_features(audio_path)
            except Exception as e:
                logger.error(f"Failed to extract spectral features: {e}")
                return None

        if not spectral_features:
            logger.warning("No spectral features available for mapping")
            return None

        # Determine segment type from structure analysis
        segment_type = None
        if self.structure_result and self.use_structure_awareness:
            # Use the dominant segment type
            if self.structure_result.segments:
                segment_type = self.structure_result.segments[0].segment_type

        # Generate audio-derived color palette
        self._audio_derived_palette = self.audio_visual_mapper.generate_color_palette_from_audio(
            spectral_features=spectral_features, segment_type=segment_type
        )

        logger.info(
            f"Audio-derived palette generated: mood={self._audio_derived_palette.mood.value}, "
            f"temp={self._audio_derived_palette.temperature:.2f}, "
            f"brightness={self._audio_derived_palette.brightness:.2f}"
        )

        return self._audio_derived_palette

    def score_clip_color_match(
        self,
        clip_colors: list[tuple[int, int, int]],
        audio_palette: Optional["AudioDerivedPalette"] = None,
    ) -> float:
        """
        Score how well a clip's colors match the audio-derived palette.

        Uses CIEDE2000 perceptual color distance for accurate matching
        in LAB color space (40% improvement over RGB Euclidean).

        Args:
            clip_colors: List of dominant RGB colors from clip
            audio_palette: Audio-derived palette (uses cached if not provided)

        Returns:
            Score from 0.0 (no match) to 1.0 (perfect match)
        """
        if not self.audio_visual_mapper:
            return 0.5  # Neutral score if mapping unavailable

        palette = audio_palette or self._audio_derived_palette
        if not palette:
            return 0.5  # Neutral score if no palette

        return self.audio_visual_mapper.score_color_palette_similarity(
            audio_palette=palette, clip_colors=clip_colors
        )

    def set_beat_times(self, beat_times: list[float]) -> None:
        """
        Set beat times for synchronization scoring.

        Args:
            beat_times: List of beat times in seconds
        """
        self._beat_times = sorted(beat_times)
        logger.info(f"Beat times set: {len(self._beat_times)} beats")

    def calculate_sync_quality(self, cut_list: list[PacingCut]) -> tuple[float, dict]:
        """
        Calculate synchronization quality between cuts and audio beats.

        Uses motion_music_sync module for beat-level sync analysis.
        Target: 95% of cuts within 200ms of nearest beat.

        Args:
            cut_list: List of PacingCut objects with timing

        Returns:
            Tuple of (sync_score 0-1, detailed_stats dict)
        """
        if not self.motion_sync or not self._beat_times:
            return 0.5, {"error": "Motion sync or beats not available"}

        # Extract cut times
        cut_times = [cut.time for cut in cut_list]

        # Calculate sync score using motion_music_sync
        score, stats = self.motion_sync.calculate_sync_score(
            cutlist_times=cut_times, audio_beats=self._beat_times
        )

        # Log sync quality
        quality = "EXCELLENT" if score >= 0.9 else "GOOD" if score >= 0.7 else "ACCEPTABLE"
        logger.info(f"Sync quality: {quality} (score={score:.3f})")

        return score, stats

    def optimize_clip_alignment(
        self, target_beat_time: float, clip_motion_times: list[float], clip_duration: float
    ) -> Optional["ClipBeatAlignment"]:
        """
        Find optimal clip start time to align motion with beat.

        Uses MotionMusicSynchronizer to find best alignment point.

        Args:
            target_beat_time: When the beat occurs
            clip_motion_times: When motion events occur in clip
            clip_duration: Total clip duration

        Returns:
            ClipBeatAlignment with optimal start time and sync quality
        """
        if not self.motion_sync:
            return None

        return self.motion_sync.find_optimal_clip_start(
            target_beat_time=target_beat_time,
            clip_motion_times=clip_motion_times,
            clip_duration=clip_duration,
        )

    def get_emotion_from_audio(self, spectral_features: dict) -> tuple[float, float] | None:
        """
        Map audio spectral features to emotion coordinates (valence, arousal).

        Uses Russell's Circumplex Model:
        - Valence: -1 (negative/sad) to +1 (positive/happy)
        - Arousal: -1 (calm/low-energy) to +1 (excited/high-energy)

        Args:
            spectral_features: Dict with brightness, richness, etc.

        Returns:
            Tuple of (valence, arousal) or None if mapping unavailable
        """
        if not self.audio_visual_mapper:
            return None

        emotion = self.audio_visual_mapper.map_audio_to_emotion_space(spectral_features)
        return (emotion.valence, emotion.arousal)

    def get_motion_from_energy(self, energy_level: float) -> str:
        """
        Map energy level to MotionType using AudioVisualMapper.

        Consistent mapping across all Phase 2 components.

        Args:
            energy_level: Energy value 0.0 to 1.0

        Returns:
            MotionType string (STATIC, SLOW, MEDIUM, FAST, EXTREME)
        """
        if self.audio_visual_mapper:
            motion_type = self.audio_visual_mapper.map_energy_to_motion(energy_level)
            return motion_type.value
        # Fallback to built-in method
        return self._energy_to_motion(energy_level)
