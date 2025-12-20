"""
CutList Controller - Handles cut list generation operations.

Extracted from MainWindow God Object (P1.6).

VERBESSERT (2024-12): Variable Clip-Längen und intelligente Startpunkte
- Clips können unterschiedlich lang sein (nicht mehr alle gleich)
- Startpunkt im Clip wird intelligent gewählt (nicht immer 0.0)
- Bei guter Passung werden längere Clips verwendet
"""

import logging
import random
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QMessageBox

from ...pacing.clip_calculation import calculate_intelligent_clip_segment
from ...pacing.dynamic_duration import DurationConstraints, DynamicDurationCalculator
from ...pacing.pacing_models import CutListEntry

# Type alias für Progress-Callback
PacingProgressCallback = Callable[[int, int, str], None]
from ...utils.path_utils import resolve_relative_path

if TYPE_CHECKING:
    from ..main_window import MainWindow

logger = logging.getLogger(__name__)


class CutListController:
    """
    Manages cut list generation operations.

    Responsibilities:
    - Validate prerequisites for cut list generation
    - Create trigger settings from dashboard parameters
    - Generate motion-matching cuts (FAISS)
    - Generate simple round-robin cuts
    - Coordinate with pacing engine
    """

    def __init__(self, main_window: "MainWindow"):
        """
        Initialize cut list controller.

        Args:
            main_window: Reference to main window for UI access
        """
        self.main_window = main_window

    def validate_prerequisites(self) -> tuple[str, list] | None:
        """
        Validate prerequisites for cut list generation.

        Returns:
            Tuple of (audio_path, clips) if valid, None otherwise
        """
        # Get audio path from timeline
        if not self.main_window.timeline_widget.audio_path:
            logger.error("Kein Audio geladen - kann keine Schnittliste generieren")
            QMessageBox.warning(
                self.main_window, "Kein Audio", "Bitte zuerst eine Audio-Datei laden!"
            )
            return None

        audio_path = str(self.main_window.timeline_widget.audio_path)

        # Get available clips
        clips = self.main_window.clip_library_widget.clips
        if not clips:
            logger.warning("Keine Video-Clips verfügbar für Schnittliste")
            return None

        return (audio_path, clips)

    def create_trigger_settings(self, params: dict):
        """
        Create trigger settings from dashboard parameters.

        Args:
            params: Dashboard parameters

        Returns:
            Configured TriggerIntensitySettings
        """
        from ...pacing.intensity_controller import TriggerIntensitySettings

        trigger_settings = TriggerIntensitySettings()

        # Update trigger settings based on dashboard parameters
        # FIX: Default all triggers to ENABLED (matching GUI checkboxes)
        # Old bug: kick/snare/hihat defaulted to False even though GUI showed them enabled
        for trigger_type in ["beat", "onset", "kick", "snare", "hihat", "energy"]:
            enabled_key = f"{trigger_type}_enabled"
            intensity_key = f"{trigger_type}_intensity"
            threshold_key = f"{trigger_type}_threshold"

            trigger_settings.set_config(
                trigger_type,
                enabled=params.get(enabled_key, True),  # FIX: Default=True (all triggers enabled)
                intensity=params.get(intensity_key, 80),
                threshold=params.get(threshold_key, 30),
            )

        # Log trigger configuration for debugging
        logger.info(
            "🎛️ Trigger config: "
            + ", ".join(
                [
                    f"{t}={'✓' if trigger_settings.get_config(t).enabled else '✗'}"
                    for t in ["beat", "onset", "kick", "snare", "hihat", "energy"]
                ]
            )
        )

        return trigger_settings

    def generate_motion_matching_cuts(
        self,
        pacing_engine,
        audio_path: str,
        clips: list,
        expected_bpm: float | None,
        target_duration: float,
        use_structure_awareness: bool,
        duration_limit: float | None = None,
        min_cut_interval: float = 4.0,  # FIX: Added - controls minimum clip duration
        phrase_alignment_mode: bool = False,
        progress_callback: PacingProgressCallback | None = None,
    ) -> list[CutListEntry]:
        """
        Generate cut list using motion-matching (FAISS).

        Args:
            pacing_engine: Configured AdvancedPacingEngine
            audio_path: Path to audio file
            clips: Available video clips
            expected_bpm: Expected BPM
            target_duration: Target video duration
            use_structure_awareness: Enable structure-aware cutting
            duration_limit: Optional duration limit for time-window optimization
            min_cut_interval: Minimum time between cuts in seconds (default 4.0)
            progress_callback: Optional callback(current, total, message) für Fortschritt

        Returns:
            List of CutListEntry objects
        """
        # Enable motion matching and FAISS
        pacing_engine.enable_motion_matching(True)
        # Auto-detect: Nutzt GPU falls CUDA verfügbar, sonst optimierter CPU-Modus
        pacing_engine.enable_faiss_matching(enabled=True, use_gpu=True)

        # PERFORMANCE FIX: Use time-window optimization if duration_limit is set
        # This significantly reduces trigger analysis time (e.g., 30s instead of 60 minutes)
        start_time = None
        end_time = None
        if duration_limit is not None:
            start_time = 0.0
            end_time = duration_limit
            logger.info(f"Using time-window optimization: 0.0s - {duration_limit}s")

        # Generate cuts WITH clip selection
        # PERFORMANCE FIX: Deaktiviere Struktur-Analyse für lange Tracks (>10 Min)
        MAX_DURATION_FOR_STRUCTURE = 600  # 10 Minuten
        if use_structure_awareness and target_duration > MAX_DURATION_FOR_STRUCTURE:
            logger.warning(
                f"Track zu lang für Struktur-Analyse ({target_duration:.1f}s > {MAX_DURATION_FOR_STRUCTURE}s) "
                f"- deaktiviere Struktur-Awareness für optimale Performance"
            )
            use_structure_awareness = False

        if use_structure_awareness:
            # CRITICAL FIX: Structure-awareness needs FAISS too!
            # 1. Analyze structure and generate structure-aware cuts
            pacing_engine.analyze_song_structure(audio_path)
            pacing_cuts = pacing_engine.generate_cut_list_with_structure(
                audio_path=audio_path,
                expected_bpm=expected_bpm,
                min_cut_interval=min_cut_interval,  # Use GUI slider value
                start_time=start_time,
                end_time=end_time,
                phrase_alignment_mode=phrase_alignment_mode,
            )

            # 2. Use FAISS for clip selection (same as normal path)
            if pacing_engine.use_faiss and pacing_engine.faiss_matcher:
                logger.info(f"Structure-Awareness using FAISS for {len(pacing_cuts)} cuts")

                # Build FAISS index
                pacing_engine.faiss_matcher.build_index(clips)

                cut_with_clips = []
                # FIX #11/#12: Use set instead of list for better performance and diversity
                used_clips_set = set()
                # FIX #11: Reduced from 80% to 50% for better clip diversity
                reset_threshold = int(len(clips) * 0.5)

                # Track previous clip for continuity
                last_clip_id = None
                continuity_weight = pacing_engine.config.continuity_weight

                for cut_idx, cut in enumerate(pacing_cuts):
                    # Reset exclusion list when needed
                    if len(used_clips_set) >= reset_threshold:
                        used_clips_set.clear()

                    # Use FAISS to find best clip
                    # FIX #12: Use set directly (no conversion overhead)
                    # VISUAL CONTINUITY: Pass previous_clip_id and weight
                    clip_id, file_path, distance = pacing_engine.faiss_matcher.find_best_clip(
                        target_motion_score=cut.strength,
                        target_energy=cut.strength,
                        target_motion_type="MEDIUM",
                        target_moods=[],
                        k=min(200, len(clips)),
                        exclude_ids=list(used_clips_set),
                        previous_clip_id=last_clip_id,
                        continuity_weight=continuity_weight,
                    )

                    # Update last clip ID
                    last_clip_id = clip_id

                    # Find full clip dict
                    selected_clip = next((c for c in clips if c.get("id") == clip_id), clips[0])

                    cut_with_clips.append((cut, selected_clip.get("file_path", ""), clip_id))
                    # FIX #12: Use set.add() instead of list.append()
                    used_clips_set.add(clip_id)

                # Clear FAISS index
                pacing_engine.faiss_matcher.clear_index()
            else:
                # Fallback: Old slow method
                logger.warning("FAISS not available for Structure-Awareness! Using slow method...")
                cut_with_clips = []
                for cut in pacing_cuts:
                    selected_clip = pacing_engine.clip_selector.select_clip(
                        clips, cut.strength, cut.trigger_type
                    )
                    cut_with_clips.append((cut, selected_clip.clip_path, selected_clip.clip_id))
        else:
            # Normal motion-matching with time-window optimization
            cut_with_clips_raw = pacing_engine.generate_cut_list_with_clips(
                audio_path=audio_path,
                available_clips=clips,
                expected_bpm=expected_bpm,
                min_cut_interval=min_cut_interval,  # Use GUI slider value
                start_time=start_time,
                end_time=end_time,
                phrase_alignment_mode=phrase_alignment_mode,
                progress_callback=progress_callback,
            )
            # Convert to consistent format
            cut_with_clips = [
                (cut, clip.get("file_path", ""), clip.get("id", "unknown"))
                for cut, clip in cut_with_clips_raw
            ]

        # Build clips_by_path lookup for duration info
        clips_by_path = {c.get("file_path", ""): c for c in clips}

        # ========================================================================
        # DYNAMIC DURATION SYSTEM: Variable Clip-Längen basierend auf Musik
        # Ersetzt das alte "fixed duration between triggers" System
        # ========================================================================

        # Get BPM from pacing engine
        bpm = 120.0  # Default fallback
        if hasattr(pacing_engine.trigger_system, "last_analysis"):
            analysis = pacing_engine.trigger_system.last_analysis
            if analysis and hasattr(analysis, "bpm"):
                bpm = analysis.bpm
                logger.info(f"Using detected BPM: {bpm:.1f}")

        # Initialize Dynamic Duration Calculator
        duration_calculator = DynamicDurationCalculator(
            bpm=bpm,
            constraints=DurationConstraints(
                min_duration=2.0,
                max_duration=16.0,
                phrase_beats=4,
                allow_full_clip=True,
                max_full_clip=20.0,
            ),
        )

        # Get energy curve for energy-based duration modulation
        energy_curve = pacing_engine.energy_curve

        # Convert to CutListEntries with DYNAMIC DURATIONS
        cut_list = []
        timeline_pos = 0.0  # Aktuelle Position in der Timeline

        for i in range(len(cut_with_clips) - 1):
            current_cut, clip_path, clip_id = cut_with_clips[i]
            next_cut, _, _ = cut_with_clips[i + 1]

            # Check duration limit
            if timeline_pos >= target_duration:
                break

            # Get absolute file path
            if not clip_path:
                continue

            file_path = str(resolve_relative_path(clip_path))

            # Get clip metadata
            clip_info = clips_by_path.get(clip_path, {})
            clip_total_duration = clip_info.get("duration", 10.0)

            # Get audio energy at cut time (for duration calculation)
            audio_energy = current_cut.strength  # Fallback
            if energy_curve is not None:
                try:
                    audio_energy = energy_curve.get_energy_at_time(current_cut.time)
                except (ValueError, IndexError):
                    pass  # Use fallback

            # Get segment type (for phrase-based duration)
            segment_type = None
            if hasattr(pacing_engine, "structure_result") and pacing_engine.structure_result:
                segment = pacing_engine.structure_analyzer.get_segment_at_time(
                    pacing_engine.structure_result, current_cut.time
                )
                if segment:
                    segment_type = segment.segment_type

            # Get match quality from clip dict (set by advanced_pacing_engine)
            match_quality = clip_info.get("match_quality", 0.5)

            # Calculate next trigger distance (for natural cut points)
            next_trigger_distance = next_cut.time - current_cut.time

            # CALCULATE OPTIMAL DURATION using DynamicDurationCalculator
            optimal_duration = duration_calculator.calculate_duration(
                audio_energy=audio_energy,
                trigger_strength=current_cut.strength,
                segment_type=segment_type,
                match_quality=match_quality,
                clip_duration=clip_total_duration,
                next_trigger_distance=next_trigger_distance,
            )

            # Calculate clip segment (startpoint in clip)
            clip_start, _ = calculate_intelligent_clip_segment(
                clip_duration=clip_total_duration,
                segment_duration=optimal_duration,
                trigger_strength=current_cut.strength,
                previous_clip_end=timeline_pos,
                clip_id=str(clip_id),
            )

            # Adjust times
            end_time = min(timeline_pos + optimal_duration, target_duration)

            if end_time <= timeline_pos:
                continue

            # Build metadata with DYNAMIC DURATION information
            metadata = {
                "file_path": file_path,
                "clip_name": Path(clip_path).stem,
                "clip_start": clip_start,
                "clip_duration": clip_total_duration,
                "trigger_type": current_cut.trigger_type,
                "trigger_strength": current_cut.strength,
                "calculated_duration": optimal_duration,  # NEW: Dynamic duration
                "segment_type": segment_type,
                "audio_energy": audio_energy,
                "match_quality": match_quality,
            }

            # Create cut entry with DYNAMIC DURATION
            cut = CutListEntry(
                clip_id=file_path, start_time=timeline_pos, end_time=end_time, metadata=metadata
            )

            cut_list.append(cut)
            timeline_pos = end_time  # Advance timeline

            # DEBUG: Log dynamic duration calculation
            logger.debug(
                f"Cut {i}: {Path(file_path).name} @ {timeline_pos:.1f}s-{end_time:.1f}s "
                f"(duration={optimal_duration:.1f}s, energy={audio_energy:.2f}, "
                f"segment={segment_type}, quality={match_quality:.2f})"
            )

        # ================================================================
        # FIX: Add final cut to cover remaining audio duration
        # ================================================================
        # Problem: The loop only processes cuts up to len-1 (skips last cut)
        # This leaves the audio after the last trigger uncovered!
        # Solution: Add one more cut from timeline_pos to target_duration
        if cut_with_clips and timeline_pos < target_duration:
            remaining_duration = target_duration - timeline_pos
            if remaining_duration >= 0.5:  # Only if meaningful duration remains
                # Use last clip for final segment (round-robin from remaining clips)
                last_cut, last_clip_path, last_clip_id = cut_with_clips[-1]
                clip_info = clips_by_path.get(last_clip_path, {})

                # Get a different clip for variety (round-robin)
                final_clip_idx = len(cut_list) % len(clips)
                final_clip = clips[final_clip_idx]
                final_clip_path = final_clip.get("file_path", last_clip_path)
                final_clip_path = (
                    str(resolve_relative_path(final_clip_path)) if final_clip_path else ""
                )
                final_clip_duration = final_clip.get("duration", 10.0)

                # Calculate segment for final cut
                clip_start, actual_duration = calculate_intelligent_clip_segment(
                    clip_duration=final_clip_duration,
                    segment_duration=remaining_duration,
                    trigger_strength=0.5,  # Medium strength for final
                    previous_clip_end=timeline_pos,
                    clip_id=str(final_clip.get("id", "final")),
                )

                # Clamp to remaining duration
                actual_duration = min(actual_duration, remaining_duration)
                end_time = timeline_pos + actual_duration

                if actual_duration >= 0.5:
                    final_metadata = {
                        "file_path": final_clip_path,
                        "clip_name": Path(final_clip_path).stem if final_clip_path else "final",
                        "clip_start": clip_start,
                        "clip_duration": final_clip_duration,
                        "trigger_type": "fill",  # Mark as fill cut
                        "trigger_strength": 0.5,
                    }

                    final_cut = CutListEntry(
                        clip_id=final_clip_path,
                        start_time=timeline_pos,
                        end_time=end_time,
                        metadata=final_metadata,
                    )
                    cut_list.append(final_cut)
                    logger.info(
                        f"✅ Added final fill cut: {timeline_pos:.1f}s - {end_time:.1f}s "
                        f"({actual_duration:.1f}s remaining audio covered)"
                    )

        # Log Statistiken über Clip-Längen-Variation
        if cut_list:
            durations = [c.end_time - c.start_time for c in cut_list]
            avg_dur = sum(durations) / len(durations)
            min_dur = min(durations)
            max_dur = max(durations)
            total_covered = sum(durations)
            coverage_percent = (total_covered / target_duration * 100) if target_duration > 0 else 0
            logger.info(
                f"Schnittliste mit Motion-Matching: {len(cut_list)} Schnitte "
                f"(Dauer: {min_dur:.1f}s - {max_dur:.1f}s, avg={avg_dur:.1f}s)"
            )
            logger.info(
                f"📊 Audio coverage: {total_covered:.1f}s / {target_duration:.1f}s "
                f"({coverage_percent:.1f}%)"
            )
        else:
            logger.info("Schnittliste mit Motion-Matching: 0 Schnitte")

        return cut_list

    def generate_simple_cuts(
        self,
        pacing_engine,
        audio_path: str,
        clips: list,
        expected_bpm: float | None,
        target_duration: float,
        duration_limit: float | None = None,
        min_cut_interval: float = 4.0,  # FIX: Added - controls minimum clip duration
        phrase_alignment_mode: bool = False,
        progress_callback: PacingProgressCallback | None = None,
    ) -> list[CutListEntry]:
        """
        Generate cut list using simple round-robin clip selection.

        Args:
            pacing_engine: Configured AdvancedPacingEngine
            audio_path: Path to audio file
            clips: Available video clips
            expected_bpm: Expected BPM
            target_duration: Target video duration
            duration_limit: Optional duration limit for time-window optimization
            min_cut_interval: Minimum time between cuts in seconds (default 4.0)
            progress_callback: Optional callback(current, total, message) für Fortschritt

        Returns:
            List of CutListEntry objects
        """
        # PERFORMANCE FIX: Use time-window optimization if duration_limit is set
        start_time = None
        end_time = None
        if duration_limit is not None:
            start_time = 0.0
            end_time = duration_limit
            logger.info(f"Using time-window optimization: 0.0s - {duration_limit}s")

        # Generate trigger-based cuts with time-window optimization
        pacing_cuts = pacing_engine.generate_cut_list(
            audio_path=audio_path,
            expected_bpm=expected_bpm,
            min_cut_interval=min_cut_interval,  # Use GUI slider value
            start_time=start_time,
            end_time=end_time,
            phrase_alignment_mode=phrase_alignment_mode,
            progress_callback=progress_callback,
        )

        # Log statistics
        stats = pacing_engine.get_trigger_statistics(pacing_cuts)
        logger.info(f"Trigger-Statistik: {stats}")

        # Convert to CutListEntries with round-robin and VARIABLE clip lengths
        cut_list = []
        clip_idx = 0
        timeline_pos = 0.0  # Aktuelle Position in der Timeline

        for i in range(len(pacing_cuts) - 1):
            current_cut = pacing_cuts[i]
            next_cut = pacing_cuts[i + 1]

            if timeline_pos >= target_duration:
                break

            segment_duration = next_cut.time - current_cut.time
            if segment_duration < 0.5:
                continue

            # Get next clip (round-robin)
            clip = clips[clip_idx % len(clips)]

            file_path = clip.get("file_path", "")
            if not file_path:
                logger.warning(f"Skipping clip {clip['id']}: No file path")
                clip_idx += 1
                continue

            file_path = str(resolve_relative_path(file_path))

            # NEU: Hole Clip-Dauer aus Metadaten
            clip_total_duration = clip.get("duration", 10.0)  # Default 10s

            # NEU: Berechne intelligenten Startpunkt und variable Dauer
            clip_start, actual_duration = calculate_intelligent_clip_segment(
                clip_duration=clip_total_duration,
                segment_duration=segment_duration,
                trigger_strength=current_cut.strength,
                previous_clip_end=timeline_pos,
                clip_id=str(clip.get("id", clip_idx)),
            )

            # Berechne neue End-Zeit basierend auf variabler Dauer
            end_time = timeline_pos + actual_duration

            # Begrenze auf target_duration
            if end_time > target_duration:
                actual_duration = target_duration - timeline_pos
                end_time = target_duration

            if actual_duration < 0.5:
                clip_idx += 1
                continue

            # Create cut entry mit variabler Dauer und intelligentem Startpunkt
            cut = CutListEntry(
                clip_id=file_path,
                start_time=timeline_pos,  # NEU: Timeline-Position
                end_time=end_time,
                metadata={
                    "file_path": file_path,
                    "clip_name": clip.get("name", "Unknown"),
                    "clip_start": clip_start,  # NEU: Nicht mehr immer 0.0!
                    "clip_duration": clip_total_duration,  # NEU: Gesamtdauer
                    "trigger_type": current_cut.trigger_type,
                    "trigger_strength": current_cut.strength,
                },
            )

            cut_list.append(cut)
            timeline_pos = end_time  # NEU: Nächste Position
            clip_idx += 1

        # ================================================================
        # FIX: Add final cut to cover remaining audio duration
        # ================================================================
        # Problem: The loop only processes cuts up to len-1 (skips last cut)
        # This leaves the audio after the last trigger uncovered!
        if pacing_cuts and timeline_pos < target_duration:
            remaining_duration = target_duration - timeline_pos
            if remaining_duration >= 0.5:  # Only if meaningful duration remains
                # Get next clip in round-robin
                final_clip = clips[clip_idx % len(clips)]
                final_clip_path = final_clip.get("file_path", "")
                if final_clip_path:
                    final_clip_path = str(resolve_relative_path(final_clip_path))
                    final_clip_duration = final_clip.get("duration", 10.0)

                    clip_start, actual_duration = calculate_intelligent_clip_segment(
                        clip_duration=final_clip_duration,
                        segment_duration=remaining_duration,
                        trigger_strength=0.5,
                        previous_clip_end=timeline_pos,
                        clip_id=str(final_clip.get("id", "final")),
                    )

                    actual_duration = min(actual_duration, remaining_duration)
                    end_time = timeline_pos + actual_duration

                    if actual_duration >= 0.5:
                        final_cut = CutListEntry(
                            clip_id=final_clip_path,
                            start_time=timeline_pos,
                            end_time=end_time,
                            metadata={
                                "file_path": final_clip_path,
                                "clip_name": final_clip.get("name", "final"),
                                "clip_start": clip_start,
                                "clip_duration": final_clip_duration,
                                "trigger_type": "fill",
                                "trigger_strength": 0.5,
                            },
                        )
                        cut_list.append(final_cut)
                        logger.info(
                            f"✅ Added final fill cut: {timeline_pos:.1f}s - {end_time:.1f}s "
                            f"({actual_duration:.1f}s remaining audio covered)"
                        )

        # Log Statistiken über Clip-Längen-Variation
        if cut_list:
            durations = [c.end_time - c.start_time for c in cut_list]
            avg_dur = sum(durations) / len(durations)
            min_dur = min(durations)
            max_dur = max(durations)
            total_covered = sum(durations)
            coverage_percent = (total_covered / target_duration * 100) if target_duration > 0 else 0
            logger.info(
                f"Schnittliste generiert: {len(cut_list)} Schnitte "
                f"(Dauer: {min_dur:.1f}s - {max_dur:.1f}s, avg={avg_dur:.1f}s)"
            )
            logger.info(
                f"📊 Audio coverage: {total_covered:.1f}s / {target_duration:.1f}s "
                f"({coverage_percent:.1f}%)"
            )
        else:
            logger.info("Schnittliste generiert: 0 Schnitte")

        return cut_list

    def generate_cut_list(
        self,
        duration_limit: float | None = None,
        progress_callback: PacingProgressCallback | None = None,
    ) -> list[CutListEntry]:
        """
        Generate cut list using advanced pacing engine with trigger-based cutting.

        Args:
            duration_limit: Maximum duration (None = use full audio)
            progress_callback: Optional callback(current, total, message) für Fortschritt

        Returns:
            List of cut entries
        """
        from ...pacing.advanced_pacing_engine import AdvancedPacingEngine

        # Get parameters from dashboard
        params = self.main_window.parameter_dashboard_widget.get_parameters()

        # Validate prerequisites
        prerequisites = self.validate_prerequisites()
        if not prerequisites:
            return []

        audio_path, clips = prerequisites

        # Create trigger settings
        trigger_settings = self.create_trigger_settings(params)

        # Check if stems are available (from stem analysis)
        use_stems = getattr(self.main_window, "stems_available", False)
        if use_stems:
            logger.info("🎵 Stem-basierte Trigger-Analyse aktiviert (präzisere Kick/Snare/HiHat)")

        # Create trigger system with stem support if available
        from ...pacing.trigger_system import TriggerSystem

        trigger_system = TriggerSystem(use_cache=True, use_stems=use_stems)

        # Create pacing engine with stem-aware trigger system
        pacing_engine = AdvancedPacingEngine(
            trigger_settings=trigger_settings, trigger_system=trigger_system
        )

        # Apply Visual Continuity Weight from GUI
        continuity_weight = params.get("continuity_weight", 0.4)
        pacing_engine.config.continuity_weight = continuity_weight
        logger.info(f"Visual Continuity Weight set to: {continuity_weight:.2f}")

        # Check feature flags
        # Phase 3: FAISS makes Motion-Matching fast! Now enabled by default.
        use_motion_matching = params.get("motion_matching_enabled", True)
        use_structure_awareness = params.get("structure_awareness_enabled", False)

        # Get expected BPM from parameters
        expected_bpm = params.get("bpm", None)

        # FIX: Get cut_duration from GUI slider and use as min_cut_interval
        # This controls how long clips play before cutting (reduces hectic feeling)
        cut_duration = params.get("cut_duration", 4.0)  # Default 4.0 seconds
        min_cut_interval = max(1.0, cut_duration)  # Minimum 1 second
        logger.info(
            f"🎬 Cut duration from GUI: {cut_duration}s → min_cut_interval={min_cut_interval}s"
        )

        # Determine target duration
        if duration_limit:
            target_duration = duration_limit
        else:
            target_duration = self.main_window.timeline_widget.duration or 120.0

        # DEBUG: Log duration_limit value
        logger.info(f"[DEBUG] duration_limit={duration_limit}, target_duration={target_duration}")

        logger.info(
            f"Generiere Schnittliste für {target_duration}s "
            f"(Motion-Matching={'aktiviert' if use_motion_matching else 'deaktiviert'}, "
            f"Struktur-Awareness={'aktiviert' if use_structure_awareness else 'deaktiviert'})"
        )

        # Enable structure awareness if requested
        if use_structure_awareness:
            pacing_engine.enable_structure_awareness(True)

        # Generate cut list using extracted methods
        try:
            if use_motion_matching:
                cut_list = self.generate_motion_matching_cuts(
                    pacing_engine,
                    audio_path,
                    clips,
                    expected_bpm,
                    target_duration,
                    use_structure_awareness,
                    duration_limit,
                    min_cut_interval=min_cut_interval,  # FIX: Pass GUI cut duration
                    phrase_alignment_mode=params.get("active_rules", {}).get(
                        "Phrase Alignment", False
                    ),
                    progress_callback=progress_callback,
                )
            else:
                cut_list = self.generate_simple_cuts(
                    pacing_engine,
                    audio_path,
                    clips,
                    expected_bpm,
                    target_duration,
                    duration_limit,
                    min_cut_interval=min_cut_interval,  # FIX: Pass GUI cut duration
                    phrase_alignment_mode=params.get("active_rules", {}).get(
                        "Phrase Alignment", False
                    ),
                    progress_callback=progress_callback,
                )

        except Exception as e:
            logger.error(f"Fehler bei Schnittlisten-Generierung: {e}", exc_info=True)
            QMessageBox.warning(
                self.main_window,
                "Pacing-Fehler",
                f"Fehler bei Schnittlisten-Generierung:\n{str(e)}",
            )
            return []

        return cut_list
