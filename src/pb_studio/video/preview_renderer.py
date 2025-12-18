"""
Preview Renderer for PB_studio

Generates 90-second preview videos with selectable start position.

Features:
- Generate preview from any position in the beatgrid
- 90-second duration (configurable)
- Uses same rendering pipeline as full video
- Fast preview generation
- Position selection by time or beat number
- Compressed audio for smaller preview files (AAC/MP3)
- Configurable audio quality presets

Dependencies:
- VideoRenderer (existing)
- AdvancedPacingEngine (existing)

Audio Quality Presets:
- 'low': MP3 96k (smallest, good for quick previews)
- 'medium': AAC 128k (default, balanced quality/size)
- 'high': AAC 192k (high quality, larger files)
- 'lossless': WAV (uncompressed, for final export)

Usage:
    # Default quality (medium, AAC 128k)
    renderer = PreviewRenderer()

    # High quality previews
    renderer = PreviewRenderer(audio_quality='high')

    # Generate preview starting at 60 seconds
    preview_path = renderer.generate_preview(
        audio_path="audio.mp3",
        clips=clip_list,
        start_position=60.0,  # Start at 60 seconds
        duration=90.0,  # 90 second preview
        output_path="preview.mp4"
    )

    # Generate preview starting at beat 128
    preview_path = renderer.generate_preview_at_beat(
        audio_path="audio.mp3",
        clips=clip_list,
        beat_number=128,
        duration=90.0,
        output_path="preview.mp4"
    )

Audio Compression Impact:
- WAV 90s @ 44.1kHz: ~15.4 MB (unkomprimiert)
- AAC 90s @ 128k: ~1.4 MB (ca. 91% kleiner)
- MP3 90s @ 96k: ~1.05 MB (ca. 93% kleiner)
"""

import logging
import secrets
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from pb_studio.pacing.advanced_pacing_engine import AdvancedPacingEngine
from pb_studio.pacing.dynamic_duration import DurationConstraints, DynamicDurationCalculator
from pb_studio.pacing.intensity_controller import TriggerIntensitySettings
from pb_studio.pacing.pacing_models import CutListEntry
from pb_studio.utils.path_utils import get_ffmpeg_path, validate_ffmpeg_path
from pb_studio.video.video_renderer import VideoRenderer

logger = logging.getLogger(__name__)

# Audio format configuration for previews
DEFAULT_PREVIEW_AUDIO_FORMAT = "aac"  # Compressed format for previews
DEFAULT_PREVIEW_AUDIO_BITRATE = "128k"  # Reasonable quality/size balance

# Quality presets for different use cases
PREVIEW_QUALITY_PRESETS = {
    "low": {"format": "mp3", "bitrate": "96k"},
    "medium": {"format": "aac", "bitrate": "128k"},
    "high": {"format": "aac", "bitrate": "192k"},
    "lossless": {"format": "wav", "bitrate": None},  # For final export
}


class PreviewRenderer:
    """
    Generates preview videos with selectable start position.

    Creates 90-second (or custom duration) preview videos starting from
    any position in the audio/beatgrid.
    """

    def __init__(self, audio_quality: str = "medium"):
        """
        Initialize PreviewRenderer.

        Args:
            audio_quality: Audio quality preset ('low', 'medium', 'high', 'lossless')
                          Default: 'medium' (AAC 128k)
        """
        self.renderer = VideoRenderer()
        self.audio_quality = audio_quality
        if audio_quality not in PREVIEW_QUALITY_PRESETS:
            logger.warning(
                f"Invalid audio quality '{audio_quality}', using 'medium'. "
                f"Valid options: {list(PREVIEW_QUALITY_PRESETS.keys())}"
            )
            self.audio_quality = "medium"
        logger.debug(
            f"PreviewRenderer initialized with audio quality: {self.audio_quality} "
            f"({PREVIEW_QUALITY_PRESETS[self.audio_quality]})"
        )

    def generate_preview(
        self,
        audio_path: str | Path,
        clips: list[dict],
        start_position: float = 0.0,
        duration: float = 90.0,
        output_path: str | Path = "preview.mp4",
        trigger_settings: TriggerIntensitySettings | None = None,
        expected_bpm: float | None = None,
        progress_callback: Callable[[float], None] | None = None,
        use_stems: bool = False,
    ) -> Path | None:
        """
        Generate preview video starting at specific time position.

        Args:
            audio_path: Path to audio file
            clips: List of available video clips
            start_position: Start time in seconds
            duration: Preview duration in seconds (default: 90.0)
            output_path: Output path for preview video
            trigger_settings: Optional trigger settings for pacing
            expected_bpm: Optional expected BPM
            progress_callback: Optional progress callback
            use_stems: Use stem-based trigger analysis if available

        Returns:
            Path to generated preview video or None on error
        """
        audio_path = Path(audio_path)
        output_path = Path(output_path)

        if not audio_path.exists():
            logger.error(f"Audio file not found: {audio_path}")
            return None

        if not clips:
            logger.error("No video clips provided for preview")
            return None

        # CRITICAL FIX: Initialize audio_segment_path and success flag outside try
        audio_segment_path = None
        preview_success = False

        try:
            logger.info(f"Generating preview: start={start_position}s, duration={duration}s")
            if progress_callback:
                progress_callback(0.01)  # Started

            # Create pacing engine
            if trigger_settings is None:
                trigger_settings = TriggerIntensitySettings()
                # Default: beat + energy triggers enabled
                trigger_settings.set_config("beat", enabled=True, intensity=80, threshold=30)
                trigger_settings.set_config("energy", enabled=True, intensity=70, threshold=30)

            if progress_callback:
                progress_callback(0.05)  # Engine config done

            # Create trigger system with stem support if requested
            from pb_studio.pacing.trigger_system import TriggerSystem

            trigger_system = TriggerSystem(use_cache=True, use_stems=use_stems)
            if use_stems:
                logger.info("🎵 Preview mit Stem-basierter Trigger-Analyse")

            pacing_engine = AdvancedPacingEngine(
                trigger_settings=trigger_settings, trigger_system=trigger_system
            )
            pacing_engine.enable_motion_matching(True)
            # PERFORMANCE FIX: Enable FAISS for 100-1000x faster clip selection
            # Auto-detect: Nutzt GPU falls CUDA verfügbar, sonst optimierter CPU-Modus
            pacing_engine.enable_faiss_matching(enabled=True, use_gpu=True)

            # Generate cut list for preview duration only
            logger.info("Generating cut list for preview segment")
            if progress_callback:
                progress_callback(0.10)  # Starting analysis

            # PERFORMANCE FIX: Only analyze preview time window (not entire audio)
            # This reduces trigger evaluation from 80,000+ to ~1,000 triggers
            end_position = start_position + duration
            cut_with_clips = pacing_engine.generate_cut_list_with_clips(
                audio_path=str(audio_path),
                available_clips=clips,
                expected_bpm=expected_bpm,
                min_cut_interval=0.5,
                start_time=start_position,
                end_time=end_position,
            )

            if not cut_with_clips:
                logger.error("Failed to generate cut list for preview")
                return None

            if progress_callback:
                progress_callback(0.15)  # Cut list ready

            # ========================================================================
            # DYNAMIC DURATION SYSTEM: Variable Clip-Längen
            # Ersetzt das alte "fixed duration between triggers" System
            # ========================================================================

            # Get BPM from pacing engine (needed for DynamicDurationCalculator)
            bpm = 120.0  # Default fallback
            if hasattr(pacing_engine.trigger_system, "last_analysis"):
                analysis = pacing_engine.trigger_system.last_analysis
                if analysis and hasattr(analysis, "bpm"):
                    bpm = analysis.bpm

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

            # Convert cuts to CutListEntries with DYNAMIC DURATIONS
            preview_cuts: list[CutListEntry] = []
            current_time = 0.0  # Track current position in timeline

            for i, (current_cut, clip) in enumerate(cut_with_clips):
                # Get clip metadata
                clip_path = clip.get("file_path", "")
                if not clip_path:
                    logger.warning(f"Clip has no file_path: {clip}")
                    continue

                file_path = str(Path(clip_path).absolute())

                # Get audio energy at cut time (for duration calculation)
                audio_energy = current_cut.strength  # Fallback
                if energy_curve is not None:
                    try:
                        # Get REAL audio energy (not just trigger strength)
                        audio_energy = energy_curve.get_energy_at_time(current_cut.time)
                    except (ValueError, IndexError):
                        logger.debug("Audio energy lookup failed, using trigger strength fallback")

                # Get segment type (for phrase-based duration)
                segment_type = None
                if hasattr(pacing_engine, "structure_result") and pacing_engine.structure_result:
                    segment = pacing_engine.structure_analyzer.get_segment_at_time(
                        pacing_engine.structure_result, current_cut.time
                    )
                    if segment:
                        segment_type = segment.segment_type

                # Get clip duration (for full-clip allowance)
                clip_duration = clip.get("duration", None)

                # CRITICAL FIX: Get match quality from FAISS (stored in clip dict by advanced_pacing_engine)
                # clip is a dict, not an object - use .get() directly
                match_quality = clip.get("match_quality", 0.5)  # Default 0.5 if not set

                # Calculate next trigger distance (for natural cut points)
                next_trigger_distance = None
                if i + 1 < len(cut_with_clips):
                    next_cut, _ = cut_with_clips[i + 1]
                    next_trigger_distance = next_cut.time - current_cut.time

                # CALCULATE OPTIMAL DURATION using DynamicDurationCalculator
                optimal_duration = duration_calculator.calculate_duration(
                    audio_energy=audio_energy,
                    trigger_strength=current_cut.strength,
                    segment_type=segment_type,
                    match_quality=match_quality,
                    clip_duration=clip_duration,
                    next_trigger_distance=next_trigger_distance,
                )

                # Adjust times relative to preview start
                adjusted_start = current_time
                adjusted_end = min(current_time + optimal_duration, duration)

                if adjusted_end <= adjusted_start:
                    continue

                # Advance timeline
                current_time = adjusted_end

                # Stop if we've reached preview duration
                if current_time >= duration:
                    # Create final cut and break
                    cut_entry = CutListEntry(
                        clip_id=file_path,
                        start_time=adjusted_start,
                        end_time=duration,
                        metadata={
                            "file_path": file_path,
                            "clip_name": clip.get("name", "Unknown"),
                            "clip_start": 0.0,
                            "trigger_type": current_cut.trigger_type,
                            "trigger_strength": current_cut.strength,
                            "calculated_duration": duration - adjusted_start,
                            "segment_type": segment_type,
                            "audio_energy": audio_energy,
                        },
                    )
                    preview_cuts.append(cut_entry)
                    break

                # Create cut entry with DYNAMIC DURATION
                cut_entry = CutListEntry(
                    clip_id=file_path,  # ✅ File path, not ID!
                    start_time=adjusted_start,
                    end_time=adjusted_end,
                    metadata={
                        "file_path": file_path,
                        "clip_name": clip.get("name", "Unknown"),
                        "clip_start": 0.0,
                        "trigger_type": current_cut.trigger_type,
                        "trigger_strength": current_cut.strength,
                        "calculated_duration": optimal_duration,
                        "segment_type": segment_type,
                        "audio_energy": audio_energy,
                        "match_quality": match_quality,
                    },
                )

                preview_cuts.append(cut_entry)

                # DEBUG: Log dynamic duration calculation
                logger.debug(
                    f"Cut {i}: {Path(file_path).name} @ {adjusted_start:.1f}s-{adjusted_end:.1f}s "
                    f"(duration={optimal_duration:.1f}s, energy={audio_energy:.2f}, "
                    f"segment={segment_type}, quality={match_quality:.2f})"
                )

            if not preview_cuts:
                logger.error(
                    f"No cuts found in preview window " f"({start_position}s - {end_position}s)"
                )
                return None

            logger.info(f"Preview cut list: {len(preview_cuts)} cuts")

            # Extract audio segment for preview with configured quality
            preset = PREVIEW_QUALITY_PRESETS[self.audio_quality]

            if progress_callback:
                progress_callback(0.18)  # Audio extract start

            audio_segment_path = self._extract_audio_segment(
                audio_path,
                start_position,
                duration,
                format=preset["format"],
                is_final_export=False,  # Preview mode: use compression
            )

            if audio_segment_path is None:
                logger.error("Failed to extract audio segment for preview")
                return None

            if progress_callback:
                progress_callback(0.20)  # Audio ready, start render

            # Wrapper for render progress (maps 0.0-1.0 to 0.2-1.0)
            def render_progress_wrapper(p: float):
                if progress_callback:
                    # Scale render progress to remaining 80%
                    scaled_progress = 0.20 + (p * 0.80)
                    progress_callback(scaled_progress)

            # Render preview video
            logger.info("Rendering preview video")
            result = self.renderer.render_video(
                cut_list=preview_cuts,
                audio_path=str(audio_segment_path),
                output_path=str(output_path),
                progress_callback=render_progress_wrapper,
            )

            if result:
                logger.info(f"Preview generated successfully: {output_path}")
                preview_success = True
                return Path(result)
            else:
                logger.error("Preview rendering failed")
                return None

        except Exception as e:
            logger.error(f"Error generating preview: {e}", exc_info=True)
            return None

        finally:
            # CRITICAL FIX: Cleanup temp audio AND output file on failure
            if audio_segment_path and audio_segment_path.exists():
                try:
                    audio_segment_path.unlink()
                    logger.debug(f"Cleaned up audio segment: {audio_segment_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup audio segment: {e}")

            if not preview_success and output_path.exists():
                try:
                    output_path.unlink()
                    logger.warning(f"Cleaned up corrupted/partial preview file: {output_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup output file: {e}")

    def generate_preview_at_beat(
        self,
        audio_path: str | Path,
        clips: list[dict],
        beat_number: int,
        duration: float = 90.0,
        output_path: str | Path = "preview.mp4",
        trigger_settings: TriggerIntensitySettings | None = None,
        expected_bpm: float | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> Path | None:
        """
        Generate preview video starting at specific beat number.

        Args:
            audio_path: Path to audio file
            clips: List of available video clips
            beat_number: Beat number to start at (1-based)
            duration: Preview duration in seconds (default: 90.0)
            output_path: Output path for preview video
            trigger_settings: Optional trigger settings
            expected_bpm: Optional expected BPM
            progress_callback: Optional progress callback

        Returns:
            Path to generated preview video or None on error
        """
        audio_path = Path(audio_path)

        try:
            logger.info(f"Converting beat {beat_number} to time position")

            # Analyze audio to get BPM
            from pb_studio.analysis.audio_analyzer import AudioAnalyzer

            analyzer = AudioAnalyzer()

            analysis = analyzer.analyze_audio(str(audio_path))
            if analysis is None:
                logger.error("Failed to analyze audio for beat conversion")
                return None

            bpm = analysis.get("tempo_bpm", expected_bpm)
            if bpm is None:
                logger.error("Could not determine BPM for beat conversion")
                return None

            # Convert beat number to time (assuming 4/4 time signature)
            beat_duration = 60.0 / bpm  # Duration of one beat in seconds
            start_position = (beat_number - 1) * beat_duration  # -1 because beat 1 = 0s

            logger.info(f"Beat {beat_number} at BPM {bpm:.1f} = {start_position:.2f}s")

            # Generate preview at calculated position
            return self.generate_preview(
                audio_path=audio_path,
                clips=clips,
                start_position=start_position,
                duration=duration,
                output_path=output_path,
                trigger_settings=trigger_settings,
                expected_bpm=bpm,
                progress_callback=progress_callback,
            )

        except Exception as e:
            logger.error(f"Error generating preview at beat: {e}", exc_info=True)
            return None

    def _extract_audio_segment(
        self,
        audio_path: Path,
        start_time: float,
        duration: float,
        format: str | None = None,
        is_final_export: bool = False,
    ) -> Path | None:
        """
        Extract audio segment using ffmpeg with configurable compression.

        Args:
            audio_path: Source audio file
            start_time: Start time in seconds
            duration: Duration in seconds
            format: Audio format ('aac', 'mp3', 'wav'). Default: from quality preset
            is_final_export: If True, use lossless WAV. Overrides format parameter.

        Returns:
            Path to extracted segment or None on error
        """

        try:
            # Determine output format and bitrate
            if is_final_export:
                # Final export: always lossless
                output_format = "wav"
                bitrate = None
            else:
                # Preview: use quality preset
                preset = PREVIEW_QUALITY_PRESETS[self.audio_quality]
                output_format = format or preset["format"]
                bitrate = preset["bitrate"]

            # Create temp file for audio segment with random suffix
            # SEC-07 FIX: Use cryptographic random suffix for unpredictable file names
            temp_dir = Path("output")
            temp_dir.mkdir(exist_ok=True, mode=0o755)
            random_suffix = secrets.token_hex(8)  # 16 hex chars

            # Determine file extension based on format
            file_ext = f".{output_format}"
            segment_path = temp_dir / f"preview_audio_{audio_path.stem}_{random_suffix}{file_ext}"

            # SECURITY: Validate paths for FFmpeg (Command Injection Protection)
            try:
                validated_audio = validate_ffmpeg_path(audio_path)
                validated_segment = validate_ffmpeg_path(segment_path)
            except ValueError as e:
                logger.error(f"FFmpeg path validation failed: {e}")
                return None

            # Build FFmpeg command based on format
            cmd = [
                str(get_ffmpeg_path()),
                "-y",  # Overwrite output
                "-ss",
                str(start_time),  # Start position
                "-t",
                str(duration),  # Duration
                "-i",
                str(validated_audio),  # Input (validated)
            ]

            if output_format == "aac":
                # AAC compression (good quality, wide compatibility)
                cmd.extend(
                    [
                        "-c:a",
                        "aac",
                        "-b:a",
                        bitrate,
                        "-movflags",
                        "+faststart",  # Optimize for streaming
                    ]
                )
            elif output_format == "mp3":
                # MP3 compression (maximum compatibility)
                cmd.extend(
                    [
                        "-c:a",
                        "libmp3lame",
                        "-b:a",
                        bitrate,
                    ]
                )
            elif output_format == "wav":
                # Lossless WAV (for final export)
                cmd.extend(
                    [
                        "-c:a",
                        "pcm_s16le",  # 16-bit PCM
                    ]
                )
            else:
                logger.error(f"Unsupported audio format: {output_format}")
                return None

            cmd.append(str(validated_segment))  # Output (validated)

            logger.debug(f"Extracting audio segment ({output_format}): {' '.join(cmd)}")

            # WINDOWS FIX: Hide console window
            extra_kwargs = {}
            if sys.platform == "win32":
                extra_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, **extra_kwargs
            )

            if result.returncode != 0:
                logger.error(f"ffmpeg failed: {result.stderr.decode()}")
                return None

            if not segment_path.exists():
                logger.error("Audio segment was not created")
                return None

            # Log file size for compression feedback
            file_size_mb = segment_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"Audio segment extracted: {segment_path.name} "
                f"({output_format}, {file_size_mb:.2f}MB)"
            )
            return segment_path

        except subprocess.TimeoutExpired:
            logger.error("Audio extraction timed out")
            return None
        except Exception as e:
            logger.error(f"Error extracting audio segment: {e}", exc_info=True)
            return None
