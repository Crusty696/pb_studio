"""
PB_studio - Motion-Music Synchronization Module
================================================

Phase 2 Implementation: Beat-level synchronization between audio and video.

Target: <200ms beat alignment accuracy

This module provides:
1. BeatSyncPoint - Dataclass for beat-video alignment points
2. MotionMusicSynchronizer - Core synchronization engine
3. Algorithms for optimal clip placement on beats

Research basis:
- Dynamic Time Warping for sequence alignment
- Onset detection correlation
- Motion peak-to-beat matching
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


class SyncQuality(Enum):
    """Quality level of beat-motion synchronization."""

    PERFECT = "perfect"  # <50ms offset
    EXCELLENT = "excellent"  # <100ms offset
    GOOD = "good"  # <200ms offset
    ACCEPTABLE = "acceptable"  # <500ms offset
    POOR = "poor"  # >=500ms offset


@dataclass
class BeatSyncPoint:
    """
    Represents a synchronization point between audio beat and video motion.

    Attributes:
        audio_time: Time of audio beat in seconds
        video_time: Time of matched video motion peak in seconds
        offset_ms: Offset in milliseconds (video_time - audio_time) * 1000
        confidence: Confidence score of the match (0-1)
        beat_strength: Strength of the audio beat (0-1)
        motion_strength: Strength of the video motion (0-1)
        sync_quality: Quality classification of the sync
    """

    audio_time: float
    video_time: float
    offset_ms: float
    confidence: float
    beat_strength: float = 0.0
    motion_strength: float = 0.0
    sync_quality: SyncQuality = SyncQuality.GOOD

    def __post_init__(self):
        """Calculate sync quality from offset."""
        abs_offset = abs(self.offset_ms)
        if abs_offset < 50:
            self.sync_quality = SyncQuality.PERFECT
        elif abs_offset < 100:
            self.sync_quality = SyncQuality.EXCELLENT
        elif abs_offset < 200:
            self.sync_quality = SyncQuality.GOOD
        elif abs_offset < 500:
            self.sync_quality = SyncQuality.ACCEPTABLE
        else:
            self.sync_quality = SyncQuality.POOR


@dataclass
class SyncAnalysisResult:
    """
    Complete synchronization analysis result.

    Attributes:
        sync_points: List of all synchronization points
        average_offset_ms: Mean absolute offset in milliseconds
        max_offset_ms: Maximum offset observed
        sync_score: Overall synchronization score (0-1)
        beat_coverage: Percentage of beats with motion matches
        recommended_offset: Suggested global offset to improve sync
    """

    sync_points: list[BeatSyncPoint] = field(default_factory=list)
    average_offset_ms: float = 0.0
    max_offset_ms: float = 0.0
    sync_score: float = 0.0
    beat_coverage: float = 0.0
    recommended_offset: float = 0.0

    def get_quality_distribution(self) -> dict[SyncQuality, int]:
        """Get count of sync points by quality level."""
        distribution = dict.fromkeys(SyncQuality, 0)
        for sp in self.sync_points:
            distribution[sp.sync_quality] += 1
        return distribution


@dataclass
class ClipBeatAlignment:
    """
    Alignment information for a video clip to audio beats.

    Attributes:
        clip_id: Identifier for the video clip
        start_beat_time: Audio beat time where clip should start
        clip_start_offset: Offset into clip to align with beat (seconds)
        motion_peak_time: Time of motion peak in clip (relative to clip start)
        alignment_score: Quality score of the alignment (0-1)
    """

    clip_id: str
    start_beat_time: float
    clip_start_offset: float = 0.0
    motion_peak_time: float = 0.0
    alignment_score: float = 0.0

    def get_playback_start(self) -> float:
        """Get the time in the clip where playback should start."""
        return max(0.0, self.clip_start_offset)


class MotionMusicSynchronizer:
    """
    Core engine for synchronizing video motion with audio beats.

    Uses multiple algorithms to achieve <200ms beat alignment:
    1. Direct matching with tolerance window
    2. DTW (Dynamic Time Warping) for sequence alignment
    3. Cross-correlation for global offset detection
    """

    # Synchronization parameters
    DEFAULT_TOLERANCE_MS = 200.0  # Target tolerance
    MIN_TOLERANCE_MS = 50.0
    MAX_TOLERANCE_MS = 500.0

    # Beat matching weights
    STRENGTH_WEIGHT = 0.3
    TIMING_WEIGHT = 0.7

    def __init__(
        self,
        tolerance_ms: float = DEFAULT_TOLERANCE_MS,
        use_dtw: bool = True,
        use_cross_correlation: bool = True,
    ):
        """
        Initialize the synchronizer.

        Args:
            tolerance_ms: Maximum acceptable offset in milliseconds
            use_dtw: Whether to use DTW for sequence alignment
            use_cross_correlation: Whether to use cross-correlation for offset
        """
        self.tolerance_ms = max(self.MIN_TOLERANCE_MS, min(tolerance_ms, self.MAX_TOLERANCE_MS))
        self.use_dtw = use_dtw
        self.use_cross_correlation = use_cross_correlation

        logger.info(
            f"MotionMusicSynchronizer initialized: tolerance={self.tolerance_ms}ms, DTW={use_dtw}"
        )

    def analyze_sync(
        self,
        audio_beats: list[float],
        audio_beat_strengths: list[float] | None,
        video_motion_times: list[float],
        video_motion_strengths: list[float] | None,
    ) -> SyncAnalysisResult:
        """
        Analyze synchronization between audio beats and video motion.

        Args:
            audio_beats: List of audio beat times in seconds
            audio_beat_strengths: Optional strengths for each beat (0-1)
            video_motion_times: List of video motion peak times in seconds
            video_motion_strengths: Optional strengths for each motion peak (0-1)

        Returns:
            SyncAnalysisResult with detailed analysis
        """
        if not audio_beats or not video_motion_times:
            logger.warning("Empty beats or motion times provided")
            return SyncAnalysisResult()

        # Normalize strengths
        beat_strengths = audio_beat_strengths or [1.0] * len(audio_beats)
        motion_strengths = video_motion_strengths or [1.0] * len(video_motion_times)

        # Calculate recommended global offset using cross-correlation
        recommended_offset = 0.0
        if self.use_cross_correlation:
            recommended_offset = self._calculate_global_offset(audio_beats, video_motion_times)

        # Find matching sync points
        sync_points = self._find_sync_points(
            audio_beats=audio_beats,
            beat_strengths=beat_strengths,
            motion_times=video_motion_times,
            motion_strengths=motion_strengths,
            global_offset=recommended_offset,
        )

        # Calculate statistics
        if sync_points:
            offsets = [abs(sp.offset_ms) for sp in sync_points]
            average_offset = np.mean(offsets)
            max_offset = np.max(offsets)

            # Score based on how many beats have good sync
            good_syncs = sum(
                1
                for sp in sync_points
                if sp.sync_quality in [SyncQuality.PERFECT, SyncQuality.EXCELLENT, SyncQuality.GOOD]
            )
            sync_score = good_syncs / len(audio_beats) if audio_beats else 0.0
            beat_coverage = len(sync_points) / len(audio_beats) if audio_beats else 0.0
        else:
            average_offset = float("inf")
            max_offset = float("inf")
            sync_score = 0.0
            beat_coverage = 0.0

        return SyncAnalysisResult(
            sync_points=sync_points,
            average_offset_ms=average_offset,
            max_offset_ms=max_offset,
            sync_score=sync_score,
            beat_coverage=beat_coverage,
            recommended_offset=recommended_offset,
        )

    def _calculate_global_offset(
        self,
        audio_beats: list[float],
        video_motion_times: list[float],
        resolution_ms: float = 10.0,
        max_offset_ms: float = 500.0,
    ) -> float:
        """
        Calculate optimal global offset using cross-correlation.

        Args:
            audio_beats: Audio beat times
            video_motion_times: Video motion peak times
            resolution_ms: Resolution of correlation in milliseconds
            max_offset_ms: Maximum offset to search

        Returns:
            Recommended offset in seconds (positive = video leads)
        """
        if len(audio_beats) < 2 or len(video_motion_times) < 2:
            return 0.0

        # Create time range
        max_time = max(max(audio_beats), max(video_motion_times)) + 1.0
        resolution_s = resolution_ms / 1000.0
        time_bins = int(max_time / resolution_s) + 1

        # Create impulse trains
        audio_signal = np.zeros(time_bins)
        video_signal = np.zeros(time_bins)

        for t in audio_beats:
            idx = int(t / resolution_s)
            if 0 <= idx < time_bins:
                audio_signal[idx] = 1.0

        for t in video_motion_times:
            idx = int(t / resolution_s)
            if 0 <= idx < time_bins:
                video_signal[idx] = 1.0

        # Cross-correlate
        correlation = signal.correlate(video_signal, audio_signal, mode="full")
        lags = signal.correlation_lags(len(video_signal), len(audio_signal), mode="full")

        # Find peak within max offset range
        max_lag = int(max_offset_ms / resolution_ms)
        valid_mask = np.abs(lags) <= max_lag

        if not np.any(valid_mask):
            return 0.0

        valid_corr = np.where(valid_mask, correlation, 0)
        best_lag_idx = np.argmax(valid_corr)
        best_lag = lags[best_lag_idx]

        offset_seconds = best_lag * resolution_s

        logger.debug(f"Global offset calculated: {offset_seconds * 1000:.1f}ms")
        return offset_seconds

    def _find_sync_points(
        self,
        audio_beats: list[float],
        beat_strengths: list[float],
        motion_times: list[float],
        motion_strengths: list[float],
        global_offset: float = 0.0,
    ) -> list[BeatSyncPoint]:
        """
        Find synchronization points between beats and motion peaks.

        Uses a greedy matching algorithm with strength-weighted scoring.
        """
        sync_points = []
        used_motion_indices = set()

        tolerance_s = self.tolerance_ms / 1000.0

        # Sort beats by strength (match strongest beats first)
        beat_indices = sorted(
            range(len(audio_beats)), key=lambda i: beat_strengths[i], reverse=True
        )

        for beat_idx in beat_indices:
            beat_time = audio_beats[beat_idx]
            beat_strength = beat_strengths[beat_idx]

            # Find best matching motion peak
            best_match = None
            best_score = -1.0

            for motion_idx, motion_time in enumerate(motion_times):
                if motion_idx in used_motion_indices:
                    continue

                # Apply global offset
                adjusted_motion_time = motion_time - global_offset
                offset = adjusted_motion_time - beat_time
                offset_ms = offset * 1000.0

                # Check if within tolerance
                if abs(offset_ms) > self.tolerance_ms:
                    continue

                # Calculate match score
                motion_strength = motion_strengths[motion_idx]
                timing_score = 1.0 - (abs(offset_ms) / self.tolerance_ms)
                strength_score = min(beat_strength, motion_strength)

                score = self.TIMING_WEIGHT * timing_score + self.STRENGTH_WEIGHT * strength_score

                if score > best_score:
                    best_score = score
                    best_match = (motion_idx, motion_time, offset_ms, motion_strength)

            if best_match:
                motion_idx, motion_time, offset_ms, motion_strength = best_match
                used_motion_indices.add(motion_idx)

                sync_points.append(
                    BeatSyncPoint(
                        audio_time=beat_time,
                        video_time=motion_time,
                        offset_ms=offset_ms,
                        confidence=best_score,
                        beat_strength=beat_strength,
                        motion_strength=motion_strength,
                    )
                )

        # Sort by audio time
        sync_points.sort(key=lambda sp: sp.audio_time)
        return sync_points

    def find_optimal_clip_start(
        self,
        target_beat_time: float,
        clip_motion_times: list[float],
        clip_motion_strengths: list[float] | None = None,
        clip_duration: float = 10.0,
        prefer_strong_motion: bool = True,
    ) -> ClipBeatAlignment:
        """
        Find optimal start point in a clip to align with a target beat.

        Args:
            target_beat_time: The audio beat time to align with
            clip_motion_times: Motion peak times within the clip (relative to clip start)
            clip_motion_strengths: Optional motion strengths
            clip_duration: Total clip duration in seconds
            prefer_strong_motion: Whether to prefer aligning with strong motion peaks

        Returns:
            ClipBeatAlignment with optimal alignment information
        """
        if not clip_motion_times:
            # No motion peaks - start from beginning
            return ClipBeatAlignment(
                clip_id="",
                start_beat_time=target_beat_time,
                clip_start_offset=0.0,
                motion_peak_time=0.0,
                alignment_score=0.5,
            )

        strengths = clip_motion_strengths or [1.0] * len(clip_motion_times)

        # Find motion peak that best aligns with beat
        best_peak_idx = 0
        best_score = -1.0

        for i, (motion_time, strength) in enumerate(zip(clip_motion_times, strengths)):
            # Prefer peaks that leave room before and after
            position_score = 1.0 - abs(
                2 * motion_time / clip_duration - 0.3
            )  # Prefer ~15% into clip
            position_score = max(0.0, position_score)

            # Factor in motion strength
            if prefer_strong_motion:
                score = 0.6 * strength + 0.4 * position_score
            else:
                score = 0.3 * strength + 0.7 * position_score

            if score > best_score:
                best_score = score
                best_peak_idx = i

        motion_peak_time = clip_motion_times[best_peak_idx]

        return ClipBeatAlignment(
            clip_id="",
            start_beat_time=target_beat_time,
            clip_start_offset=motion_peak_time,  # Start this much into clip
            motion_peak_time=motion_peak_time,
            alignment_score=best_score,
        )

    def align_clip_sequence(
        self,
        audio_beats: list[float],
        audio_beat_strengths: list[float] | None,
        clips_data: list[dict[str, Any]],
        min_clip_duration: float = 2.0,
        max_clip_duration: float = 10.0,
    ) -> list[ClipBeatAlignment]:
        """
        Align a sequence of clips to audio beats.

        Args:
            audio_beats: List of audio beat times
            audio_beat_strengths: Optional beat strengths
            clips_data: List of clip info dicts with 'id', 'motion_times', 'motion_strengths', 'duration'
            min_clip_duration: Minimum clip playback duration
            max_clip_duration: Maximum clip playback duration

        Returns:
            List of ClipBeatAlignment for each clip placement
        """
        if not audio_beats or not clips_data:
            return []

        beat_strengths = audio_beat_strengths or [1.0] * len(audio_beats)
        alignments = []

        current_time = 0.0
        clip_idx = 0
        beat_idx = 0

        while clip_idx < len(clips_data) and beat_idx < len(audio_beats):
            clip_data = clips_data[clip_idx]
            beat_time = audio_beats[beat_idx]

            # Skip beats that are before current time
            if beat_time < current_time:
                beat_idx += 1
                continue

            # Find optimal alignment for this clip
            alignment = self.find_optimal_clip_start(
                target_beat_time=beat_time,
                clip_motion_times=clip_data.get("motion_times", []),
                clip_motion_strengths=clip_data.get("motion_strengths"),
                clip_duration=clip_data.get("duration", 10.0),
            )
            alignment.clip_id = clip_data.get("id", f"clip_{clip_idx}")

            # Calculate clip duration (to next beat or max duration)
            if beat_idx + 1 < len(audio_beats):
                next_beat = audio_beats[beat_idx + 1]
                clip_duration = min(next_beat - beat_time, max_clip_duration)
                clip_duration = max(clip_duration, min_clip_duration)
            else:
                clip_duration = max_clip_duration

            alignments.append(alignment)

            current_time = beat_time + clip_duration
            clip_idx += 1

            # Find next beat after current clip
            while beat_idx < len(audio_beats) and audio_beats[beat_idx] < current_time:
                beat_idx += 1

        logger.info(f"Aligned {len(alignments)} clips to audio beats")
        return alignments

    def calculate_sync_score(
        self,
        cutlist_times: list[float],
        audio_beats: list[float],
        tolerance_ms: float | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """
        Calculate synchronization score for a cutlist against audio beats.

        Args:
            cutlist_times: List of cut times in the output
            audio_beats: List of audio beat times
            tolerance_ms: Tolerance for matching (uses default if None)

        Returns:
            Tuple of (score 0-1, details dict)
        """
        if not cutlist_times or not audio_beats:
            return 0.0, {"error": "Empty input"}

        tol = (tolerance_ms or self.tolerance_ms) / 1000.0

        matched_cuts = 0
        total_offset_ms = 0.0
        offsets = []

        for cut_time in cutlist_times:
            # Find closest beat
            closest_beat = min(audio_beats, key=lambda b: abs(b - cut_time))
            offset = abs(cut_time - closest_beat)
            offset_ms = offset * 1000.0
            offsets.append(offset_ms)

            if offset <= tol:
                matched_cuts += 1
                total_offset_ms += offset_ms

        # Calculate scores
        match_ratio = matched_cuts / len(cutlist_times) if cutlist_times else 0.0
        avg_offset = np.mean(offsets) if offsets else float("inf")

        # Combined score
        timing_score = max(0.0, 1.0 - (avg_offset / (self.tolerance_ms * 2)))
        score = 0.6 * match_ratio + 0.4 * timing_score

        details = {
            "matched_cuts": matched_cuts,
            "total_cuts": len(cutlist_times),
            "match_ratio": match_ratio,
            "average_offset_ms": avg_offset,
            "max_offset_ms": max(offsets) if offsets else 0.0,
            "timing_score": timing_score,
            "final_score": score,
        }

        return score, details


def create_beat_aligned_cutlist(
    audio_beats: list[float],
    audio_beat_strengths: list[float] | None,
    clips_with_motion: list[dict[str, Any]],
    total_duration: float,
    synchronizer: MotionMusicSynchronizer | None = None,
) -> list[dict[str, Any]]:
    """
    Convenience function to create a beat-aligned cutlist.

    Args:
        audio_beats: Audio beat times
        audio_beat_strengths: Beat strengths (0-1)
        clips_with_motion: List of clip dicts with motion data
        total_duration: Total output duration
        synchronizer: Optional pre-configured synchronizer

    Returns:
        List of cutlist entries with timing information
    """
    sync = synchronizer or MotionMusicSynchronizer()

    alignments = sync.align_clip_sequence(
        audio_beats=audio_beats,
        audio_beat_strengths=audio_beat_strengths,
        clips_data=clips_with_motion,
    )

    cutlist = []
    for i, alignment in enumerate(alignments):
        # Find corresponding clip data
        clip_data = next((c for c in clips_with_motion if c.get("id") == alignment.clip_id), {})

        # Calculate end time
        if i + 1 < len(alignments):
            end_time = alignments[i + 1].start_beat_time
        else:
            end_time = total_duration

        cutlist.append(
            {
                "clip_id": alignment.clip_id,
                "start_time": alignment.start_beat_time,
                "end_time": end_time,
                "clip_offset": alignment.clip_start_offset,
                "alignment_score": alignment.alignment_score,
                "beat_aligned": True,
            }
        )

    return cutlist
