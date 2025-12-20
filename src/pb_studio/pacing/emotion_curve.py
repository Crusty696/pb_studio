"""
PB_studio - Emotion Curve Generator
====================================

Phase 3B: Emotion-based Selection Implementation

Generates emotion curves (valence/arousal) over the video timeline
for intelligent clip selection based on mood matching.

Features:
1. Timeline-based emotion tracking
2. Smooth emotion transitions
3. Segment-aware emotion mapping
4. Clip-emotion matching scores

Author: PB_studio
Created: 2025-12-05
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..utils.logger import get_logger
from .audio_visual_mapper import MOOD_EMOTION_MAP, AudioVisualMapper, EmotionCoordinates, MoodType

logger = get_logger(__name__)


@dataclass
class EmotionPoint:
    """Single point on the emotion curve."""

    time: float  # Time in seconds
    valence: float  # -1 to 1
    arousal: float  # -1 to 1
    segment_type: str | None = None
    confidence: float = 1.0

    @property
    def coordinates(self) -> EmotionCoordinates:
        return EmotionCoordinates(valence=self.valence, arousal=self.arousal)

    def to_mood(self) -> MoodType:
        return self.coordinates.to_mood_type()


@dataclass
class EmotionCurve:
    """
    Emotion curve over time for a video/audio track.

    Stores valence and arousal values at regular intervals,
    allowing interpolation for any time point.
    """

    points: list[EmotionPoint] = field(default_factory=list)
    duration: float = 0.0
    sample_rate: float = 0.5  # Points per second

    def add_point(self, point: EmotionPoint) -> None:
        """Add a point to the curve."""
        self.points.append(point)
        self.points.sort(key=lambda p: p.time)
        self.duration = max(self.duration, point.time)

    def get_emotion_at(self, time: float) -> EmotionCoordinates:
        """
        Get interpolated emotion at a specific time.

        Uses linear interpolation between nearest points.
        """
        if not self.points:
            return EmotionCoordinates(valence=0.0, arousal=0.0)

        # Handle edge cases
        if time <= self.points[0].time:
            return self.points[0].coordinates
        if time >= self.points[-1].time:
            return self.points[-1].coordinates

        # Find surrounding points
        for i in range(len(self.points) - 1):
            if self.points[i].time <= time < self.points[i + 1].time:
                p1, p2 = self.points[i], self.points[i + 1]
                # FIX: Prevent Division by Zero when duplicate keyframes exist
                time_diff = p2.time - p1.time
                if time_diff < 1e-9:
                    return p1.coordinates  # Return first point for duplicates
                # Linear interpolation
                t = (time - p1.time) / time_diff
                valence = p1.valence + t * (p2.valence - p1.valence)
                arousal = p1.arousal + t * (p2.arousal - p1.arousal)
                return EmotionCoordinates(valence=valence, arousal=arousal)

        return self.points[-1].coordinates

    def get_mood_at(self, time: float) -> MoodType:
        """Get the mood type at a specific time."""
        return self.get_emotion_at(time).to_mood_type()

    def get_average_emotion(self, start_time: float, end_time: float) -> EmotionCoordinates:
        """Get average emotion over a time range."""
        if not self.points or start_time >= end_time:
            return EmotionCoordinates(valence=0.0, arousal=0.0)

        # Sample at regular intervals
        num_samples = max(2, int((end_time - start_time) * 4))  # 4 samples/sec
        times = np.linspace(start_time, end_time, num_samples)

        valences = []
        arousals = []
        for t in times:
            emotion = self.get_emotion_at(t)
            valences.append(emotion.valence)
            arousals.append(emotion.arousal)

        return EmotionCoordinates(
            valence=float(np.mean(valences)), arousal=float(np.mean(arousals))
        )

    def get_emotion_gradient(self, time: float, window: float = 1.0) -> tuple[float, float]:
        """
        Get rate of change of emotion at a time point.

        Returns:
            (valence_gradient, arousal_gradient) - positive = increasing
        """
        e1 = self.get_emotion_at(time - window / 2)
        e2 = self.get_emotion_at(time + window / 2)

        valence_grad = (e2.valence - e1.valence) / window
        arousal_grad = (e2.arousal - e1.arousal) / window

        return valence_grad, arousal_grad

    def find_emotion_peaks(
        self, emotion_type: str = "arousal", threshold: float = 0.7
    ) -> list[float]:
        """
        Find time points where emotion peaks above threshold.

        Args:
            emotion_type: "arousal" or "valence"
            threshold: Minimum value for peak detection

        Returns:
            List of time points
        """
        peaks = []
        for i, point in enumerate(self.points):
            value = point.arousal if emotion_type == "arousal" else point.valence

            if value >= threshold:
                # Check if it's a local maximum
                is_peak = True
                if i > 0:
                    prev_val = (
                        self.points[i - 1].arousal
                        if emotion_type == "arousal"
                        else self.points[i - 1].valence
                    )
                    if prev_val > value:
                        is_peak = False
                if i < len(self.points) - 1:
                    next_val = (
                        self.points[i + 1].arousal
                        if emotion_type == "arousal"
                        else self.points[i + 1].valence
                    )
                    if next_val > value:
                        is_peak = False

                if is_peak:
                    peaks.append(point.time)

        return peaks

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "points": [
                {
                    "time": p.time,
                    "valence": p.valence,
                    "arousal": p.arousal,
                    "segment_type": p.segment_type,
                    "confidence": p.confidence,
                }
                for p in self.points
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmotionCurve:
        """Deserialize from dictionary."""
        curve = cls(duration=data.get("duration", 0.0), sample_rate=data.get("sample_rate", 0.5))
        for p_data in data.get("points", []):
            curve.points.append(
                EmotionPoint(
                    time=p_data["time"],
                    valence=p_data["valence"],
                    arousal=p_data["arousal"],
                    segment_type=p_data.get("segment_type"),
                    confidence=p_data.get("confidence", 1.0),
                )
            )
        return curve


class EmotionCurveGenerator:
    """
    Generates emotion curves from audio analysis data.

    Converts audio features (energy, spectral, beats) into
    valence/arousal values over time.
    """

    def __init__(
        self,
        sample_rate: float = 2.0,  # Emotion samples per second
        smoothing_window: float = 1.0,  # Seconds for smoothing
    ):
        self.sample_rate = sample_rate
        self.smoothing_window = smoothing_window
        self.audio_mapper = AudioVisualMapper()

        logger.info(
            f"EmotionCurveGenerator initialized: "
            f"sample_rate={sample_rate}/s, smoothing={smoothing_window}s"
        )

    def generate_from_audio_features(
        self,
        energy_curve: list[tuple[float, float]],  # [(time, energy), ...]
        spectral_features: dict[str, list[tuple[float, float]]] | None = None,
        segments: list[dict[str, Any]] | None = None,
        duration: float | None = None,
    ) -> EmotionCurve:
        """
        Generate emotion curve from audio analysis data.

        Args:
            energy_curve: List of (time, energy) tuples
            spectral_features: Optional dict with spectral feature curves
                - "brightness": [(time, value), ...]
                - "richness": [(time, value), ...]
            segments: Optional list of segment dicts with type and time range
            duration: Total duration (auto-detected if not provided)

        Returns:
            EmotionCurve with valence/arousal over time
        """
        if not energy_curve:
            logger.warning("No energy curve provided, creating neutral emotion curve")
            return EmotionCurve()

        # Determine duration
        if duration is None:
            duration = max(t for t, _ in energy_curve)

        curve = EmotionCurve(duration=duration, sample_rate=self.sample_rate)

        # Create lookup for segment types
        segment_at_time = self._build_segment_lookup(segments, duration) if segments else {}

        # Generate emotion points at regular intervals
        num_samples = int(duration * self.sample_rate) + 1
        sample_times = np.linspace(0, duration, num_samples)

        for t in sample_times:
            # Get energy at this time
            energy = self._interpolate_curve(energy_curve, t)

            # Get spectral features if available
            brightness = 0.5
            richness = 0.5
            if spectral_features:
                if "brightness" in spectral_features:
                    brightness = self._interpolate_curve(spectral_features["brightness"], t)
                if "richness" in spectral_features:
                    richness = self._interpolate_curve(spectral_features["richness"], t)

            # Map to emotion space
            emotion = self.audio_mapper.map_audio_to_emotion_space(
                {
                    "brightness": brightness,
                    "richness": richness,
                    "rms_energy": energy,
                    "percussiveness": energy * 0.8,  # Approximate
                }
            )

            # Get segment type if available
            segment_type = segment_at_time.get(int(t * 2), None)

            curve.add_point(
                EmotionPoint(
                    time=t,
                    valence=emotion.valence,
                    arousal=emotion.arousal,
                    segment_type=segment_type,
                )
            )

        # Apply smoothing
        curve = self._smooth_curve(curve)

        logger.info(f"Generated emotion curve: {len(curve.points)} points over {duration:.1f}s")

        return curve

    def generate_from_energy_array(
        self, energy: np.ndarray, sr: int = 22050, hop_length: int = 512
    ) -> EmotionCurve:
        """
        Generate emotion curve from raw energy array.

        Args:
            energy: 1D numpy array of energy values
            sr: Sample rate of original audio
            hop_length: Hop length used for analysis

        Returns:
            EmotionCurve
        """
        # Convert to time-value pairs
        times = np.arange(len(energy)) * hop_length / sr
        energy_normalized = energy / (np.max(energy) + 1e-6)

        energy_curve = list(zip(times, energy_normalized))
        duration = times[-1] if len(times) > 0 else 0.0

        return self.generate_from_audio_features(energy_curve, duration=duration)

    def generate_from_segments(
        self, segments: list[dict[str, Any]], duration: float
    ) -> EmotionCurve:
        """
        Generate emotion curve from music segments only.

        Args:
            segments: List of segment dicts with:
                - "type": segment type (intro, verse, chorus, etc.)
                - "start": start time
                - "end": end time
                - "energy": optional energy level (0-1)

        Returns:
            EmotionCurve with segment-based emotions
        """
        curve = EmotionCurve(duration=duration, sample_rate=self.sample_rate)

        # Segment type to emotion mapping
        segment_emotions = {
            "intro": EmotionCoordinates(valence=0.0, arousal=-0.3),
            "verse": EmotionCoordinates(valence=0.2, arousal=0.0),
            "chorus": EmotionCoordinates(valence=0.6, arousal=0.5),
            "drop": EmotionCoordinates(valence=0.5, arousal=0.9),
            "bridge": EmotionCoordinates(valence=-0.1, arousal=-0.2),
            "outro": EmotionCoordinates(valence=0.1, arousal=-0.4),
            "buildup": EmotionCoordinates(valence=0.3, arousal=0.6),
            "breakdown": EmotionCoordinates(valence=-0.2, arousal=-0.5),
        }

        for segment in segments:
            seg_type = segment.get("type", "").lower()
            start = segment.get("start", 0)
            end = segment.get("end", start + 4)
            energy = segment.get("energy", 0.5)

            # Get base emotion for segment type
            base_emotion = segment_emotions.get(
                seg_type, EmotionCoordinates(valence=0.0, arousal=0.0)
            )

            # Modulate by energy
            valence = base_emotion.valence + (energy - 0.5) * 0.3
            arousal = base_emotion.arousal + (energy - 0.5) * 0.5

            # Add points at segment boundaries and middle
            for t in [start, (start + end) / 2, end]:
                if 0 <= t <= duration:
                    curve.add_point(
                        EmotionPoint(
                            time=t,
                            valence=np.clip(valence, -1, 1),
                            arousal=np.clip(arousal, -1, 1),
                            segment_type=seg_type,
                        )
                    )

        # Apply smoothing for transitions
        curve = self._smooth_curve(curve)

        return curve

    def _interpolate_curve(self, curve_points: list[tuple[float, float]], time: float) -> float:
        """Interpolate value at a specific time from curve points."""
        if not curve_points:
            return 0.5

        # Sort by time
        points = sorted(curve_points, key=lambda x: x[0])

        # Handle edge cases
        if time <= points[0][0]:
            return points[0][1]
        if time >= points[-1][0]:
            return points[-1][1]

        # Find surrounding points and interpolate
        for i in range(len(points) - 1):
            if points[i][0] <= time < points[i + 1][0]:
                t1, v1 = points[i]
                t2, v2 = points[i + 1]
                # FIX: Prevent Division by Zero when duplicate times exist
                time_diff = t2 - t1
                if time_diff < 1e-9:
                    return v1  # Return first value for duplicates
                t = (time - t1) / time_diff
                return v1 + t * (v2 - v1)

        return points[-1][1]

    def _build_segment_lookup(
        self, segments: list[dict[str, Any]], duration: float
    ) -> dict[int, str]:
        """Build time-to-segment-type lookup."""
        lookup = {}
        for seg in segments:
            seg_type = seg.get("type", "unknown")
            start = seg.get("start", 0)
            end = seg.get("end", start + 4)

            # Store at half-second resolution
            for t in range(int(start * 2), int(end * 2) + 1):
                lookup[t] = seg_type

        return lookup

    def _smooth_curve(self, curve: EmotionCurve) -> EmotionCurve:
        """Apply smoothing to emotion curve."""
        if len(curve.points) < 3:
            return curve

        # Extract values
        times = [p.time for p in curve.points]
        valences = [p.valence for p in curve.points]
        arousals = [p.arousal for p in curve.points]
        segment_types = [p.segment_type for p in curve.points]
        confidences = [p.confidence for p in curve.points]

        # Apply moving average smoothing
        window_samples = max(1, int(self.smoothing_window * self.sample_rate))

        smoothed_valences = self._moving_average(valences, window_samples)
        smoothed_arousals = self._moving_average(arousals, window_samples)

        # Create new curve with smoothed values
        smoothed_curve = EmotionCurve(duration=curve.duration, sample_rate=curve.sample_rate)

        for i, t in enumerate(times):
            smoothed_curve.add_point(
                EmotionPoint(
                    time=t,
                    valence=smoothed_valences[i],
                    arousal=smoothed_arousals[i],
                    segment_type=segment_types[i],
                    confidence=confidences[i],
                )
            )

        return smoothed_curve

    def _moving_average(self, values: list[float], window: int) -> list[float]:
        """Apply moving average smoothing."""
        if window <= 1:
            return values

        result = []
        half_window = window // 2

        for i in range(len(values)):
            start = max(0, i - half_window)
            end = min(len(values), i + half_window + 1)
            result.append(np.mean(values[start:end]))

        return result


class EmotionClipMatcher:
    """
    Matches clips to emotion curve for optimal selection.

    Combines emotion matching with MMR diversity for
    high-quality, varied video generation.
    """

    def __init__(
        self,
        emotion_weight: float = 0.4,
        diversity_weight: float = 0.3,
        relevance_weight: float = 0.3,
    ):
        """
        Initialize matcher with scoring weights.

        Args:
            emotion_weight: Weight for emotion matching (0-1)
            diversity_weight: Weight for clip diversity (0-1)
            relevance_weight: Weight for motion/energy matching (0-1)
        """
        # Normalize weights
        total = emotion_weight + diversity_weight + relevance_weight
        self.emotion_weight = emotion_weight / total
        self.diversity_weight = diversity_weight / total
        self.relevance_weight = relevance_weight / total

        self.selection_history: list[int] = []
        self.emotion_history: list[EmotionCoordinates] = []

        logger.info(
            f"EmotionClipMatcher initialized: "
            f"emotion={self.emotion_weight:.1%}, "
            f"diversity={self.diversity_weight:.1%}, "
            f"relevance={self.relevance_weight:.1%}"
        )

    def score_clip(
        self,
        clip_analysis: dict[str, Any],
        target_emotion: EmotionCoordinates,
        target_motion: float,
        target_energy: float,
    ) -> tuple[float, dict[str, float]]:
        """
        Score a clip for selection.

        Args:
            clip_analysis: Clip analysis data with mood/motion info
            target_emotion: Target emotion from curve
            target_motion: Target motion level (0-1)
            target_energy: Target energy level (0-1)

        Returns:
            (total_score, component_scores_dict)
        """
        scores = {}

        # 1. Emotion matching score
        clip_emotion = self._extract_clip_emotion(clip_analysis)
        emotion_distance = target_emotion.distance_to(clip_emotion)
        # Max distance in emotion space is sqrt(8) ~ 2.83
        emotion_score = max(0, 1 - emotion_distance / 2.0)
        scores["emotion"] = emotion_score

        # 2. Relevance score (motion/energy matching)
        clip_motion = clip_analysis.get("motion", {}).get("motion_score", 0.5)
        clip_energy = clip_analysis.get("mood", {}).get("energy", 0.5)

        motion_diff = abs(target_motion - clip_motion)
        energy_diff = abs(target_energy - clip_energy)
        relevance_score = 1 - (motion_diff + energy_diff) / 2
        scores["relevance"] = relevance_score

        # 3. Diversity score (based on history)
        if self.emotion_history:
            # Calculate average distance to recent emotions
            distances = [
                clip_emotion.distance_to(hist_emotion)
                for hist_emotion in self.emotion_history[-10:]
            ]
            avg_distance = np.mean(distances)
            diversity_score = min(1.0, avg_distance / 1.5)
        else:
            diversity_score = 1.0
        scores["diversity"] = diversity_score

        # Calculate weighted total
        total_score = (
            self.emotion_weight * emotion_score
            + self.relevance_weight * relevance_score
            + self.diversity_weight * diversity_score
        )

        return total_score, scores

    def select_best_clip(
        self,
        candidates: list[dict[str, Any]],
        target_emotion: EmotionCoordinates,
        target_motion: float,
        target_energy: float,
        exclude_ids: set | None = None,
    ) -> tuple[int, float, dict[str, float]] | None:
        """
        Select best clip from candidates.

        Args:
            candidates: List of clip dicts with 'id' and 'analysis'
            target_emotion: Target emotion
            target_motion: Target motion level
            target_energy: Target energy level
            exclude_ids: Clip IDs to exclude

        Returns:
            (clip_id, score, component_scores) or None
        """
        exclude_ids = exclude_ids or set()
        best_clip = None
        best_score = -1
        best_components = {}

        for clip in candidates:
            clip_id = clip.get("id")
            if clip_id in exclude_ids:
                continue

            analysis = clip.get("analysis", {})
            score, components = self.score_clip(
                analysis, target_emotion, target_motion, target_energy
            )

            if score > best_score:
                best_score = score
                best_clip = clip_id
                best_components = components

        if best_clip is not None:
            # Update history
            self.selection_history.append(best_clip)
            clip_emotion = self._extract_clip_emotion(
                next(c.get("analysis", {}) for c in candidates if c.get("id") == best_clip)
            )
            self.emotion_history.append(clip_emotion)

            # Limit history size
            if len(self.selection_history) > 50:
                self.selection_history = self.selection_history[-50:]
                self.emotion_history = self.emotion_history[-50:]

        return (best_clip, best_score, best_components) if best_clip else None

    def reset_history(self) -> None:
        """Reset selection history."""
        self.selection_history = []
        self.emotion_history = []

    def _extract_clip_emotion(self, analysis: dict[str, Any]) -> EmotionCoordinates:
        """Extract emotion coordinates from clip analysis."""
        mood_data = analysis.get("mood", {})
        moods = mood_data.get("moods", [])
        energy = mood_data.get("energy", 0.5)

        # Average emotion from detected moods
        if moods:
            valences = []
            arousals = []
            for mood_str in moods:
                try:
                    mood = MoodType(mood_str)
                    if mood in MOOD_EMOTION_MAP:
                        emotion = MOOD_EMOTION_MAP[mood]
                        valences.append(emotion.valence)
                        arousals.append(emotion.arousal)
                except (ValueError, KeyError):
                    continue

            if valences:
                valence = np.mean(valences)
                arousal = np.mean(arousals)
            else:
                valence = 0.0
                arousal = (energy - 0.5) * 2
        else:
            valence = 0.0
            arousal = (energy - 0.5) * 2

        return EmotionCoordinates(valence=valence, arousal=arousal)


# =============================================================================
# Convenience Functions
# =============================================================================


def create_emotion_curve_generator(
    sample_rate: float = 2.0, smoothing_window: float = 1.0
) -> EmotionCurveGenerator:
    """Create an EmotionCurveGenerator instance."""
    return EmotionCurveGenerator(sample_rate=sample_rate, smoothing_window=smoothing_window)


def create_emotion_matcher(
    emotion_weight: float = 0.4, diversity_weight: float = 0.3, relevance_weight: float = 0.3
) -> EmotionClipMatcher:
    """Create an EmotionClipMatcher instance."""
    return EmotionClipMatcher(
        emotion_weight=emotion_weight,
        diversity_weight=diversity_weight,
        relevance_weight=relevance_weight,
    )
