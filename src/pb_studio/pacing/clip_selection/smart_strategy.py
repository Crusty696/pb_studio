"""
SmartMatcher-based Clip Selection Strategy

Wraps the SmartMatcher for intelligent clip selection.
"""

import time

from ...utils.logger import get_logger
from .base_strategy import ClipSelectionResult, ClipSelectionStrategy

logger = get_logger()


class SmartStrategy(ClipSelectionStrategy):
    """
    Intelligent clip selection using SmartMatcher (Unified Logic).

    Best for:
    - General purpose quality selection
    - When FAISS is overkill or unavailable
    - Balancing visual/audio/structure
    """

    def __init__(self, smart_matcher):
        """
        Initialize Smart strategy.

        Args:
            smart_matcher: SmartMatcher instance
        """
        self._matcher = smart_matcher

    @property
    def name(self) -> str:
        return "Smart"

    def is_available(self) -> bool:
        """Check if smart matcher is available."""
        return self._matcher is not None

    def select_clip(
        self,
        target_energy: float,
        target_motion: float,
        target_motion_type: str,
        target_mood: str,
        excluded_clips: set[int],
        available_clips: list,
        **kwargs,
    ) -> ClipSelectionResult | None:
        """
        Select clip using SmartMatcher.

        Args:
            target_energy: Target energy (0.0-1.0)
            target_motion: Target motion score (0.0-1.0)
            target_motion_type: Motion type string
            target_mood: Target mood string
            excluded_clips: Clip IDs to exclude
            available_clips: Available clips list

        Returns:
            ClipSelectionResult or None
        """
        if not self.is_available():
            return None

        start_time = time.time()

        try:
            # Filter candidates (SmartMatcher usually takes IDs but we can pre-filter)
            candidate_ids = [
                c.get("id")
                for c in available_clips
                if c.get("id") is not None and c.get("id") not in excluded_clips
            ]

            if not candidate_ids:
                return None

            # Map arguments to SmartMatcher expected format
            # Using find_best_clip signature from analysis
            smart_result = self._matcher.find_best_clip(
                target_mood=target_mood,
                target_motion=target_motion,
                target_energy=target_energy,
                min_duration=kwargs.get("min_duration", 0.0),
                candidate_ids=candidate_ids,
                avoid_recent=5,  # Hardcoded in original, could be configurable
            )

            if not smart_result or not smart_result.clip_id:
                return None

            # Find full clip dict
            # We iterate available_clips to find the matching dict
            selected_clip = next(
                (c for c in available_clips if c.get("id") == smart_result.clip_id), None
            )

            if not selected_clip:
                return None

            search_time = (time.time() - start_time) * 1000

            return ClipSelectionResult(
                clip_id=smart_result.clip_id,
                clip_data=selected_clip,
                distance=1.0 - smart_result.score if hasattr(smart_result, "score") else 0.0,
                strategy_name=self.name,
                search_time_ms=search_time,
            )

        except Exception as e:
            logger.warning(f"Smart matching failed: {e}")
            return None
