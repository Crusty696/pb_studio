"""
Round-Robin Fallback Strategy

Simple sequential clip selection as a last resort.
Always available, no external dependencies.
"""

import time

from ...utils.logger import get_logger
from .base_strategy import ClipSelectionResult, ClipSelectionStrategy

logger = get_logger()


class RoundRobinStrategy(ClipSelectionStrategy):
    """
    Simple round-robin clip selection.

    Best for:
    - Fallback when other strategies fail
    - Testing and debugging
    - When diversity is more important than matching
    """

    def __init__(self):
        """Initialize round-robin strategy."""
        self._index = 0

    @property
    def name(self) -> str:
        return "RoundRobin"

    def is_available(self) -> bool:
        """Always available as fallback."""
        return True

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
        Select clip using round-robin (sequential) order.

        Args:
            target_energy: Not used
            target_motion: Not used
            target_motion_type: Not used
            target_mood: Not used
            excluded_clips: Clip IDs to exclude
            available_clips: Available clips list

        Returns:
            ClipSelectionResult or None if no clips available
        """
        if not available_clips:
            return None

        start_time = time.time()

        # Filter excluded clips
        filtered_clips = [c for c in available_clips if c.get("id") not in excluded_clips]

        if not filtered_clips:
            # All excluded, use original list
            filtered_clips = available_clips

        # Simple round-robin selection
        clip_data = filtered_clips[self._index % len(filtered_clips)]
        self._index += 1

        clip_id = clip_data.get("id", 0)
        search_time = (time.time() - start_time) * 1000

        return ClipSelectionResult(
            clip_id=clip_id,
            clip_data=clip_data,
            distance=1.0,  # No matching performed
            strategy_name=self.name,
            search_time_ms=search_time,
        )

    def reset(self) -> None:
        """Reset round-robin index."""
        self._index = 0
