"""
FAISS-based Clip Selection Strategy

Uses FAISS vector search for fast, approximate nearest neighbor matching.
100-1000x faster than brute-force for large clip collections.
"""

import time

from ...utils.logger import get_logger
from .base_strategy import ClipSelectionResult, ClipSelectionStrategy

logger = get_logger()


class FAISSStrategy(ClipSelectionStrategy):
    """
    FAISS-based clip selection using vector similarity search.

    Best for:
    - Large clip collections (100+ clips)
    - When motion/energy matching is important
    - Production use with performance requirements
    """

    def __init__(self, faiss_matcher, clips_by_id: dict[int, dict] | None = None):
        """
        Initialize FAISS strategy.

        Args:
            faiss_matcher: FAISSClipMatcher instance
            clips_by_id: Optional pre-built dict for O(1) clip lookup
        """
        self._matcher = faiss_matcher
        self._clips_by_id = clips_by_id or {}
        self._is_prepared = False

    @property
    def name(self) -> str:
        return "FAISS"

    def is_available(self) -> bool:
        """Check if FAISS matcher is ready."""
        return self._matcher is not None and self._matcher.is_ready()

    def prepare(self, available_clips: list) -> None:
        """Build FAISS index if needed."""
        if self._matcher is None:
            return

        start_time = time.time()

        if not self._matcher.is_ready():
            self._matcher.build_index(available_clips)
            build_time = time.time() - start_time
            logger.info(f"FAISS index built in {build_time * 1000:.1f}ms")
        else:
            logger.debug("FAISS index already built, reusing")

        # Build clips_by_id for O(1) lookup
        if not self._clips_by_id:
            self._clips_by_id = {
                clip.get("id"): clip for clip in available_clips if clip.get("id") is not None
            }

        self._is_prepared = True

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
        Select clip using FAISS vector search.

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
            # Extract continuity parameters
            continuity_weight = kwargs.get("continuity_weight", 0.0)
            previous_clip_id = kwargs.get("previous_clip_id")

            # FAISS search with exclusion set (O(1) per check)
            clip_id, file_path, distance = self._matcher.find_best_clip(
                target_motion_score=target_motion,
                target_energy=target_energy,
                target_motion_type=target_motion_type,
                target_moods=[target_mood],
                k=min(200, len(available_clips)),
                exclude_ids=excluded_clips,
                previous_clip_id=previous_clip_id,
                continuity_weight=continuity_weight,
            )

            if clip_id is None:
                return None

            # O(1) lookup for full clip data
            clip_data = self._clips_by_id.get(clip_id)
            if clip_data is None:
                # Fallback to first available
                clip_data = available_clips[0] if available_clips else None
                if clip_data is None:
                    return None

            search_time = (time.time() - start_time) * 1000

            return ClipSelectionResult(
                clip_id=clip_id,
                clip_data=clip_data,
                distance=distance,
                strategy_name=self.name,
                search_time_ms=search_time,
            )

        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return None

    def cleanup(self) -> None:
        """Clear FAISS index to free memory."""
        if self._matcher is not None:
            try:
                self._matcher.clear_index()
                logger.debug("FAISS index cleared")
            except Exception as e:
                logger.warning(f"Failed to clear FAISS index: {e}")
        self._is_prepared = False
