"""
Diversity Manager for Clip Selection

Manages clip usage tracking to ensure variety in video output.
Prevents the same clips from being used too frequently.
"""

from dataclasses import dataclass, field

from ...utils.logger import get_logger

logger = get_logger()


@dataclass
class DiversityManager:
    """
    Manages clip diversity by tracking used clips.

    Uses a set for O(1) membership checks and automatically
    resets when threshold percentage of clips have been used.
    """

    total_clips: int
    reset_threshold_percent: float = 0.75

    # Internal state
    _used_clips: set[int] = field(default_factory=set)
    _usage_order: list[int] = field(default_factory=list)
    _reset_count: int = 0

    def __post_init__(self):
        self._reset_threshold = int(self.total_clips * self.reset_threshold_percent)

    @property
    def excluded_clips(self) -> set[int]:
        """Get set of clips to exclude from selection."""
        return self._used_clips

    @property
    def unique_clips_used(self) -> int:
        """Number of unique clips used since last reset."""
        return len(self._used_clips)

    @property
    def total_selections(self) -> int:
        """Total number of clip selections made."""
        return len(self._usage_order)

    def should_reset(self) -> bool:
        """Check if exclusion list should be reset."""
        return len(self._used_clips) >= self._reset_threshold

    def record_used(self, clip_id: int) -> None:
        """
        Record that a clip was used.

        Args:
            clip_id: ID of the used clip
        """
        self._used_clips.add(clip_id)
        self._usage_order.append(clip_id)

    def reset_exclusions(self) -> None:
        """Reset the exclusion list while keeping statistics."""
        if self._used_clips:
            logger.debug(
                f"Resetting exclusion list (used {len(self._used_clips)} unique clips, "
                f"reset #{self._reset_count + 1})"
            )
        self._used_clips.clear()
        self._reset_count += 1

    def check_and_reset(self) -> bool:
        """
        Check if reset is needed and perform it.

        Returns:
            True if reset was performed
        """
        if self.should_reset():
            self.reset_exclusions()
            return True
        return False

    def get_statistics(self) -> dict:
        """Get diversity statistics."""
        total_available = self.total_clips
        unique_used = len(set(self._usage_order))  # All-time unique
        usage_percent = (unique_used / total_available * 100) if total_available > 0 else 0

        return {
            "total_clips": total_available,
            "unique_clips_used": unique_used,
            "total_selections": len(self._usage_order),
            "usage_percent": usage_percent,
            "reset_count": self._reset_count,
            "current_excluded": len(self._used_clips),
        }
