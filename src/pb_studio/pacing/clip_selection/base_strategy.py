"""
Base Strategy for Clip Selection

Abstract base class defining the interface for all clip selection strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ClipSelectionResult:
    """Result of a clip selection operation."""

    clip_id: int
    clip_data: dict[str, Any]
    distance: float = 0.0
    strategy_name: str = ""
    search_time_ms: float = 0.0


class ClipSelectionStrategy(ABC):
    """Abstract base class for clip selection strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return strategy name for logging."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this strategy can be used (dependencies ready)."""
        pass

    @abstractmethod
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
        Select a clip matching the criteria.

        Args:
            target_energy: Target energy level (0.0-1.0)
            target_motion: Target motion score (0.0-1.0)
            target_motion_type: Motion type ('STATIC', 'SLOW', 'MEDIUM', 'FAST', 'EXTREME')
            target_mood: Target mood string
            excluded_clips: Set of clip IDs to exclude
            available_clips: List of available clip dictionaries

        Returns:
            ClipSelectionResult or None if no suitable clip found
        """
        pass

    def prepare(self, available_clips: list) -> None:
        """
        Prepare strategy for selection (e.g., build index).
        Override in subclasses if needed.
        """
        pass

    def cleanup(self) -> None:
        """
        Cleanup resources after selection.
        Override in subclasses if needed.
        """
        pass
