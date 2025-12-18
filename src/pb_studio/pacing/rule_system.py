"""
PB_studio Rule-Based Pacing System - Automated Cut Generation via Rules (Task 29)

Implements a rule engine for automatic video clip selection and placement based on
audio analysis, metadata, and user-defined logical rules. Enables complex pacing
strategies through declarative rule definitions.

Features:
- Condition system for audio/video metadata evaluation
- Action system for cut generation with clip selection
- Rule composition (IF condition THEN action)
- Rule priority and conflict resolution
- Integration with EnergyAnalyzer, BeatGridInfo, and metadata systems
- Hard cut generation only (no transitions)

Usage:
    from pb_studio.pacing import (
        RuleEngine, Rule, Condition, Action,
        EnergyCondition, ClipSelectionAction
    )

    # Define rules
    rule1 = Rule(
        name="High Energy -> Fast Clips",
        condition=EnergyCondition(energy_zone="high"),
        action=ClipSelectionAction(
            category="action",
            duration_range=(0.5, 1.0)
        ),
        priority=10
    )

    # Create rule engine and apply rules
    engine = RuleEngine(beatgrid=beatgrid, energy_curve=energy_curve)
    engine.add_rule(rule1)
    cuts = engine.apply_rules(duration=60.0)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .energy_curve import EnergyCurveData, EnergyZone
from .pacing_engine import BeatGridInfo
from .pacing_models import CutListEntry

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Type Aliases
# ============================================================================


class ComparisonOperator(str, Enum):
    """Comparison operators for condition evaluation."""

    EQUAL = "=="
    NOT_EQUAL = "!="
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    IN = "in"
    NOT_IN = "not_in"


class LogicalOperator(str, Enum):
    """Logical operators for combining conditions."""

    AND = "and"
    OR = "or"
    NOT = "not"


# ============================================================================
# Condition Base Classes
# ============================================================================


class Condition(ABC):
    """
    Abstract base class for all rule conditions.

    Conditions evaluate context (audio analysis, metadata, timeline position)
    to determine if a rule should trigger.
    """

    @abstractmethod
    def evaluate(self, context: dict[str, Any]) -> bool:
        """
        Evaluate condition against context.

        Args:
            context: Dictionary containing:
                - energy_curve: EnergyCurveData instance
                - beatgrid: BeatGridInfo instance
                - current_time: Current timeline position
                - available_clips: List of available video clips
                - metadata: Additional metadata

        Returns:
            True if condition is satisfied, False otherwise
        """
        raise NotImplementedError("Condition.evaluate must be implemented by subclasses")

    @abstractmethod
    def __repr__(self) -> str:
        """String representation for debugging."""
        return self.__class__.__name__


@dataclass
class EnergyCondition(Condition):
    """
    Condition based on audio energy level or zone.

    Evaluates current energy at a specific timeline position.
    """

    energy_zone: EnergyZone | None = None
    energy_min: float | None = None
    energy_max: float | None = None
    operator: ComparisonOperator = ComparisonOperator.EQUAL

    def evaluate(self, context: dict[str, Any]) -> bool:
        """Evaluate energy condition."""
        energy_curve: EnergyCurveData = context.get("energy_curve")
        current_time: float = context.get("current_time", 0.0)

        if not energy_curve:
            logger.warning("No energy curve in context, condition fails")
            return False

        # Get current energy
        current_energy = energy_curve.get_energy_at_time(current_time)

        # Check zone condition
        if self.energy_zone is not None:
            current_zone = energy_curve.get_energy_zone(current_time)
            result = current_zone == self.energy_zone
            logger.debug(f"EnergyCondition: zone={current_zone} vs {self.energy_zone} -> {result}")
            return result

        # Check min/max conditions
        if self.energy_min is not None and current_energy < self.energy_min:
            return False
        if self.energy_max is not None and current_energy > self.energy_max:
            return False

        return True

    def __repr__(self) -> str:
        if self.energy_zone:
            return f"EnergyCondition(zone={self.energy_zone})"
        return f"EnergyCondition(min={self.energy_min}, max={self.energy_max})"


@dataclass
class BeatCondition(Condition):
    """
    Condition based on beatgrid position (downbeat, specific beat, etc.).

    Useful for ensuring cuts align with musical structure.
    """

    require_downbeat: bool = False
    beat_in_bar: int | None = None  # 0-based beat number within bar

    def evaluate(self, context: dict[str, Any]) -> bool:
        """Evaluate beat condition."""
        beatgrid: BeatGridInfo = context.get("beatgrid")
        current_time: float = context.get("current_time", 0.0)

        if not beatgrid:
            logger.warning("No beatgrid in context, condition fails")
            return False

        # Calculate beat position (inline calculation from PacingEngine)
        relative_time = current_time - beatgrid.beatgrid_offset
        bar_duration = beatgrid.bar_duration
        beat_duration = beatgrid.beat_duration

        bar_num = int(relative_time / bar_duration)
        beat_in_bar = int((relative_time % bar_duration) / beat_duration)
        beat_offset = relative_time % beat_duration

        # Check downbeat requirement (beat 0 of bar)
        if self.require_downbeat:
            result = beat_in_bar == 0 and abs(beat_offset) < 0.05  # 50ms tolerance
            logger.debug(f"BeatCondition: downbeat check -> {result}")
            return result

        # Check specific beat in bar
        if self.beat_in_bar is not None:
            result = beat_in_bar == self.beat_in_bar
            logger.debug(f"BeatCondition: beat {beat_in_bar} vs {self.beat_in_bar} -> {result}")
            return result

        return True

    def __repr__(self) -> str:
        if self.require_downbeat:
            return "BeatCondition(require_downbeat=True)"
        if self.beat_in_bar is not None:
            return f"BeatCondition(beat_in_bar={self.beat_in_bar})"
        return "BeatCondition(always_true)"


@dataclass
class TimeRangeCondition(Condition):
    """
    Condition based on timeline position range.

    Useful for limiting rules to specific sections (intro, verse, chorus, etc.).
    """

    start_time: float = 0.0
    end_time: float | None = None

    def evaluate(self, context: dict[str, Any]) -> bool:
        """Evaluate time range condition."""
        current_time: float = context.get("current_time", 0.0)

        if current_time < self.start_time:
            return False

        if self.end_time is not None and current_time >= self.end_time:
            return False

        logger.debug(f"TimeRangeCondition: {current_time}s in [{self.start_time}, {self.end_time})")
        return True

    def __repr__(self) -> str:
        return f"TimeRangeCondition(start={self.start_time}, end={self.end_time})"


@dataclass
class CompositeCondition(Condition):
    """
    Combines multiple conditions with logical operators (AND, OR, NOT).

    Enables complex conditional logic like:
    (EnergyCondition(high) AND BeatCondition(downbeat)) OR TimeRangeCondition(...)
    """

    conditions: list[Condition] = field(default_factory=list)
    operator: LogicalOperator = LogicalOperator.AND

    def evaluate(self, context: dict[str, Any]) -> bool:
        """Evaluate composite condition."""
        if not self.conditions:
            return True  # Empty condition list = always true

        results = [cond.evaluate(context) for cond in self.conditions]

        if self.operator == LogicalOperator.AND:
            return all(results)
        elif self.operator == LogicalOperator.OR:
            return any(results)
        elif self.operator == LogicalOperator.NOT:
            # For NOT, apply to first condition only
            return not results[0] if results else True

        return False

    def __repr__(self) -> str:
        return f"CompositeCondition({self.operator.value}, {len(self.conditions)} conditions)"


# ============================================================================
# Action Base Classes
# ============================================================================


class Action(ABC):
    """
    Abstract base class for all rule actions.

    Actions generate CutListEntry objects based on context when triggered.
    """

    @abstractmethod
    def execute(self, context: dict[str, Any]) -> CutListEntry | None:
        """
        Execute action and generate cut.

        Args:
            context: Same context dict as conditions

        Returns:
            CutListEntry if successful, None if action cannot execute
        """
        raise NotImplementedError("Action.execute must be implemented by subclasses")

    @abstractmethod
    def __repr__(self) -> str:
        """String representation for debugging."""
        return self.__class__.__name__


@dataclass
class ClipSelectionAction(Action):
    """
    Action that selects and places a video clip based on criteria.

    Selects clips from a category/pool with specified duration constraints.
    """

    category: str | None = None  # Clip category/tag filter
    clip_id_prefix: str | None = None  # Specific clip ID pattern
    duration_range: tuple[float, float] = (1.0, 3.0)  # (min, max) duration
    snap_to_beat: bool = True

    def execute(self, context: dict[str, Any]) -> CutListEntry | None:
        """Execute clip selection action."""
        current_time = context.get("current_time", 0.0)
        beatgrid: BeatGridInfo = context.get("beatgrid")
        available_clips = context.get("available_clips", [])

        # Select clip (simplified - would normally query database/pool)
        clip_id = self._select_clip(available_clips)

        # Calculate duration
        min_dur, max_dur = self.duration_range

        # For now, use fixed duration (could be randomized or energy-driven)
        duration = (min_dur + max_dur) / 2

        # Snap start time to beat if requested
        start_time = current_time
        if self.snap_to_beat and beatgrid:
            # Snap to nearest beat (round up to prevent overlaps)
            beat_num = int(current_time / beatgrid.beat_duration)
            snapped_time = beat_num * beatgrid.beat_duration + beatgrid.beatgrid_offset

            # If snap would go backwards, use next beat
            if snapped_time < current_time:
                snapped_time = (beat_num + 1) * beatgrid.beat_duration + beatgrid.beatgrid_offset

            start_time = snapped_time

        end_time = start_time + duration

        logger.debug(
            f"ClipSelectionAction: selected {clip_id}, "
            f"{start_time:.3f}s-{end_time:.3f}s (dur={duration:.3f}s)"
        )

        return CutListEntry(clip_id=clip_id, start_time=start_time, end_time=end_time)

    def _select_clip(self, available_clips: list) -> str:
        """
        Select clip from available pool based on criteria.

        Simplified implementation - production version would:
        - Query video database by category/tags
        - Apply filters (duration, format, etc.)
        - Handle clip selection strategy (random, sequential, weighted)
        """
        if self.clip_id_prefix:
            # Use specified clip ID pattern
            return f"{self.clip_id_prefix}_001"

        if self.category:
            # Use category-based naming
            return f"{self.category}_clip_001"

        # Default fallback
        return "clip_001"

    def __repr__(self) -> str:
        return f"ClipSelectionAction(category={self.category}, duration={self.duration_range})"


@dataclass
class EnergyDrivenAction(Action):
    """
    Action that generates cuts with energy-driven duration mapping.

    Similar to EnergyBasedPacingEngine adaptive mode, but within rule framework.
    """

    clip_id_prefix: str = "clip"
    duration_range: tuple[float, float] = (0.5, 2.0)  # (high_energy_dur, low_energy_dur)
    snap_to_beat: bool = True

    def execute(self, context: dict[str, Any]) -> CutListEntry | None:
        """Execute energy-driven action."""
        current_time = context.get("current_time", 0.0)
        energy_curve: EnergyCurveData = context.get("energy_curve")
        beatgrid: BeatGridInfo = context.get("beatgrid")

        if not energy_curve:
            logger.warning("No energy curve in context, action cannot execute")
            return None

        # Get current energy
        current_energy = energy_curve.get_energy_at_time(current_time)

        # Map energy to duration (inverse: high energy = short cuts)
        short_dur, long_dur = self.duration_range
        duration = long_dur - (current_energy * (long_dur - short_dur))

        # Snap start time to beat if requested
        start_time = current_time
        if self.snap_to_beat and beatgrid:
            # Snap to nearest beat (round up to prevent overlaps)
            beat_num = int(current_time / beatgrid.beat_duration)
            snapped_time = beat_num * beatgrid.beat_duration + beatgrid.beatgrid_offset

            # If snap would go backwards, use next beat
            if snapped_time < current_time:
                snapped_time = (beat_num + 1) * beatgrid.beat_duration + beatgrid.beatgrid_offset

            start_time = snapped_time

        end_time = start_time + duration

        # Generate clip ID
        clip_id = f"{self.clip_id_prefix}_{int(current_time * 100):06d}"

        logger.debug(f"EnergyDrivenAction: energy={current_energy:.3f} -> duration={duration:.3f}s")

        return CutListEntry(clip_id=clip_id, start_time=start_time, end_time=end_time)

    def __repr__(self) -> str:
        return f"EnergyDrivenAction(duration_range={self.duration_range})"


# ============================================================================
# Rule and Rule Engine
# ============================================================================


@dataclass
class Rule:
    """
    Complete rule: IF condition THEN action.

    Combines a condition (when to trigger) with an action (what to do).
    Priority determines evaluation order when multiple rules match.
    """

    name: str
    condition: Condition
    action: Action
    priority: int = 0  # Higher priority = evaluated first
    enabled: bool = True

    def evaluate_and_execute(self, context: dict[str, Any]) -> CutListEntry | None:
        """
        Evaluate condition and execute action if satisfied.

        Returns:
            CutListEntry if rule triggered, None otherwise
        """
        if not self.enabled:
            logger.debug(f"Rule '{self.name}' disabled, skipping")
            return None

        if self.condition.evaluate(context):
            logger.info(
                f"Rule '{self.name}' triggered at t={context.get('current_time', 0.0):.3f}s"
            )
            return self.action.execute(context)

        return None

    def __repr__(self) -> str:
        return f"Rule(name='{self.name}', priority={self.priority}, enabled={self.enabled})"


class RuleEngine:
    """
    Rule engine for applying pacing rules to generate cuts.

    Manages rule collection, evaluation order, and cut generation.
    Integrates with energy curves, beatgrids, and video clip metadata.
    """

    def __init__(
        self,
        beatgrid: BeatGridInfo,
        energy_curve: EnergyCurveData | None = None,
        available_clips: list | None = None,
    ):
        """
        Initialize rule engine.

        Args:
            beatgrid: BeatGridInfo for tempo/timing
            energy_curve: Optional EnergyCurveData for energy-based rules
            available_clips: Optional list of available video clips
        """
        self.beatgrid = beatgrid
        self.energy_curve = energy_curve
        self.available_clips = available_clips or []

        self.rules: list[Rule] = []

        logger.info(
            f"RuleEngine initialized: BPM={beatgrid.bpm}, "
            f"has_energy={energy_curve is not None}, "
            f"clips={len(self.available_clips)}"
        )

    def add_rule(self, rule: Rule) -> None:
        """
        Add rule to engine.

        Rules are sorted by priority (highest first).
        """
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
        logger.info(f"Added rule: {rule}")

    def remove_rule(self, name: str) -> bool:
        """Remove rule by name."""
        for i, rule in enumerate(self.rules):
            if rule.name == name:
                removed = self.rules.pop(i)
                logger.info(f"Removed rule: {removed}")
                return True
        return False

    def clear_rules(self) -> int:
        """Clear all rules."""
        count = len(self.rules)
        self.rules.clear()
        logger.info(f"Cleared {count} rules")
        return count

    def apply_rules(
        self, duration: float, time_step: float = 0.5, metadata: dict | None = None
    ) -> list[CutListEntry]:
        """
        Apply all rules over timeline to generate cuts.

        Args:
            duration: Total duration to process
            time_step: Time increment for rule evaluation (seconds)
            metadata: Optional additional metadata for context

        Returns:
            List of validated CutListEntry objects
        """
        if not self.rules:
            logger.warning("No rules defined, cannot generate cuts")
            return []

        cuts: list[CutListEntry] = []
        current_time = 0.0

        logger.info(f"Applying {len(self.rules)} rules over {duration}s " f"(step={time_step}s)")

        while current_time < duration:
            # Build context for this timeline position
            context = {
                "energy_curve": self.energy_curve,
                "beatgrid": self.beatgrid,
                "current_time": current_time,
                "available_clips": self.available_clips,
                "metadata": metadata or {},
            }

            # Evaluate rules in priority order
            # First matching rule generates cut
            for rule in self.rules:
                cut = rule.evaluate_and_execute(context)
                if cut:
                    # Clip cut to duration boundary if needed
                    if cut.end_time > duration:
                        cut = CutListEntry(
                            clip_id=cut.clip_id, start_time=cut.start_time, end_time=duration
                        )
                    cuts.append(cut)
                    # Advance timeline past this cut to avoid overlaps
                    current_time = cut.end_time
                    break
            else:
                # No rule matched, advance by time_step
                current_time += time_step

        logger.info(f"Generated {len(cuts)} cuts from rule application")

        return cuts

    def get_rules_summary(self) -> dict:
        """
        Get summary of all rules.

        Returns:
            Dictionary with rule metadata
        """
        return {
            "total_rules": len(self.rules),
            "enabled_rules": sum(1 for r in self.rules if r.enabled),
            "rules": [
                {
                    "name": rule.name,
                    "priority": rule.priority,
                    "enabled": rule.enabled,
                    "condition": repr(rule.condition),
                    "action": repr(rule.action),
                }
                for rule in self.rules
            ],
        }

    def __repr__(self) -> str:
        return f"RuleEngine(rules={len(self.rules)}, BPM={self.beatgrid.bpm:.1f})"
