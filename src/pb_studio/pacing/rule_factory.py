"""
Rule Factory for PB_studio Pacing System.

Factory class for creating predefined pacing rules.
Extracted from MainWindow to reduce God Object pattern.
"""

from .energy_curve import EnergyZone
from .rule_system import (
    BeatCondition,
    ClipSelectionAction,
    EnergyCondition,
    EnergyDrivenAction,
    Rule,
    TimeRangeCondition,
)


class RuleFactory:
    """
    Factory for creating predefined pacing rules.

    Provides methods to create common rule configurations used in
    the pacing engine for video-audio synchronization.
    """

    @staticmethod
    def create_beat_sync_rule() -> Rule:
        """
        Create Beat-Synchronization rule for precise beat alignment.

        Returns:
            Rule configured for beat synchronization with high priority
        """
        return Rule(
            name="Beat Synchronization",
            condition=BeatCondition(require_downbeat=True),
            action=ClipSelectionAction(snap_to_beat=True),
            priority=10,
        )

    @staticmethod
    def create_energy_rule() -> Rule:
        """
        Create Energy-Following rule for dynamic intensity matching.

        Returns:
            Rule configured for energy-driven clip selection
        """
        return Rule(
            name="Energy Following",
            condition=EnergyCondition(energy_zone=EnergyZone.HIGH),
            action=EnergyDrivenAction(duration_range=(0.5, 2.0)),
            priority=8,
        )

    @staticmethod
    def create_phrase_rule() -> Rule:
        """
        Create Phrase-Alignment rule for musical phrase boundaries (every 4 bars).

        Returns:
            Rule configured for phrase alignment with longer clip durations
        """
        return Rule(
            name="Phrase Alignment",
            condition=BeatCondition(require_downbeat=True, beat_in_bar=0),
            action=ClipSelectionAction(snap_to_beat=True, duration_range=(2.0, 4.0)),
            priority=7,
        )

    @staticmethod
    def create_duration_rule() -> Rule:
        """
        Create Duration-Constraints rule for clip length limits.

        Returns:
            Rule configured to enforce duration constraints
        """
        return Rule(
            name="Duration Constraints",
            condition=TimeRangeCondition(start_time=0.0),
            action=ClipSelectionAction(duration_range=(1.0, 4.0), snap_to_beat=True),
            priority=5,
        )

    @staticmethod
    def create_variety_rule() -> Rule:
        """
        Create Variety-Enforcement rule with varied durations.

        Returns:
            Rule configured to encourage variety in clip durations
        """
        return Rule(
            name="Variety Enforcement",
            condition=TimeRangeCondition(start_time=0.0),
            action=ClipSelectionAction(duration_range=(0.5, 3.0), snap_to_beat=True),
            priority=6,
        )

    @classmethod
    def get_rule_by_name(cls, rule_name: str) -> Rule | None:
        """
        Get a rule by its display name.

        Args:
            rule_name: Display name of the rule

        Returns:
            Rule instance or None if not found
        """
        rule_mapping = {
            "Beat Synchronization": cls.create_beat_sync_rule,
            "Energy Following": cls.create_energy_rule,
            "Phrase Alignment": cls.create_phrase_rule,
            "Duration Constraints": cls.create_duration_rule,
            "Variety Enforcement": cls.create_variety_rule,
        }

        factory_method = rule_mapping.get(rule_name)
        if factory_method:
            return factory_method()
        return None

    @classmethod
    def get_all_rule_names(cls) -> list[str]:
        """
        Get list of all available rule names.

        Returns:
            List of rule display names
        """
        return [
            "Beat Synchronization",
            "Energy Following",
            "Phrase Alignment",
            "Duration Constraints",
            "Variety Enforcement",
        ]
