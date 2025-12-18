"""
PB_studio Pacing Engine - Manual Trigger & Pacing System (Task 26)

Converts user timeline inputs (manual clicks/triggers) into validated CutListEntry objects
with beatgrid synchronization support.

Features:
- Manual trigger handling (timeline click → CutListEntry)
- Beatgrid synchronization (snap to beat/downbeat)
- Cut validation and conflict detection
- Support for beat subdivision (1/4, 1/8, 1/16 notes)
- Integration with Pydantic PacingBlueprint models

Usage:
    from pb_studio.pacing import PacingEngine

    # Initialize with audio track beatgrid
    engine = PacingEngine(
        bpm=140.0,
        beatgrid_offset=0.0,
        total_duration=120.0
    )

    # Add manual trigger at timeline position
    cut = engine.add_manual_trigger(
        clip_id="clip_001",
        timeline_position=5.5,
        snap_to_beat=True
    )

    # Generate blueprint
    blueprint = engine.generate_blueprint(name="Manual Edit")
"""

import logging
from dataclasses import dataclass
from typing import Literal

from .pacing_models import AudioTrackReference, CutListEntry, PacingBlueprint

logger = logging.getLogger(__name__)


# Type aliases for clarity
BeatDivision = Literal["1/1", "1/2", "1/4", "1/8", "1/16"]
SnapMode = Literal["none", "beat", "downbeat", "subdivision"]


@dataclass
class BeatGridInfo:
    """
    Beatgrid information for synchronization.

    Attributes:
        bpm: Beats per minute
        beatgrid_offset: Offset to first downbeat in seconds
        beat_duration: Duration of one beat in seconds
        time_signature: Time signature (e.g., 4 for 4/4)
    """

    bpm: float
    beatgrid_offset: float = 0.0
    time_signature: int = 4

    @property
    def beat_duration(self) -> float:
        """Calculate duration of one beat in seconds."""
        return 60.0 / self.bpm

    @property
    def bar_duration(self) -> float:
        """Calculate duration of one bar in seconds."""
        return self.beat_duration * self.time_signature

    def get_subdivision_duration(self, division: BeatDivision) -> float:
        """
        Calculate duration for a beat subdivision.

        Args:
            division: Beat subdivision (1/1, 1/2, 1/4, 1/8, 1/16)

        Returns:
            Duration in seconds
        """
        divisor_map = {
            "1/1": 1.0,  # Whole beat
            "1/2": 2.0,  # Half beat (8th notes in 4/4)
            "1/4": 4.0,  # Quarter beat (16th notes in 4/4)
            "1/8": 8.0,  # Eighth beat (32nd notes in 4/4)
            "1/16": 16.0,  # Sixteenth beat
        }
        divisor = divisor_map.get(division, 1.0)
        return self.beat_duration / divisor


class PacingEngine:
    """
    Core pacing engine for manual trigger & cut list management.

    Handles conversion of user timeline inputs to validated CutListEntry objects
    with optional beatgrid synchronization.
    """

    def __init__(
        self,
        bpm: float,
        beatgrid_offset: float = 0.0,
        time_signature: int = 4,
        total_duration: float | None = None,
    ):
        """
        Initialize pacing engine.

        Args:
            bpm: Beats per minute of audio track
            beatgrid_offset: Offset to first downbeat in seconds
            time_signature: Time signature (default: 4 for 4/4)
            total_duration: Total duration of timeline in seconds (optional)
        """
        self.beatgrid = BeatGridInfo(
            bpm=bpm, beatgrid_offset=beatgrid_offset, time_signature=time_signature
        )
        self.total_duration = total_duration

        # Internal cut list storage
        self._cuts: list[CutListEntry] = []

        logger.info(
            f"PacingEngine initialized: BPM={bpm}, "
            f"offset={beatgrid_offset}s, time_signature={time_signature}/4"
        )

    def snap_to_beat(
        self, timeline_position: float, mode: SnapMode = "beat", division: BeatDivision = "1/1"
    ) -> float:
        """
        Snap timeline position to nearest beat/downbeat/subdivision.

        Args:
            timeline_position: Original timeline position in seconds
            mode: Snap mode (none, beat, downbeat, subdivision)
            division: Beat subdivision for 'subdivision' mode

        Returns:
            Snapped timeline position in seconds
        """
        if mode == "none":
            return timeline_position

        # Adjust for beatgrid offset
        relative_time = timeline_position - self.beatgrid.beatgrid_offset

        if mode == "beat":
            # Snap to nearest beat
            beat_duration = self.beatgrid.beat_duration
            beat_index = round(relative_time / beat_duration)
            snapped_time = beat_index * beat_duration

        elif mode == "downbeat":
            # Snap to nearest downbeat (first beat of bar)
            bar_duration = self.beatgrid.bar_duration
            bar_index = round(relative_time / bar_duration)
            snapped_time = bar_index * bar_duration

        elif mode == "subdivision":
            # Snap to nearest subdivision
            subdivision_duration = self.beatgrid.get_subdivision_duration(division)
            subdivision_index = round(relative_time / subdivision_duration)
            snapped_time = subdivision_index * subdivision_duration

        else:
            snapped_time = relative_time

        # Add beatgrid offset back
        final_time = snapped_time + self.beatgrid.beatgrid_offset

        logger.debug(
            f"Snap: {timeline_position:.3f}s → {final_time:.3f}s "
            f"(mode={mode}, division={division})"
        )

        return max(0.0, final_time)  # Ensure non-negative

    def add_manual_trigger(
        self,
        clip_id: str,
        timeline_position: float,
        duration: float | None = None,
        snap_to_beat: bool = True,
        snap_mode: SnapMode = "beat",
        beat_division: BeatDivision = "1/1",
    ) -> CutListEntry:
        """
        Add manual trigger at timeline position, converting to CutListEntry.

        This is the core function for Task 26: converts user clicks/inputs
        into validated cut list entries with beatgrid synchronization.

        Args:
            clip_id: Video clip identifier
            timeline_position: User-specified position in seconds
            duration: Optional cut duration (if None, extends to next cut or timeline end)
            snap_to_beat: Enable beatgrid snapping
            snap_mode: Snap mode (beat, downbeat, subdivision)
            beat_division: Beat subdivision for subdivision mode

        Returns:
            Created and validated CutListEntry

        Raises:
            ValueError: If position is invalid or conflicts with existing cuts
        """
        # Snap to beatgrid if enabled
        if snap_to_beat:
            start_time = self.snap_to_beat(
                timeline_position, mode=snap_mode, division=beat_division
            )
        else:
            start_time = timeline_position

        # Calculate end_time
        if duration is not None:
            end_time = start_time + duration
        else:
            # Extend to next cut or timeline end
            end_time = self._get_next_cut_start(start_time)
            if end_time is None:
                if self.total_duration is not None:
                    end_time = self.total_duration
                else:
                    # Default: extend 1 beat
                    end_time = start_time + self.beatgrid.beat_duration

        # Snap end_time to beatgrid if enabled
        if snap_to_beat:
            end_time = self.snap_to_beat(end_time, mode=snap_mode, division=beat_division)

        # Create CutListEntry (Pydantic validates end > start)
        cut = CutListEntry(clip_id=clip_id, start_time=start_time, end_time=end_time)

        # Validate no conflicts with existing cuts
        self._validate_no_conflicts(cut)

        # Add to internal list (maintain chronological order)
        self._cuts.append(cut)
        self._cuts.sort(key=lambda c: c.start_time)

        logger.info(
            f"Manual trigger added: {clip_id} @ {start_time:.3f}s-{end_time:.3f}s "
            f"(snap={snap_to_beat}, duration={cut.duration:.3f}s)"
        )

        return cut

    def _get_next_cut_start(self, after_time: float) -> float | None:
        """
        Find start time of next cut after given time.

        Args:
            after_time: Time to search after

        Returns:
            Start time of next cut, or None if no cuts follow
        """
        for cut in self._cuts:
            if cut.start_time > after_time:
                return cut.start_time
        return None

    def _validate_no_conflicts(self, new_cut: CutListEntry) -> None:
        """
        Validate that new cut doesn't conflict with existing cuts.

        Args:
            new_cut: Cut to validate

        Raises:
            ValueError: If cut overlaps with existing cut
        """
        for existing_cut in self._cuts:
            # Check for overlap
            if (
                new_cut.start_time < existing_cut.end_time
                and new_cut.end_time > existing_cut.start_time
            ):
                raise ValueError(
                    f"Cut conflict: new cut ({new_cut.start_time:.2f}s-{new_cut.end_time:.2f}s) "
                    f"overlaps with existing cut ({existing_cut.start_time:.2f}s-"
                    f"{existing_cut.end_time:.2f}s, clip_id={existing_cut.clip_id})"
                )

    def remove_cut(self, clip_id: str, start_time: float) -> bool:
        """
        Remove cut by clip_id and start_time.

        Args:
            clip_id: Clip identifier
            start_time: Start time of cut to remove

        Returns:
            True if cut was removed, False if not found
        """
        for i, cut in enumerate(self._cuts):
            if cut.clip_id == clip_id and cut.start_time == start_time:
                removed = self._cuts.pop(i)
                logger.info(f"Cut removed: {removed}")
                return True

        logger.warning(f"Cut not found for removal: {clip_id} @ {start_time}s")
        return False

    def clear_cuts(self) -> int:
        """
        Clear all cuts from engine.

        Returns:
            Number of cuts removed
        """
        count = len(self._cuts)
        self._cuts.clear()
        logger.info(f"All cuts cleared: {count} cuts removed")
        return count

    def get_cuts(self) -> list[CutListEntry]:
        """
        Get all cuts in chronological order.

        Returns:
            List of CutListEntry objects (sorted by start_time)
        """
        return self._cuts.copy()

    def generate_blueprint(
        self, name: str, description: str | None = None, audio_track_id: str | None = None
    ) -> PacingBlueprint:
        """
        Generate PacingBlueprint from current cuts.

        Args:
            name: Blueprint name
            description: Optional blueprint description
            audio_track_id: Optional audio track identifier

        Returns:
            Validated PacingBlueprint

        Raises:
            ValueError: If cuts list is empty or validation fails
        """
        if not self._cuts:
            raise ValueError("Cannot generate blueprint: no cuts defined")

        # Calculate total duration from cuts
        if self._cuts:
            total_duration = max(cut.end_time for cut in self._cuts)
        else:
            total_duration = 0.0

        # Create audio track reference if audio_track_id provided
        audio_track = None
        if audio_track_id:
            audio_track = AudioTrackReference(
                track_id=audio_track_id,
                bpm=self.beatgrid.bpm,
                beatgrid_offset=self.beatgrid.beatgrid_offset,
            )

        # Create blueprint (Pydantic validates everything)
        blueprint = PacingBlueprint(
            name=name,
            total_duration=total_duration,
            description=description,
            cuts=self._cuts.copy(),
            audio_track=audio_track,
        )

        logger.info(
            f"Blueprint generated: '{name}' with {blueprint.cut_count} cuts, "
            f"duration={blueprint.total_duration:.2f}s"
        )

        return blueprint

    def get_beat_position(self, timeline_position: float) -> tuple[int, float]:
        """
        Get beat number and offset for timeline position.

        Args:
            timeline_position: Position in seconds

        Returns:
            Tuple of (beat_number, beat_offset_seconds)

        Example:
            >>> engine.get_beat_position(5.5)
            (12, 0.214)  # Beat 12, offset 0.214s into the beat
        """
        relative_time = timeline_position - self.beatgrid.beatgrid_offset
        beat_duration = self.beatgrid.beat_duration

        beat_number = int(relative_time / beat_duration)
        beat_offset = relative_time % beat_duration

        return beat_number, beat_offset

    def get_downbeat_position(self, timeline_position: float) -> tuple[int, int, float]:
        """
        Get bar number, beat within bar, and offset for timeline position.

        Args:
            timeline_position: Position in seconds

        Returns:
            Tuple of (bar_number, beat_in_bar, beat_offset_seconds)

        Example:
            >>> engine.get_downbeat_position(5.5)
            (3, 1, 0.214)  # Bar 3, beat 1 of bar, offset 0.214s
        """
        relative_time = timeline_position - self.beatgrid.beatgrid_offset
        bar_duration = self.beatgrid.bar_duration
        beat_duration = self.beatgrid.beat_duration

        bar_number = int(relative_time / bar_duration)
        time_in_bar = relative_time % bar_duration
        beat_in_bar = int(time_in_bar / beat_duration)
        beat_offset = time_in_bar % beat_duration

        return bar_number, beat_in_bar, beat_offset

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"PacingEngine(bpm={self.beatgrid.bpm:.1f}, "
            f"cuts={len(self._cuts)}, "
            f"duration={self.total_duration}s)"
        )
