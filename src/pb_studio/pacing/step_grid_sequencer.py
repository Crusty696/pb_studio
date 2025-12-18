"""
PB_studio Step-Grid-Sequencer - Drum-Machine-Style Clip Placement (Task 27)

Implements a step sequencer interface for rhythmic video clip placement,
similar to drum machine step programming. Enables precise beat-synchronized
editing with visual grid-based workflow.

Features:
- Grid-based step programming (like drum machines)
- Multiple step resolutions (1/1, 1/2, 1/4, 1/8, 1/16 beats)
- Pattern management (save/load/modify)
- Automatic CutListEntry generation from patterns
- Integration with PacingEngine beatgrid
- Support for multiple clip layers/tracks

Usage:
    from pb_studio.pacing import StepGridSequencer, BeatGridInfo

    # Initialize sequencer with beatgrid
    beatgrid = BeatGridInfo(bpm=128.0, time_signature=4)
    sequencer = StepGridSequencer(
        beatgrid=beatgrid,
        num_steps=16,
        step_resolution="1/4"
    )

    # Activate steps with clip assignments
    sequencer.activate_step(0, clip_id="intro")
    sequencer.activate_step(4, clip_id="verse")
    sequencer.activate_step(8, clip_id="chorus")

    # Generate cuts from pattern
    cuts = sequencer.generate_cuts(cut_duration=4.0)
"""

import logging
from dataclasses import dataclass, field
from typing import Literal

from .pacing_engine import BeatGridInfo
from .pacing_models import CutListEntry

logger = logging.getLogger(__name__)


# Type aliases
StepResolution = Literal["1/1", "1/2", "1/4", "1/8", "1/16"]


@dataclass
class Step:
    """
    Represents a single step in the sequencer grid.

    Attributes:
        index: Step position in grid (0-based)
        active: Whether step is activated
        clip_id: Optional video clip identifier for this step
        velocity: Step velocity/intensity (0.0-1.0, for future energy control)
    """

    index: int
    active: bool = False
    clip_id: str | None = None
    velocity: float = 1.0

    def __post_init__(self):
        """Validate step parameters."""
        if self.index < 0:
            raise ValueError(f"Step index must be non-negative, got {self.index}")
        if not 0.0 <= self.velocity <= 1.0:
            raise ValueError(f"Velocity must be 0.0-1.0, got {self.velocity}")

    def activate(self, clip_id: str, velocity: float = 1.0) -> None:
        """
        Activate step with clip assignment.

        Args:
            clip_id: Video clip identifier
            velocity: Step intensity (0.0-1.0)
        """
        self.active = True
        self.clip_id = clip_id
        self.velocity = velocity
        logger.debug(f"Step {self.index} activated: clip_id={clip_id}, velocity={velocity}")

    def deactivate(self) -> None:
        """Deactivate step and clear clip assignment."""
        self.active = False
        self.clip_id = None
        self.velocity = 1.0
        logger.debug(f"Step {self.index} deactivated")

    def __repr__(self) -> str:
        """String representation for debugging."""
        if self.active:
            return f"Step({self.index}, active, clip={self.clip_id!r}, vel={self.velocity:.2f})"
        return f"Step({self.index}, inactive)"


@dataclass
class StepPattern:
    """
    Represents a complete step pattern (collection of steps).

    Attributes:
        name: Pattern name/identifier
        steps: List of Step objects
        step_resolution: Time resolution per step
        loop: Whether pattern should loop
    """

    name: str
    steps: list[Step] = field(default_factory=list)
    step_resolution: StepResolution = "1/4"
    loop: bool = False

    @property
    def active_steps(self) -> list[Step]:
        """Get all activated steps in pattern."""
        return [step for step in self.steps if step.active]

    @property
    def num_steps(self) -> int:
        """Total number of steps in pattern."""
        return len(self.steps)

    def __repr__(self) -> str:
        """String representation for debugging."""
        active_count = len(self.active_steps)
        return (
            f"StepPattern(name={self.name!r}, steps={self.num_steps}, "
            f"active={active_count}, resolution={self.step_resolution})"
        )


class StepGridSequencer:
    """
    Step-grid sequencer for rhythmic video clip placement.

    Provides drum-machine-style interface for creating rhythmic cut patterns
    synchronized to beatgrid. Converts step patterns to CutListEntry objects
    for integration with PacingEngine.
    """

    def __init__(
        self,
        beatgrid: BeatGridInfo,
        num_steps: int = 16,
        step_resolution: StepResolution = "1/4",
        pattern_name: str = "Pattern 1",
    ):
        """
        Initialize step-grid sequencer.

        Args:
            beatgrid: BeatGridInfo for tempo/timing calculations
            num_steps: Number of steps in grid (default: 16)
            step_resolution: Time resolution per step (default: 1/4 beat)
            pattern_name: Name for initial pattern

        Raises:
            ValueError: If num_steps < 1 or > 128
        """
        if not 1 <= num_steps <= 128:
            raise ValueError(f"num_steps must be 1-128, got {num_steps}")

        self.beatgrid = beatgrid
        self.num_steps = num_steps
        self.step_resolution = step_resolution

        # Create initial empty pattern
        self.pattern = StepPattern(
            name=pattern_name,
            steps=[Step(index=i) for i in range(num_steps)],
            step_resolution=step_resolution,
        )

        logger.info(
            f"StepGridSequencer initialized: {num_steps} steps, "
            f"resolution={step_resolution}, BPM={beatgrid.bpm}"
        )

    def get_step_duration(self) -> float:
        """
        Calculate duration of one step in seconds.

        Returns:
            Step duration in seconds
        """
        return self.beatgrid.get_subdivision_duration(self.step_resolution)

    def get_step_time(self, step_index: int) -> float:
        """
        Calculate timeline position for step.

        Args:
            step_index: Step index (0-based)

        Returns:
            Timeline position in seconds (includes beatgrid offset)
        """
        step_duration = self.get_step_duration()
        relative_time = step_index * step_duration
        return relative_time + self.beatgrid.beatgrid_offset

    def activate_step(self, step_index: int, clip_id: str, velocity: float = 1.0) -> Step:
        """
        Activate step with clip assignment.

        Args:
            step_index: Step position (0-based)
            clip_id: Video clip identifier
            velocity: Step intensity (0.0-1.0)

        Returns:
            Activated Step object

        Raises:
            ValueError: If step_index out of range
        """
        if not 0 <= step_index < self.num_steps:
            raise ValueError(f"Step index {step_index} out of range (0-{self.num_steps-1})")

        step = self.pattern.steps[step_index]
        step.activate(clip_id=clip_id, velocity=velocity)

        logger.info(
            f"Step {step_index} activated: clip={clip_id}, "
            f"time={self.get_step_time(step_index):.3f}s"
        )

        return step

    def deactivate_step(self, step_index: int) -> None:
        """
        Deactivate step and clear clip assignment.

        Args:
            step_index: Step position (0-based)

        Raises:
            ValueError: If step_index out of range
        """
        if not 0 <= step_index < self.num_steps:
            raise ValueError(f"Step index {step_index} out of range (0-{self.num_steps-1})")

        step = self.pattern.steps[step_index]
        step.deactivate()

        logger.info(f"Step {step_index} deactivated")

    def clear_pattern(self) -> int:
        """
        Clear all active steps in pattern.

        Returns:
            Number of steps cleared
        """
        count = 0
        for step in self.pattern.steps:
            if step.active:
                step.deactivate()
                count += 1

        logger.info(f"Pattern cleared: {count} steps deactivated")
        return count

    def generate_cuts(
        self, cut_duration: float | None = None, extend_to_next_step: bool = True
    ) -> list[CutListEntry]:
        """
        Generate CutListEntry objects from active steps.

        This is the core Task 27 function: converts step pattern to cut list.

        Args:
            cut_duration: Optional fixed cut duration in seconds
                         If None, extends to next active step or step duration
            extend_to_next_step: If True, extend cuts to next active step
                                If False, use cut_duration or step_duration

        Returns:
            List of validated CutListEntry objects

        Raises:
            ValueError: If no active steps or cuts would overlap
        """
        active_steps = self.pattern.active_steps

        if not active_steps:
            raise ValueError("Cannot generate cuts: no active steps in pattern")

        cuts: list[CutListEntry] = []
        step_duration = self.get_step_duration()

        for i, step in enumerate(active_steps):
            start_time = self.get_step_time(step.index)

            # Calculate end_time
            if cut_duration is not None:
                # Use fixed duration
                end_time = start_time + cut_duration
            elif extend_to_next_step and i < len(active_steps) - 1:
                # Extend to next active step
                next_step = active_steps[i + 1]
                end_time = self.get_step_time(next_step.index)
            else:
                # Use step duration
                end_time = start_time + step_duration

            # Create CutListEntry (Pydantic validates end > start)
            cut = CutListEntry(clip_id=step.clip_id, start_time=start_time, end_time=end_time)
            cuts.append(cut)

            logger.debug(
                f"Cut generated from step {step.index}: "
                f"{cut.clip_id} @ {cut.start_time:.3f}s-{cut.end_time:.3f}s"
            )

        logger.info(f"Generated {len(cuts)} cuts from {len(active_steps)} active steps")

        return cuts

    def set_step_resolution(self, resolution: StepResolution) -> None:
        """
        Change step resolution (affects timing of all steps).

        Args:
            resolution: New step resolution (1/1, 1/2, 1/4, 1/8, 1/16)
        """
        self.step_resolution = resolution
        self.pattern.step_resolution = resolution

        logger.info(f"Step resolution changed to {resolution}")

    def resize_grid(self, new_num_steps: int) -> None:
        """
        Resize grid (add or remove steps).

        Args:
            new_num_steps: New number of steps

        Raises:
            ValueError: If new_num_steps < 1 or > 128
        """
        if not 1 <= new_num_steps <= 128:
            raise ValueError(f"num_steps must be 1-128, got {new_num_steps}")

        current_steps = len(self.pattern.steps)

        if new_num_steps > current_steps:
            # Add new steps
            for i in range(current_steps, new_num_steps):
                self.pattern.steps.append(Step(index=i))
            logger.info(f"Grid expanded: {current_steps} -> {new_num_steps} steps")

        elif new_num_steps < current_steps:
            # Remove steps from end
            removed_active = sum(1 for step in self.pattern.steps[new_num_steps:] if step.active)
            self.pattern.steps = self.pattern.steps[:new_num_steps]
            logger.info(
                f"Grid reduced: {current_steps} -> {new_num_steps} steps "
                f"({removed_active} active steps removed)"
            )

        self.num_steps = new_num_steps

    def get_pattern_info(self) -> dict:
        """
        Get comprehensive pattern information.

        Returns:
            Dictionary with pattern metadata
        """
        active_steps = self.pattern.active_steps

        return {
            "name": self.pattern.name,
            "num_steps": self.num_steps,
            "step_resolution": self.step_resolution,
            "step_duration": self.get_step_duration(),
            "active_steps": len(active_steps),
            "total_duration": self.get_step_time(self.num_steps),
            "bpm": self.beatgrid.bpm,
            "beatgrid_offset": self.beatgrid.beatgrid_offset,
            "active_step_indices": [s.index for s in active_steps],
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        active_count = len(self.pattern.active_steps)
        return (
            f"StepGridSequencer(steps={self.num_steps}, "
            f"active={active_count}, resolution={self.step_resolution}, "
            f"BPM={self.beatgrid.bpm:.1f})"
        )
