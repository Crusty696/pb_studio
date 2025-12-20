"""
Pydantic Models for PB_studio Pacing Engine

Production-ready data models for cut list (blueprint) management with strict validation.

Models:
- CutListEntry: Single hard cut in the timeline
- AudioTrackReference: Audio track metadata for beat synchronization
- PacingBlueprint: Complete cut list structure with validation

Features:
- Strict validation (end > start, no overlaps, chronological order)
- JSON serialization/deserialization
- SQLAlchemy 2.0 integration (from_attributes=True)
- Type-safe with Python 3.12 type hints
- Production-ready error handling

Usage:
    from pb_studio.pacing import CutListEntry, PacingBlueprint

    # Create cut list entry
    cut = CutListEntry(
        clip_id="clip_001",
        start_time=0.0,
        end_time=5.5
    )

    # Create blueprint
    blueprint = PacingBlueprint(
        name="Main Edit",
        total_duration=10.0,
        cuts=[
            CutListEntry(clip_id="clip_001", start_time=0.0, end_time=5.0),
            CutListEntry(clip_id="clip_002", start_time=5.0, end_time=10.0)
        ]
    )

    # Export to JSON
    json_str = blueprint.model_dump_json(indent=2)

    # Import from JSON
    restored = PacingBlueprint.model_validate_json(json_str)
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator
from typing_extensions import Self


@dataclass
class PacingCut:
    """
    Ein einzelner Schnitt in der Pacing-Schnittliste.

    Attributes:
        time: Zeitpunkt des Schnitts in Sekunden
        trigger_type: Auslösender Trigger-Typ ('beat', 'onset', etc.)
        strength: Skalierte Trigger-Stärke (0.0-1.0)
        raw_strength: Original Trigger-Stärke vor Scaling (0.0-1.0)
        segment_type: Song-Segment-Typ (intro, verse, chorus, drop, outro)
        active: Ob der Cut aktiv ist (Standard: True)
    """

    time: float
    trigger_type: str
    strength: float
    raw_strength: float
    segment_type: str | None = None
    active: bool = True


# BUGFIX #16: Make tolerance configurable as a module constant
# Default: 10ms tolerance for floating-point precision in duration validation
DURATION_TOLERANCE_SECONDS = 0.01  # Can be overridden by setting this before validation


class CutListEntry(BaseModel):
    """
    Represents a single hard cut in the video timeline.

    A hard cut is an instantaneous transition between two video clips
    with no effects, transitions, or blending.

    Validates:
    - Video clip ID is non-empty string
    - Start/end times are non-negative with float precision
    - End time must be strictly greater than start time

    Attributes:
        clip_id: Reference to video clip identifier
        start_time: Cut start time in seconds (float precision)
        end_time: Cut end time in seconds (float precision)
    """

    # Pydantic V2 configuration
    model_config = ConfigDict(
        # Production settings
        validate_assignment=True,  # Re-validate on field assignment
        str_strip_whitespace=True,  # Auto-strip whitespace from strings
        str_min_length=1,  # Prevent empty strings
        # JSON serialization settings
        ser_json_timedelta="float",  # Serialize timedeltas as floats
        use_enum_values=True,  # Use enum values in serialization
        # SQLAlchemy integration
        from_attributes=True,  # Enable ORM mode for SQLAlchemy
        # Schema generation
        json_schema_extra={
            "example": {
                "clip_id": "clip_001",
                "start_time": 0.0,
                "end_time": 5.25,
            }
        },
    )

    # Field definitions with constraints
    clip_id: Annotated[
        str,
        Field(
            min_length=1,
            max_length=255,
            description="Reference to video clip identifier",
            examples=["clip_001", "intro_scene", "action_cut_01"],
        ),
    ]

    start_time: Annotated[
        float,
        Field(
            ge=0.0,  # Greater than or equal to 0
            description="Cut start time in seconds (float precision)",
            examples=[0.0, 1.5, 10.25],
        ),
    ]

    end_time: Annotated[
        float,
        Field(
            ge=0.0,
            description="Cut end time in seconds (float precision)",
            examples=[5.0, 10.75, 30.5],
        ),
    ]

    # Optional metadata dict for additional info (file_path, strength, etc.)
    metadata: dict[str, Any] | None = Field(
        default=None, description="Optional metadata for the cut entry"
    )

    # Field validator: Enforce end > start
    @field_validator("end_time", mode="after")
    @classmethod
    def validate_end_after_start(cls, v: float, info: ValidationInfo) -> float:
        """
        Ensure end_time is strictly greater than start_time.

        Uses 'after' mode for type-safe validation (runs after Pydantic's validation).

        Args:
            v: The end_time value to validate
            info: Validation context with access to other field values

        Returns:
            Validated end_time value

        Raises:
            ValueError: If end_time <= start_time or start_time is missing
        """
        # BUGFIX #15: Raise error if start_time is missing (required for validation)
        if "start_time" not in info.data:
            raise ValueError(
                "start_time must be set before end_time can be validated. "
                "Ensure start_time is provided in the model constructor."
            )

        start = info.data["start_time"]
        if v <= start:
            raise ValueError(f"end_time ({v}s) must be greater than start_time ({start}s)")
        return v

    # Computed properties for convenience
    @property
    def duration(self) -> float:
        """Calculate cut duration in seconds."""
        return self.end_time - self.start_time

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"CutListEntry(clip_id={self.clip_id!r}, "
            f"start={self.start_time:.2f}s, end={self.end_time:.2f}s, "
            f"duration={self.duration:.2f}s)"
        )


class AudioTrackReference(BaseModel):
    """
    Audio track metadata for beat synchronization.

    Contains information about the audio track to enable beat-synchronized
    video editing (cuts on beats/downbeats).

    Attributes:
        track_id: Audio track identifier
        bpm: Beats per minute (tempo)
        beatgrid_offset: Beatgrid offset in seconds (first downbeat position)
    """

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {"track_id": "audio_001", "bpm": 140.0, "beatgrid_offset": 0.0}
        },
    )

    track_id: Annotated[
        str, Field(min_length=1, max_length=255, description="Audio track identifier")
    ]

    bpm: Annotated[
        float,
        Field(
            gt=0.0,
            le=300.0,  # Reasonable BPM range (60-300 is typical)
            description="Beats per minute (tempo)",
            examples=[120.0, 140.5, 174.0],
        ),
    ]

    beatgrid_offset: Annotated[
        float,
        Field(
            ge=0.0, description="Beatgrid offset in seconds (first downbeat position)", default=0.0
        ),
    ]

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"AudioTrackReference(track_id={self.track_id!r}, "
            f"bpm={self.bpm:.1f}, offset={self.beatgrid_offset:.2f}s)"
        )


class PacingBlueprint(BaseModel):
    """
    Complete cut list structure with validation for chronological order
    and non-overlapping cuts.

    The blueprint represents a complete video edit timeline with hard cuts only.
    All cuts are validated to ensure:
    - Chronological order (sorted by start_time)
    - No overlaps in timeline
    - Total duration consistency

    Attributes:
        name: Blueprint name/title
        created_at: Timestamp of blueprint creation
        total_duration: Total duration of all cuts in seconds
        cuts: List of cut list entries (chronologically ordered)
        audio_track: Optional audio track reference for beat sync
        description: Optional blueprint description
    """

    model_config = ConfigDict(
        # Production settings
        validate_assignment=True,
        str_strip_whitespace=True,
        # SQLAlchemy integration
        from_attributes=True,  # Critical for SQLAlchemy ORM objects
        # Serialization settings
        use_enum_values=True,
        # Schema customization
        json_schema_extra={
            "example": {
                "name": "Action Sequence Edit",
                "created_at": "2024-01-15T10:30:00Z",
                "total_duration": 120.5,
                "cuts": [
                    {"clip_id": "clip_001", "start_time": 0.0, "end_time": 5.0},
                    {"clip_id": "clip_002", "start_time": 5.0, "end_time": 10.5},
                ],
                "audio_track": {"track_id": "audio_001", "bpm": 140.0, "beatgrid_offset": 0.0},
            }
        },
    )

    # Metadata fields
    name: Annotated[
        str,
        Field(
            min_length=1,
            max_length=255,
            description="Blueprint name/title",
            examples=["Main Edit", "Director's Cut", "Fast-Paced Version"],
        ),
    ]

    created_at: Annotated[
        datetime,
        Field(
            description="Timestamp of blueprint creation (UTC)",
            default_factory=lambda: datetime.now(timezone.utc),
        ),
    ]

    total_duration: Annotated[
        float, Field(ge=0.0, description="Total duration of all cuts in seconds")
    ]

    # Core data
    cuts: Annotated[
        list[CutListEntry],
        Field(
            description="List of cut list entries (chronologically ordered)", default_factory=list
        ),
    ]

    audio_track: Annotated[
        AudioTrackReference | None,
        Field(default=None, description="Optional audio track reference for beat sync"),
    ]

    # Optional metadata
    description: Annotated[
        str | None,
        Field(default=None, max_length=1000, description="Optional blueprint description"),
    ]

    # Field validator: Ensure cuts are chronologically ordered
    @field_validator("cuts", mode="after")
    @classmethod
    def validate_chronological_order(cls, cuts: list[CutListEntry]) -> list[CutListEntry]:
        """
        Validate that cuts are in chronological order by start_time.

        Uses 'after' mode for type-safe validation.

        Args:
            cuts: List of cut list entries

        Returns:
            Validated cuts list

        Raises:
            ValueError: If cuts are not in chronological order
        """
        if len(cuts) < 2:
            return cuts

        for i in range(len(cuts) - 1):
            current = cuts[i]
            next_cut = cuts[i + 1]

            if current.start_time > next_cut.start_time:
                raise ValueError(
                    f"Cuts must be in chronological order: "
                    f"cut at index {i} (start={current.start_time}s) "
                    f"comes after cut at index {i + 1} (start={next_cut.start_time}s)"
                )

        return cuts

    # Model validator: Check for overlapping cuts
    @model_validator(mode="after")
    def validate_no_overlapping_cuts(self) -> Self:
        """
        Validate that no cuts overlap in the timeline.

        Uses 'after' mode to work with fully validated CutListEntry objects.
        This runs after all field validators complete.

        Returns:
            Self (validated blueprint)

        Raises:
            ValueError: If any cuts overlap in timeline
        """
        cuts = self.cuts

        if len(cuts) < 2:
            return self

        for i in range(len(cuts) - 1):
            current = cuts[i]
            next_cut = cuts[i + 1]

            # Check if current cut's end overlaps with next cut's start
            if current.end_time > next_cut.start_time:
                raise ValueError(
                    f"Overlapping cuts detected: "
                    f"Cut {i} (clip_id={current.clip_id}, {current.start_time}s-{current.end_time}s) "
                    f"overlaps with Cut {i + 1} (clip_id={next_cut.clip_id}, "
                    f"{next_cut.start_time}s-{next_cut.end_time}s)"
                )

        return self

    # Model validator: Validate total_duration consistency
    @model_validator(mode="after")
    def validate_total_duration(self) -> Self:
        """
        Validate that total_duration matches the timeline span.

        For manual triggers with gaps, total_duration represents the timeline span
        (from 0 to max end_time), not the sum of cut durations.

        Allows small floating-point tolerance (10ms).

        Returns:
            Self (validated blueprint)

        Raises:
            ValueError: If total_duration doesn't match timeline span
        """
        if not self.cuts:
            if self.total_duration != 0.0:
                raise ValueError(
                    f"total_duration must be 0.0 when cuts list is empty, got {self.total_duration}"
                )
            return self

        # Calculate timeline span (max end_time)
        calculated_duration = max(cut.end_time for cut in self.cuts)
        # BUGFIX #16: Use configurable tolerance constant
        tolerance = DURATION_TOLERANCE_SECONDS

        if abs(calculated_duration - self.total_duration) > tolerance:
            raise ValueError(
                f"total_duration ({self.total_duration}s) does not match "
                f"timeline span ({calculated_duration:.2f}s)"
            )

        return self

    # Business rule validator: Reasonable limits
    @model_validator(mode="after")
    def validate_business_rules(self) -> Self:
        """
        Apply business rules validation.

        Rules:
        - Maximum 10000 cuts per blueprint
        - Maximum 24 hours total duration

        Returns:
            Self (validated blueprint)

        Raises:
            ValueError: If business rules are violated
        """
        # Business rule: Maximum 10000 cuts per blueprint
        MAX_CUTS = 10000
        if len(self.cuts) > MAX_CUTS:
            raise ValueError(f"Maximum {MAX_CUTS} cuts allowed, got {len(self.cuts)}")

        # Business rule: Total duration must not exceed 24 hours
        MAX_DURATION = 24 * 60 * 60  # 24 hours in seconds
        if self.total_duration > MAX_DURATION:
            raise ValueError(
                f"Total duration exceeds maximum of {MAX_DURATION}s (got {self.total_duration}s)"
            )

        return self

    # Computed properties
    @property
    def cut_count(self) -> int:
        """Number of cuts in the blueprint."""
        return len(self.cuts)

    @property
    def average_cut_duration(self) -> float:
        """Average duration of cuts in seconds."""
        if not self.cuts:
            return 0.0
        return self.total_duration / len(self.cuts)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"PacingBlueprint(name={self.name!r}, "
            f"cuts={len(self.cuts)}, "
            f"total_duration={self.total_duration:.2f}s, "
            f"avg_cut={self.average_cut_duration:.2f}s)"
        )


# Helper functions for JSON serialization


def export_blueprint_to_json(blueprint: PacingBlueprint, pretty: bool = True) -> str:
    """
    Export blueprint to JSON string.

    Args:
        blueprint: PacingBlueprint to export
        pretty: Enable pretty-printing with indentation

    Returns:
        JSON string representation

    Example:
        json_str = export_blueprint_to_json(blueprint, pretty=True)
    """
    return blueprint.model_dump_json(
        exclude_none=True,  # Omit None values
        indent=2 if pretty else None,
        by_alias=False,  # Use Python field names
    )


def export_blueprint_to_dict(blueprint: PacingBlueprint) -> dict:
    """
    Export blueprint to dictionary (JSON-serializable).

    Args:
        blueprint: PacingBlueprint to export

    Returns:
        Dictionary with JSON-compatible types (datetime → ISO string)

    Example:
        data = export_blueprint_to_dict(blueprint)
        # Save to database as JSON
    """
    return blueprint.model_dump(exclude_none=True, mode="json")  # Convert to JSON-compatible types


def import_blueprint_from_json(json_str: str) -> PacingBlueprint:
    """
    Import blueprint from JSON string.

    Automatically validates all constraints (chronological order, no overlaps, etc.).

    Args:
        json_str: JSON string representation

    Returns:
        Validated PacingBlueprint

    Raises:
        ValidationError: If JSON is invalid or violates constraints

    Example:
        blueprint = import_blueprint_from_json(json_data)
    """
    return PacingBlueprint.model_validate_json(json_str)


def import_blueprint_from_dict(data: dict) -> PacingBlueprint:
    """
    Import blueprint from dictionary.

    Automatically validates all constraints.

    Args:
        data: Dictionary representation

    Returns:
        Validated PacingBlueprint

    Raises:
        ValidationError: If dict is invalid or violates constraints

    Example:
        blueprint = import_blueprint_from_dict(db_data)
    """
    return PacingBlueprint.model_validate(data)
