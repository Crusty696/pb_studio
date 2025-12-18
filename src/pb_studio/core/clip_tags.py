"""
Clip Tags System für PB_studio

Ermöglicht Tag-basierte Organisation von Clips.

Author: PB_studio Development Team
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ClipTag:
    """
    Represents a tag for clip organization.

    Attributes:
        name: Unique tag name
        color: Hex color code (e.g., "#FF5733")
        description: Optional tag description
    """

    name: str
    color: str
    description: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ClipTag":
        """Create from dictionary."""
        return cls(**data)


class TagManager:
    """
    Manages clip tags and assignments.

    Features:
        - Create/delete tags with colors
        - Assign tags to clips
        - Query clips by tags
        - Persist to JSON

    Example:
        >>> tm = TagManager()
        >>> tm.create_tag("action", "#FF0000", "High-energy clips")
        >>> tm.assign_tag(clip_id=1, tag_name="action")
        >>> clips = tm.get_clips_by_tag("action")
        >>> print(clips)
        [1]
    """

    def __init__(self, data_file: Path | None = None):
        """
        Initialize Tag Manager.

        Args:
            data_file: Path to JSON file for persistence
        """
        self.data_file = data_file or Path(".taskmaster/data/clip_tags.json")
        self.tags: dict[str, ClipTag] = {}
        self.assignments: dict[int, set[str]] = {}  # clip_id -> {tag_names}
        self._load()

    def create_tag(self, name: str, color: str, description: str = "") -> ClipTag:
        """
        Create a new tag.

        Args:
            name: Unique tag name
            color: Hex color code (e.g., "#FF5733")
            description: Optional description

        Returns:
            Created ClipTag instance

        Raises:
            ValueError: If tag already exists

        Example:
            >>> tm = TagManager()
            >>> tag = tm.create_tag("dramatic", "#9B59B6", "Dramatic scenes")
        """
        if name in self.tags:
            raise ValueError(f"Tag '{name}' already exists")

        tag = ClipTag(name=name, color=color, description=description)
        self.tags[name] = tag
        self._save()
        logger.info(f"Created tag: {name}")
        return tag

    def delete_tag(self, name: str) -> bool:
        """
        Delete a tag and remove all assignments.

        Args:
            name: Tag name to delete

        Returns:
            True if deleted, False if not found
        """
        if name not in self.tags:
            return False

        # Remove from all assignments
        for clip_tags in self.assignments.values():
            clip_tags.discard(name)

        del self.tags[name]
        self._save()
        logger.info(f"Deleted tag: {name}")
        return True

    def assign_tag(self, clip_id: int, tag_name: str) -> None:
        """
        Assign tag to clip.

        Args:
            clip_id: Clip identifier
            tag_name: Tag name to assign

        Raises:
            ValueError: If tag doesn't exist
        """
        if tag_name not in self.tags:
            raise ValueError(f"Tag '{tag_name}' doesn't exist")

        if clip_id not in self.assignments:
            self.assignments[clip_id] = set()

        self.assignments[clip_id].add(tag_name)
        self._save()
        logger.debug(f"Assigned tag '{tag_name}' to clip {clip_id}")

    def remove_tag(self, clip_id: int, tag_name: str) -> bool:
        """
        Remove tag from clip.

        Args:
            clip_id: Clip identifier
            tag_name: Tag name to remove

        Returns:
            True if removed, False if not assigned
        """
        if clip_id not in self.assignments:
            return False

        if tag_name in self.assignments[clip_id]:
            self.assignments[clip_id].remove(tag_name)
            self._save()
            logger.debug(f"Removed tag '{tag_name}' from clip {clip_id}")
            return True

        return False

    def get_clip_tags(self, clip_id: int) -> list[ClipTag]:
        """
        Get all tags assigned to a clip.

        Args:
            clip_id: Clip identifier

        Returns:
            List of ClipTag objects
        """
        tag_names = self.assignments.get(clip_id, set())
        return [self.tags[name] for name in tag_names if name in self.tags]

    def get_clips_by_tag(self, tag_name: str) -> list[int]:
        """
        Get all clips with specific tag.

        Args:
            tag_name: Tag name to search

        Returns:
            List of clip IDs
        """
        return [clip_id for clip_id, tags in self.assignments.items() if tag_name in tags]

    def get_all_tags(self) -> list[ClipTag]:
        """Get all available tags."""
        return list(self.tags.values())

    def _load(self) -> None:
        """Load tags and assignments from file."""
        if not self.data_file.exists():
            # Ensure directory exists
            self.data_file.parent.mkdir(parents=True, exist_ok=True)
            return

        try:
            data = json.loads(self.data_file.read_text(encoding="utf-8"))

            # Load tags
            self.tags = {
                name: ClipTag.from_dict(tag_data) for name, tag_data in data.get("tags", {}).items()
            }

            # Load assignments (convert lists to sets)
            self.assignments = {
                int(clip_id): set(tag_names)
                for clip_id, tag_names in data.get("assignments", {}).items()
            }

            logger.info(f"Loaded {len(self.tags)} tags")
        except Exception as e:
            logger.error(f"Failed to load tags: {e}")

    def _save(self) -> None:
        """Save tags and assignments to file."""
        # Ensure directory exists
        self.data_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "tags": {name: tag.to_dict() for name, tag in self.tags.items()},
            "assignments": {
                str(clip_id): list(tag_names) for clip_id, tag_names in self.assignments.items()
            },
        }

        try:
            self.data_file.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as e:
            logger.error(f"Failed to save tags: {e}")
