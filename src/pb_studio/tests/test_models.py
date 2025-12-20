
import pytest
import os
import shutil
from pathlib import Path
from pb_studio.pacing.pacing_models import CutListEntry, PacingBlueprint

class TestPacingModels:
    def test_cut_list_entry_validation(self):
        """Test valid and invalid CutListEntry creation."""
        # Valid entry
        entry = CutListEntry(clip_id="clip_1", start_time=0.0, end_time=5.0)
        assert entry.clip_id == "clip_1"
        assert entry.duration == 5.0

        # Invalid entry (end <= start)
        with pytest.raises(ValueError):
            CutListEntry(clip_id="clip_1", start_time=5.0, end_time=5.0)

        # Invalid entry (negative time)
        with pytest.raises(ValueError):
             CutListEntry(clip_id="clip_1", start_time=-1.0, end_time=5.0)

    def test_blueprint_validation(self):
        """Test PacingBlueprint validation logic."""
        cuts = [
            CutListEntry(clip_id="c1", start_time=0.0, end_time=5.0),
            CutListEntry(clip_id="c2", start_time=5.0, end_time=10.0)
        ]

        # Valid blueprint
        bp = PacingBlueprint(
            name="Test",
            total_duration=10.0,
            cuts=cuts
        )
        assert len(bp.cuts) == 2

        # Invalid overlap
        overlap_cuts = [
            CutListEntry(clip_id="c1", start_time=0.0, end_time=6.0),
            CutListEntry(clip_id="c2", start_time=5.0, end_time=10.0)
        ]
        with pytest.raises(ValueError):
            PacingBlueprint(name="Overlap", total_duration=10.0, cuts=overlap_cuts)
