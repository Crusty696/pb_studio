import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from pb_studio.database.models import Base, Project, AudioTrack, BeatGrid, VideoClip, PacingBlueprint
# Import analysis models to ensure they are registered in the mapper
from pb_studio.database.models_analysis import (
    ClipAnalysisStatus,
    ClipColors,
    ClipMotion,
    ClipSceneType,
    ClipMood,
    ClipObjects,
    ClipStyle,
    ClipFingerprint
)

# Use an in-memory SQLite database for testing
@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

def test_create_project(db_session, tmp_path):
    """Test creating a simple project."""
    project_path = tmp_path / "test_project"
    project = Project(name="Test Project", path=str(project_path))
    db_session.add(project)
    db_session.commit()

    assert project.id is not None
    assert project.name == "Test Project"
    assert project.target_fps == 30  # Default value

def test_audio_track_creation(db_session, tmp_path):
    """Test creating an audio track linked to a project."""
    project_path = tmp_path / "audio_proj"
    project = Project(name="Audio Project", path=str(project_path))
    db_session.add(project)
    db_session.commit()

    audio_path = tmp_path / "music.mp3"
    track = AudioTrack(
        project_id=project.id,
        name="Test Track",
        file_path=str(audio_path),
        duration=120.5,
        bpm=128.0
    )
    db_session.add(track)
    db_session.commit()

    assert track.id is not None
    assert track.project == project
    assert track.bpm == 128.0

def test_video_clip_json_fields(db_session, tmp_path):
    """Test JSON serialization for VideoClip tags and scenes."""
    project_path = tmp_path / "video_proj"
    project = Project(name="Video Project", path=str(project_path))
    db_session.add(project)
    db_session.commit()

    video_path = tmp_path / "video.mp4"
    clip = VideoClip(
        project_id=project.id,
        name="Test Clip",
        file_path=str(video_path),
        duration=10.0
    )

    # Test Tags
    tags = ["action", "outdoor"]
    clip.set_tags(tags)

    # Test Scene Timestamps
    scenes = [0.0, 3.5, 8.2]
    clip.set_scene_timestamps(scenes)

    db_session.add(clip)
    db_session.commit()

    # Retrieve and verify
    fetched_clip = db_session.query(VideoClip).filter_by(name="Test Clip").first()
    assert fetched_clip.get_tags() == tags
    assert fetched_clip.get_scene_timestamps() == scenes
    assert fetched_clip.total_scenes == 3

def test_beatgrid_json_fields(db_session, tmp_path):
    """Test JSON handling in BeatGrid."""
    project_path = tmp_path / "grid_proj"
    project = Project(name="Grid Project", path=str(project_path))
    db_session.add(project)
    db_session.commit()

    track = AudioTrack(project_id=project.id, name="Track", file_path="x")
    db_session.add(track)
    db_session.commit()

    beats = [0.5, 1.0, 1.5, 2.0]
    grid = BeatGrid(
        audio_track_id=track.id,
        total_beats=len(beats),
        beat_times="[]" # Initial placeholder
    )
    grid.set_beat_times(beats)

    db_session.add(grid)
    db_session.commit()

    fetched_grid = db_session.query(BeatGrid).first()
    assert fetched_grid.get_beat_times() == beats
    assert fetched_grid.total_beats == 4

def test_pacing_blueprint_json(db_session, tmp_path):
    """Test PacingBlueprint JSON storage."""
    project_path = tmp_path / "bp_proj"
    project = Project(name="Blueprint Project", path=str(project_path))
    db_session.add(project)
    db_session.commit()

    bp_data = {
        "triggers": [{"time": 1.0, "type": "cut"}, {"time": 2.0, "type": "fade"}],
        "meta": {"version": 1}
    }

    bp = PacingBlueprint(
        project_id=project.id,
        name="Test Blueprint",
        blueprint_json="{}"
    )
    bp.set_blueprint(bp_data)

    db_session.add(bp)
    db_session.commit()

    fetched_bp = db_session.query(PacingBlueprint).first()
    assert fetched_bp.get_blueprint() == bp_data
    assert fetched_bp.total_triggers == 2
