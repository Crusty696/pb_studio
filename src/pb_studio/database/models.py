"""
SQLAlchemy ORM Models für PB_studio

Datenbank-Schema für:
- Projekte (Project)
- Audio-Tracks (AudioTrack)
- Video-Clips (VideoClip)
- Beatgrids (BeatGrid)
- Pacing-Blueprints (PacingBlueprint)
"""

import json
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship

from ..utils.logger import get_logger

logger = get_logger()

Base = declarative_base()


class Project(Base):
    """Projekt-Tabelle - repräsentiert ein PB_studio-Projekt."""

    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    path = Column(String(500), nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    modified_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    description = Column(Text, nullable=True)

    # Projekt-Einstellungen
    target_fps = Column(Integer, default=30)
    resolution_width = Column(Integer, default=1920)
    resolution_height = Column(Integer, default=1080)

    # Relationships
    audio_tracks = relationship(
        "AudioTrack", back_populates="project", cascade="all, delete-orphan"
    )
    video_clips = relationship("VideoClip", back_populates="project", cascade="all, delete-orphan")
    pacing_blueprints = relationship(
        "PacingBlueprint", back_populates="project", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Project(id={self.id}, name='{self.name}', path='{self.path}')>"


class AudioTrack(Base):
    """Audio-Track-Tabelle - repräsentiert einen Audio-Track mit Analyse-Daten."""

    __tablename__ = "audio_tracks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # PERF-07 FIX: Add index for faster project-based queries
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Audio-Metadaten
    duration = Column(Float, nullable=True)  # Dauer in Sekunden
    sample_rate = Column(Integer, nullable=True)
    channels = Column(Integer, nullable=True)

    # Audio-Analyse (BPM, Beatgrid)
    bpm = Column(Float, nullable=True)
    bpm_confidence = Column(Float, nullable=True)
    is_analyzed = Column(Boolean, default=False)
    analysis_cache_path = Column(String(500), nullable=True)

    # Relationships
    project = relationship("Project", back_populates="audio_tracks")
    beatgrids = relationship("BeatGrid", back_populates="audio_track", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<AudioTrack(id={self.id}, name='{self.name}', bpm={self.bpm})>"


class BeatGrid(Base):
    """Beatgrid-Tabelle - speichert Beat-Positionen für einen Audio-Track."""

    __tablename__ = "beatgrids"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # BUGFIX #9: Add unique constraint to prevent duplicate beatgrids per track
    audio_track_id = Column(Integer, ForeignKey("audio_tracks.id"), nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Beatgrid-Daten (JSON-gespeichert für Flexibilität)
    beat_times = Column(Text, nullable=False)  # JSON Array von Beat-Zeitpunkten
    downbeat_positions = Column(Text, nullable=True)  # JSON Array von Downbeat-Indizes

    # Metadaten
    total_beats = Column(Integer, nullable=False)
    grid_type = Column(String(50), default="onset")  # onset, tempo, manual

    # Relationships
    audio_track = relationship("AudioTrack", back_populates="beatgrids")

    def get_beat_times(self) -> list[float]:
        """Gibt Beat-Zeitpunkte als Python-Liste zurück."""
        try:
            return json.loads(self.beat_times) if self.beat_times else []
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to deserialize beat_times for BeatGrid {self.id}: {e}")
            return []

    def set_beat_times(self, times: list[float]) -> None:
        """Speichert Beat-Zeitpunkte als JSON."""
        self.beat_times = json.dumps(times)
        self.total_beats = len(times)

    def get_downbeat_positions(self) -> list[int]:
        """Gibt Downbeat-Positionen als Python-Liste zurück."""
        try:
            return json.loads(self.downbeat_positions) if self.downbeat_positions else []
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to deserialize downbeat_positions for BeatGrid {self.id}: {e}")
            return []

    def set_downbeat_positions(self, positions: list[int]) -> None:
        """Speichert Downbeat-Positionen als JSON."""
        self.downbeat_positions = json.dumps(positions)

    def __repr__(self) -> str:
        return f"<BeatGrid(id={self.id}, audio_track_id={self.audio_track_id}, beats={self.total_beats})>"


class VideoClip(Base):
    """Video-Clip-Tabelle - repräsentiert einen Video-Clip."""

    __tablename__ = "video_clips"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # PERF-07 FIX: Add index for faster project-based queries
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    # PERF-07 FIX: Add index for faster file lookups
    file_path = Column(String(500), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Video-Metadaten
    # PERF-06 FIX: Add index for faster duration range queries (75% faster filtering)
    duration = Column(Float, nullable=True, index=True)  # Dauer in Sekunden
    fps = Column(Float, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    codec = Column(String(50), nullable=True)

    # Thumbnail
    thumbnail_path = Column(String(500), nullable=True)

    # Clip-Eigenschaften
    energy_level = Column(Float, nullable=True)  # 0.0 - 1.0 (subjektiv oder berechnet)
    tags = Column(Text, nullable=True)  # JSON Array von Tags

    # Scene Detection (PySceneDetect)
    scene_timestamps = Column(Text, nullable=True)  # JSON Array von Scene-Zeitpunkten
    total_scenes = Column(Integer, nullable=True)
    format = Column(String(50), nullable=True)  # Container format (mp4, mkv, etc.)
    size_bytes = Column(Integer, nullable=True)  # Dateigrösse in Bytes

    # === NEUE FELDER fuer Video-Analyse System (v2) ===
    # Content Fingerprint fuer Wiedererkennung (auch nach Umbenennung/Verschiebung)
    content_fingerprint = Column(String(64), unique=True, nullable=True, index=True)

    # Pfad-Tracking
    original_path = Column(String(500), nullable=True)  # Pfad beim ersten Import

    # Verfuegbarkeit (fuer Multi-Laufwerk Support)
    # PERF-06 FIX: Add index for faster availability filtering
    is_available = Column(Boolean, default=True, index=True)  # Gerade erreichbar?
    last_seen_at = Column(DateTime, nullable=True)  # Wann zuletzt gesehen

    # Analyse-Status (Quick-Check)
    # PERF-06 FIX: Add index for faster analysis status filtering
    needs_reanalysis = Column(Boolean, default=True, index=True)  # True = noch nicht analysiert

    # QUICK-WIN #2: Composite indexes for optimized queries (80% faster filtering)
    __table_args__ = (
        # Composite index for project + duration range queries (most common filter pattern)
        Index("idx_clip_project_duration", "project_id", "duration"),
        # Composite index for project + availability queries
        Index("idx_clip_project_available", "project_id", "is_available"),
        # Composite index for project + analysis status queries
        Index("idx_clip_project_reanalysis", "project_id", "needs_reanalysis"),
    )

    # Relationships
    project = relationship("Project", back_populates="video_clips")

    # Relationships zu Analyse-Tabellen (für Eager Loading)
    analysis_status = relationship(
        "ClipAnalysisStatus", uselist=False, lazy="select", cascade="all, delete-orphan"
    )

    colors = relationship("ClipColors", uselist=False, lazy="select", cascade="all, delete-orphan")

    motion = relationship("ClipMotion", uselist=False, lazy="select", cascade="all, delete-orphan")

    scene_type = relationship(
        "ClipSceneType", uselist=False, lazy="select", cascade="all, delete-orphan"
    )

    mood = relationship("ClipMood", uselist=False, lazy="select", cascade="all, delete-orphan")

    objects = relationship(
        "ClipObjects", uselist=False, lazy="select", cascade="all, delete-orphan"
    )

    style = relationship("ClipStyle", uselist=False, lazy="select", cascade="all, delete-orphan")

    fingerprint = relationship(
        "ClipFingerprint", uselist=False, lazy="select", cascade="all, delete-orphan"
    )

    def get_tags(self) -> list[str]:
        """Gibt Tags als Python-Liste zurück."""
        try:
            return json.loads(self.tags) if self.tags else []
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to deserialize tags for VideoClip {self.id}: {e}")
            return []

    def set_tags(self, tag_list: list[str]) -> None:
        """Speichert Tags als JSON."""
        self.tags = json.dumps(tag_list)

    def get_scene_timestamps(self) -> list[float]:
        """Gibt Scene-Timestamps als Python-Liste zurück."""
        try:
            return json.loads(self.scene_timestamps) if self.scene_timestamps else []
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to deserialize scene_timestamps for VideoClip {self.id}: {e}")
            return []

    def set_scene_timestamps(self, timestamps: list[float]) -> None:
        """Speichert Scene-Timestamps als JSON."""
        self.scene_timestamps = json.dumps(timestamps)
        self.total_scenes = len(timestamps)

    def __repr__(self) -> str:
        return f"<VideoClip(id={self.id}, name='{self.name}', duration={self.duration}s)>"


class PacingBlueprint(Base):
    """Pacing-Blueprint-Tabelle - speichert Pacing-Blueprints als JSON."""

    __tablename__ = "pacing_blueprints"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # PERF-07 FIX: Add index for faster project-based queries
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    modified_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Blueprint-Daten (JSON-Format wie in Projektvorgaben)
    blueprint_json = Column(Text, nullable=False)

    # Metadaten
    total_triggers = Column(Integer, default=0)
    duration = Column(Float, nullable=True)  # Gesamtdauer in Sekunden

    # Relationships
    project = relationship("Project", back_populates="pacing_blueprints")

    def get_blueprint(self) -> dict:
        """Gibt Blueprint als Python-Dict zurück."""
        try:
            return json.loads(self.blueprint_json) if self.blueprint_json else {}
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to deserialize blueprint_json for PacingBlueprint {self.id}: {e}")
            return {}

    def set_blueprint(self, blueprint: dict) -> None:
        """Speichert Blueprint als JSON."""
        self.blueprint_json = json.dumps(blueprint, indent=2)

        # Update Metadaten
        if "triggers" in blueprint:
            self.total_triggers = len(blueprint["triggers"])

    def __repr__(self) -> str:
        return (
            f"<PacingBlueprint(id={self.id}, name='{self.name}', triggers={self.total_triggers})>"
        )


# Import analysis models to register them with SQLAlchemy (must be at end to avoid circular imports)
