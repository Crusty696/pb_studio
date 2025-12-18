"""
PB_studio Database Package

Dual-Database-Architektur:
- SQLite: Persistenz (Projekte, Tracks, Clips, Beatgrids)
- DuckDB: Analytics (Statistiken, Nutzungsdaten)
"""

from .connection import (
    DatabaseManager,
    DuckDBManager,
    get_db_manager,
    get_db_session,
    get_duckdb_manager,
)
from .crud import (
    create_audio_track,
    create_beatgrid,
    create_project,
    create_video_clip,
    delete_project,
    get_all_projects,
    get_audio_tracks_by_project,
    get_beatgrid_by_audio_track,
    get_project,
    get_video_clips_by_project,
    update_audio_track,
    update_project,
)
from .models import AudioTrack, Base, BeatGrid, PacingBlueprint, Project, VideoClip
from .schema import (
    backup_database,
    get_database_stats,
    init_analytics_database,
    init_database,
    verify_database_schema,
)

__all__ = [
    # Connection
    "DatabaseManager",
    "DuckDBManager",
    "get_db_manager",
    "get_db_session",
    "get_duckdb_manager",
    # Models
    "Base",
    "Project",
    "AudioTrack",
    "VideoClip",
    "BeatGrid",
    "PacingBlueprint",
    # Schema
    "init_database",
    "init_analytics_database",
    "verify_database_schema",
    "backup_database",
    "get_database_stats",
    # CRUD
    "create_project",
    "get_project",
    "get_all_projects",
    "update_project",
    "delete_project",
    "create_audio_track",
    "get_audio_tracks_by_project",
    "update_audio_track",
    "create_video_clip",
    "get_video_clips_by_project",
    "create_beatgrid",
    "get_beatgrid_by_audio_track",
]
