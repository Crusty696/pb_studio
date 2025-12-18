"""
CRUD-Operationen für PB_studio Datenbank

Bietet einfache Create, Read, Update, Delete-Funktionen für alle Entities.
Automatisches Projektordner-Management.
"""

import shutil
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session, selectinload

from ..utils.path_utils import validate_file_path
from .connection import managed_session
from .models import AudioTrack, BeatGrid, PacingBlueprint, Project, VideoClip

# PERF-08 FIX: Default pagination constants
DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 1000
from ..utils.logger import get_logger

logger = get_logger()


# ============================================================
# PROJECT CRUD
# ============================================================


def create_project(
    name: str,
    path: str,
    description: str | None = None,
    target_fps: int = 30,
    resolution: tuple[int, int] = (1920, 1080),
    session: Session | None = None,
) -> Project | None:
    """
    Erstellt ein neues Projekt mit automatischer Ordnerstruktur.

    Args:
        name: Projektname
        path: Projekt-Basispfad
        description: Projektbeschreibung (optional)
        target_fps: Ziel-FPS (default: 30)
        resolution: Auflösung als (width, height) Tuple
        session: SQLAlchemy Session (optional)

    Returns:
        Project-Objekt oder None bei Fehler
    """
    try:
        with managed_session(session) as db:
            project_path = Path(path)

            # Prüfe ob Projekt bereits existiert
            existing = db.query(Project).filter_by(path=str(project_path)).first()
            if existing:
                logger.warning(f"Projekt existiert bereits: {project_path}")
                return existing

            # Erstelle Projektordner-Struktur
            folders = ["audio", "video", "exports", "temp", "cache"]
            for folder in folders:
                (project_path / folder).mkdir(parents=True, exist_ok=True)

            logger.info(f"Projektordner erstellt: {project_path}")

            # Erstelle Projekt in Datenbank
            project = Project(
                name=name,
                path=str(project_path),
                description=description,
                target_fps=target_fps,
                resolution_width=resolution[0],
                resolution_height=resolution[1],
            )

            db.add(project)
            db.flush()  # Refresh ID without commit
            db.refresh(project)

            logger.info(f"Projekt erstellt: {name} (ID: {project.id})")
            return project

    except Exception as e:
        logger.error(f"Fehler beim Erstellen des Projekts: {e}")
        return None


def get_project(
    project_id: int, session: Session | None = None, eager_load: bool = True
) -> Project | None:
    """
    Holt ein Projekt anhand der ID.

    PERF-FIX: Eager loading standardmäßig aktiviert um N+1 Queries zu vermeiden.

    Args:
        project_id: Projekt-ID
        session: SQLAlchemy Session (optional)
        eager_load: Load relationships eagerly (default: True)

    Returns:
        Project-Objekt oder None
    """
    with managed_session(session) as db:
        query = db.query(Project).filter_by(id=project_id)

        if eager_load:
            query = query.options(
                selectinload(Project.audio_tracks),
                selectinload(Project.video_clips),
                selectinload(Project.pacing_blueprints),
            )

        return query.first()


def get_all_projects(
    session: Session | None = None,
    limit: int | None = DEFAULT_PAGE_SIZE,  # BUGFIX #8: Set default to 100
    offset: int = 0,
    eager_load: bool = False,
) -> list[Project]:
    """
    Holt alle Projekte mit optionaler Pagination.

    PERF-08 FIX: Added pagination support to prevent memory issues with large datasets.
    PERF-06 FIX: Added eager_load option to prevent N+1 queries.
    BUGFIX #8: Default limit set to 100 for consistent pagination behavior.

    Args:
        session: SQLAlchemy Session (optional)
        limit: Maximum number of results (default: 100, max: 1000)
        offset: Number of results to skip (default: 0)
        eager_load: Load relationships eagerly to prevent N+1 queries

    Returns:
        Liste von Project-Objekten
    """
    with managed_session(session) as db:
        query = db.query(Project)

        # PERF-06 FIX: Eager loading for relationships
        if eager_load:
            query = query.options(
                selectinload(Project.audio_tracks),
                selectinload(Project.video_clips),
                selectinload(Project.pacing_blueprints),
            )

        query = query.order_by(Project.modified_at.desc())

        # PERF-08 FIX: Apply pagination
        if offset > 0:
            query = query.offset(offset)
        if limit is not None:
            # Enforce maximum page size
            safe_limit = min(limit, MAX_PAGE_SIZE)
            query = query.limit(safe_limit)

        return query.all()


def update_project(project_id: int, session: Session | None = None, **kwargs) -> Project | None:
    """
    Aktualisiert Projekt-Attribute.

    Args:
        project_id: Projekt-ID
        session: SQLAlchemy Session (optional)
        **kwargs: Zu aktualisierende Attribute

    Returns:
        Aktualisiertes Project-Objekt oder None
    """
    try:
        with managed_session(session) as db:
            project = db.query(Project).filter_by(id=project_id).first()
            if not project:
                logger.warning(f"Projekt nicht gefunden: ID {project_id}")
                return None

            # Update Attribute
            for key, value in kwargs.items():
                if hasattr(project, key):
                    setattr(project, key, value)

            project.modified_at = datetime.utcnow()
            db.flush()
            db.refresh(project)

            logger.info(f"Projekt aktualisiert: {project.name} (ID: {project.id})")
            return project

    except Exception as e:
        logger.error(f"Fehler beim Aktualisieren des Projekts: {e}")
        return None


def delete_project(
    project_id: int, delete_files: bool = False, session: Session | None = None
) -> bool:
    """
    Löscht ein Projekt.

    BUG-FIX: Delete files BEFORE removing from DB to prevent race condition.

    Args:
        project_id: Projekt-ID
        delete_files: Falls True, werden auch die Projektdateien gelöscht
        session: SQLAlchemy Session (optional)

    Returns:
        True bei Erfolg, False bei Fehler
    """
    project_path = None
    project_name = None

    try:
        # BUG-FIX: First, gather project info and delete files BEFORE DB transaction
        with managed_session(session) as db:
            project = db.query(Project).filter_by(id=project_id).first()
            if not project:
                logger.warning(f"Projekt nicht gefunden: ID {project_id}")
                return False

            project_path = Path(project.path)
            project_name = project.name

        # BUG-FIX: Delete files BEFORE DB deletion to prevent race condition
        if delete_files and project_path and project_path.exists():
            try:
                shutil.rmtree(project_path)
                logger.info(f"Projektordner gelöscht: {project_path}")
            except Exception as e:
                logger.error(f"Failed to delete project files: {e}")
                # Continue with DB deletion even if file deletion fails
                # This prevents orphaned DB records

        # Now delete from database (Cascade löscht alle Relationen)
        with managed_session(session) as db:
            project = db.query(Project).filter_by(id=project_id).first()
            if project:
                db.delete(project)
                logger.info(f"Projekt gelöscht aus Datenbank: {project_name}")

        return True

    except Exception as e:
        logger.error(f"Fehler beim Löschen des Projekts: {e}")
        return False


# ============================================================
# AUDIO TRACK CRUD
# ============================================================


def create_audio_track(
    project_id: int,
    name: str,
    file_path: str,
    session: Session | None = None,
) -> AudioTrack | None:
    """
    Erstellt einen neuen Audio-Track.

    BUG-FIX: Added FK validation to prevent orphaned records.

    Args:
        project_id: Projekt-ID
        name: Track-Name
        file_path: Pfad zur Audio-Datei
        session: SQLAlchemy Session (optional)

    Returns:
        AudioTrack-Objekt oder None bei Fehler
    """
    try:
        with managed_session(session) as db:
            # BUG-FIX: Validate FK exists
            project = db.query(Project).filter_by(id=project_id).first()
            if not project:
                logger.error(f"Cannot create audio track: Project {project_id} does not exist")
                return None

            track = AudioTrack(project_id=project_id, name=name, file_path=file_path)

            db.add(track)
            db.flush()
            db.refresh(track)

            logger.info(f"Audio-Track erstellt: {name} (ID: {track.id})")
            return track

    except Exception as e:
        logger.error(f"Fehler beim Erstellen des Audio-Tracks: {e}")
        return None


def get_audio_tracks_by_project(
    project_id: int,
    session: Session | None = None,
    limit: int | None = None,
    offset: int = 0,
    eager_load: bool = False,
) -> list[AudioTrack]:
    """
    Holt alle Audio-Tracks eines Projekts mit optionaler Pagination.

    PERF-08 FIX: Added pagination support to prevent memory issues with large datasets.
    PERF-06 FIX: Added eager_load option to prevent N+1 queries.

    Args:
        project_id: Projekt-ID
        session: SQLAlchemy Session (optional)
        limit: Maximum number of results (default: None = all, max: 1000)
        offset: Number of results to skip (default: 0)
        eager_load: Load beatgrids eagerly to prevent N+1 queries

    Returns:
        Liste von AudioTrack-Objekten
    """
    with managed_session(session) as db:
        query = db.query(AudioTrack).filter_by(project_id=project_id)

        # PERF-06 FIX: Eager loading for beatgrids
        if eager_load:
            query = query.options(selectinload(AudioTrack.beatgrids))

        # PERF-08 FIX: Apply pagination
        if offset > 0:
            query = query.offset(offset)
        if limit is not None:
            safe_limit = min(limit, MAX_PAGE_SIZE)
            query = query.limit(safe_limit)

        return query.all()


def update_audio_track(
    track_id: int, session: Session | None = None, **kwargs
) -> AudioTrack | None:
    """
    Aktualisiert Audio-Track-Attribute.

    Args:
        track_id: Track-ID
        session: SQLAlchemy Session (optional)
        **kwargs: Zu aktualisierende Attribute

    Returns:
        Aktualisiertes AudioTrack-Objekt oder None
    """
    try:
        with managed_session(session) as db:
            track = db.query(AudioTrack).filter_by(id=track_id).first()
            if not track:
                logger.warning(f"Audio-Track nicht gefunden: ID {track_id}")
                return None

            for key, value in kwargs.items():
                if hasattr(track, key):
                    setattr(track, key, value)

            db.flush()
            db.refresh(track)

            logger.info(f"Audio-Track aktualisiert: {track.name} (ID: {track.id})")
            return track

    except Exception as e:
        logger.error(f"Fehler beim Aktualisieren des Audio-Tracks: {e}")
        return None


def delete_audio_track(track_id: int, session: Session | None = None) -> bool:
    """
    Löscht einen Audio-Track.

    Args:
        track_id: Track-ID
        session: SQLAlchemy Session (optional)

    Returns:
        True bei Erfolg, False bei Fehler
    """
    try:
        with managed_session(session) as db:
            track = db.query(AudioTrack).filter_by(id=track_id).first()
            if not track:
                return False
            db.delete(track)
            logger.info(f"Audio track deleted: {track.name}")
            return True
    except Exception as e:
        logger.error(f"Error deleting audio track: {e}")
        return False


# ============================================================
# VIDEO CLIP CRUD
# ============================================================


def create_video_clip(
    project_id: int,
    name: str,
    file_path: str,
    session: Session | None = None,
) -> VideoClip | None:
    """
    Erstellt einen neuen Video-Clip.

    BUG-FIX: Added FK validation to prevent orphaned records.

    Args:
        project_id: Projekt-ID
        name: Clip-Name
        file_path: Pfad zur Video-Datei
        session: SQLAlchemy Session (optional)

    Returns:
        VideoClip-Objekt oder None bei Fehler
    """
    # SECURITY: Validate file_path (CWE-22 Path Traversal Prevention)
    try:
        validated_path = validate_file_path(
            file_path, must_exist=True, extensions=[".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv"]
        )
        file_path = str(validated_path)
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Ungueltiger Dateipfad '{file_path}': {e}")
        return None

    try:
        with managed_session(session) as db:
            # BUG-FIX: Validate FK exists
            project = db.query(Project).filter_by(id=project_id).first()
            if not project:
                logger.error(f"Cannot create video clip: Project {project_id} does not exist")
                return None

            clip = VideoClip(project_id=project_id, name=name, file_path=file_path)

            db.add(clip)
            db.flush()
            db.refresh(clip)

            logger.info(f"Video-Clip erstellt: {name} (ID: {clip.id})")
            return clip

    except Exception as e:
        logger.error(f"Fehler beim Erstellen des Video-Clips: {e}")
        return None


def get_video_clips_by_project(
    project_id: int,
    session: Session | None = None,
    limit: int | None = None,
    offset: int = 0,
    eager_load: bool = False,  # BUG-FIX: Default False - optional eager loading to prevent N+1 queries
) -> list[VideoClip]:
    """
    Holt alle Video-Clips eines Projekts mit optionaler Pagination.

    PERF-08 FIX: Added pagination support to prevent memory issues with large datasets.
    BUG-FIX: Made eager_load optional with default False.

    Args:
        project_id: Projekt-ID
        session: SQLAlchemy Session (optional)
        limit: Maximum number of results (default: None = all, max: 1000)
        offset: Number of results to skip (default: 0)
        eager_load: Load analysis data eagerly to prevent N+1 queries (default: False)

    Returns:
        Liste von VideoClip-Objekten
    """
    with managed_session(session) as db:
        query = db.query(VideoClip).filter_by(project_id=project_id)

        # PERF-06 FIX: Eager loading for analysis relationships
        if eager_load:
            query = query.options(
                selectinload(VideoClip.analysis_status),
                selectinload(VideoClip.colors),
                selectinload(VideoClip.motion),
                selectinload(VideoClip.scene_type),
                selectinload(VideoClip.mood),
                selectinload(VideoClip.objects),
                selectinload(VideoClip.style),
                selectinload(VideoClip.fingerprint),
            )

        # PERF-08 FIX: Apply pagination
        if offset > 0:
            query = query.offset(offset)
        if limit is not None:
            safe_limit = min(limit, MAX_PAGE_SIZE)
            query = query.limit(safe_limit)

        return query.all()


def delete_video_clip(
    clip_id: int, delete_files: bool = False, session: Session | None = None
) -> bool:
    """
    Löscht einen Video-Clip.

    Args:
        clip_id: Clip-ID
        delete_files: Falls True, wird auch die Video-Datei gelöscht
        session: SQLAlchemy Session (optional)

    Returns:
        True bei Erfolg, False bei Fehler
    """
    clip_path = None
    try:
        with managed_session(session) as db:
            clip = db.query(VideoClip).filter_by(id=clip_id).first()
            if not clip:
                return False

            clip_path = Path(clip.file_path)
            db.delete(clip)
            logger.info(f"Video clip deleted from DB: {clip.name}")

        if delete_files and clip_path and clip_path.exists():
            clip_path.unlink()
            logger.info(f"Video file deleted: {clip_path}")

        return True
    except Exception as e:
        logger.error(f"Error deleting video clip: {e}")
        return False


# ============================================================
# BEATGRID CRUD
# ============================================================


def create_beatgrid(
    audio_track_id: int,
    beat_times: list[float],
    downbeat_positions: list[int] | None = None,
    grid_type: str = "onset",
    session: Session | None = None,
) -> BeatGrid | None:
    """
    Erstellt ein Beatgrid für einen Audio-Track.

    BUG-FIX: Added FK validation to prevent orphaned records.

    Args:
        audio_track_id: Audio-Track-ID
        beat_times: Liste von Beat-Zeitpunkten in Sekunden
        downbeat_positions: Optionale Liste von Downbeat-Indizes
        grid_type: Typ des Beatgrids (onset, tempo, manual)
        session: SQLAlchemy Session (optional)

    Returns:
        BeatGrid-Objekt oder None bei Fehler
    """
    try:
        with managed_session(session) as db:
            # BUG-FIX: Validate FK exists
            audio_track = db.query(AudioTrack).filter_by(id=audio_track_id).first()
            if not audio_track:
                logger.error(f"Cannot create beatgrid: AudioTrack {audio_track_id} does not exist")
                return None

            beatgrid = BeatGrid(
                audio_track_id=audio_track_id, grid_type=grid_type, beat_times="", total_beats=0
            )

            beatgrid.set_beat_times(beat_times)
            if downbeat_positions:
                beatgrid.set_downbeat_positions(downbeat_positions)

            db.add(beatgrid)
            db.flush()
            db.refresh(beatgrid)

            logger.info(f"Beatgrid erstellt für Track {audio_track_id} ({len(beat_times)} Beats)")
            return beatgrid

    except Exception as e:
        logger.error(f"Fehler beim Erstellen des Beatgrids: {e}")
        return None


def get_beatgrid_by_audio_track(
    audio_track_id: int, session: Session | None = None
) -> BeatGrid | None:
    """
    Holt das Beatgrid eines Audio-Tracks.

    Args:
        audio_track_id: Audio-Track-ID
        session: SQLAlchemy Session (optional)

    Returns:
        BeatGrid-Objekt oder None
    """
    with managed_session(session) as db:
        return db.query(BeatGrid).filter_by(audio_track_id=audio_track_id).first()


# ============================================================
# BATCH DELETE OPERATIONS
# ============================================================


def batch_delete_video_clips(
    clip_ids: list[int], delete_files: bool = False, session: Session | None = None
) -> int:
    """
    Löscht mehrere Video-Clips auf einmal.

    BUG-FIX: Added batch delete function for efficient bulk operations.

    Args:
        clip_ids: Liste von Video-Clip-IDs
        delete_files: Falls True, werden auch die Video-Dateien gelöscht
        session: SQLAlchemy Session (optional)

    Returns:
        Anzahl der gelöschten Clips
    """
    if not clip_ids:
        return 0

    deleted_count = 0
    clip_paths = []

    try:
        with managed_session(session) as db:
            # Sammle Pfade vor dem Löschen (falls delete_files=True)
            if delete_files:
                clips = db.query(VideoClip).filter(VideoClip.id.in_(clip_ids)).all()
                clip_paths = [Path(clip.file_path) for clip in clips if clip.file_path]

            # Batch delete aus Datenbank
            deleted_count = (
                db.query(VideoClip)
                .filter(VideoClip.id.in_(clip_ids))
                .delete(synchronize_session=False)
            )
            logger.info(f"Batch deleted {deleted_count} video clips from database")

        # Optional: Lösche Dateien (nach DB-Commit)
        if delete_files and clip_paths:
            for clip_path in clip_paths:
                try:
                    if clip_path.exists():
                        clip_path.unlink()
                        logger.debug(f"Deleted file: {clip_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete file {clip_path}: {e}")

        return deleted_count

    except Exception as e:
        logger.error(f"Error during batch delete of video clips: {e}")
        return 0


def batch_delete_audio_tracks(
    track_ids: list[int], delete_files: bool = False, session: Session | None = None
) -> int:
    """
    Löscht mehrere Audio-Tracks auf einmal.

    BUG-FIX: Added batch delete function for efficient bulk operations.

    Args:
        track_ids: Liste von Audio-Track-IDs
        delete_files: Falls True, werden auch die Audio-Dateien gelöscht
        session: SQLAlchemy Session (optional)

    Returns:
        Anzahl der gelöschten Tracks
    """
    if not track_ids:
        return 0

    deleted_count = 0
    track_paths = []

    try:
        with managed_session(session) as db:
            # Sammle Pfade vor dem Löschen (falls delete_files=True)
            if delete_files:
                tracks = db.query(AudioTrack).filter(AudioTrack.id.in_(track_ids)).all()
                track_paths = [Path(track.file_path) for track in tracks if track.file_path]

            # Batch delete aus Datenbank (cascade löscht auch beatgrids)
            deleted_count = (
                db.query(AudioTrack)
                .filter(AudioTrack.id.in_(track_ids))
                .delete(synchronize_session=False)
            )
            logger.info(f"Batch deleted {deleted_count} audio tracks from database")

        # Optional: Lösche Dateien (nach DB-Commit)
        if delete_files and track_paths:
            for track_path in track_paths:
                try:
                    if track_path.exists():
                        track_path.unlink()
                        logger.debug(f"Deleted file: {track_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete file {track_path}: {e}")

        return deleted_count

    except Exception as e:
        logger.error(f"Error during batch delete of audio tracks: {e}")
        return 0


def batch_delete_pacing_blueprints(blueprint_ids: list[int], session: Session | None = None) -> int:
    """
    Löscht mehrere Pacing-Blueprints auf einmal.

    BUG-FIX: Added batch delete function for efficient bulk operations.

    Args:
        blueprint_ids: Liste von Blueprint-IDs
        session: SQLAlchemy Session (optional)

    Returns:
        Anzahl der gelöschten Blueprints
    """
    if not blueprint_ids:
        return 0

    try:
        with managed_session(session) as db:
            deleted_count = (
                db.query(PacingBlueprint)
                .filter(PacingBlueprint.id.in_(blueprint_ids))
                .delete(synchronize_session=False)
            )

            logger.info(f"Batch deleted {deleted_count} pacing blueprints from database")
            return deleted_count

    except Exception as e:
        logger.error(f"Error during batch delete of pacing blueprints: {e}")
        return 0


def delete_beatgrid(beatgrid_id: int, session: Session | None = None) -> bool:
    """
    Löscht ein Beatgrid.

    Args:
        beatgrid_id: Beatgrid-ID
        session: SQLAlchemy Session (optional)

    Returns:
        True bei Erfolg, False bei Fehler
    """
    try:
        with managed_session(session) as db:
            bg = db.query(BeatGrid).filter_by(id=beatgrid_id).first()
            if not bg:
                return False
            db.delete(bg)
            logger.info(f"Beatgrid deleted: ID {beatgrid_id}")
            return True
    except Exception as e:
        logger.error(f"Error deleting beatgrid: {e}")
        return False


def update_beatgrid(
    beatgrid_id: int,
    beat_times: list[float] | None = None,
    downbeat_positions: list[int] | None = None,
    session: Session | None = None,
) -> BeatGrid | None:
    """
    Aktualisiert ein Beatgrid.

    Args:
        beatgrid_id: Beatgrid-ID
        beat_times: Neue Liste von Beat-Zeitpunkten in Sekunden (optional)
        downbeat_positions: Neue Liste von Downbeat-Indizes (optional)
        session: SQLAlchemy Session (optional)

    Returns:
        Aktualisiertes BeatGrid-Objekt oder None
    """
    try:
        with managed_session(session) as db:
            bg = db.query(BeatGrid).filter_by(id=beatgrid_id).first()
            if not bg:
                return None

            if beat_times is not None:
                bg.set_beat_times(beat_times)
            if downbeat_positions is not None:
                bg.set_downbeat_positions(downbeat_positions)

            db.flush()
            db.refresh(bg)
            logger.info(f"Beatgrid updated: ID {beatgrid_id}")
            return bg
    except Exception as e:
        logger.error(f"Error updating beatgrid: {e}")
        return None


# ============================================================
# PACING BLUEPRINT CRUD
# ============================================================


def create_pacing_blueprint(
    project_id: int,
    name: str,
    blueprint: dict,
    session: Session | None = None,
) -> PacingBlueprint | None:
    """
    Erstellt ein Pacing-Blueprint.

    BUG-FIX: Added FK validation to prevent orphaned records.

    Args:
        project_id: Projekt-ID
        name: Blueprint-Name
        blueprint: Blueprint-Daten als Dictionary
        session: SQLAlchemy Session (optional)

    Returns:
        PacingBlueprint-Objekt oder None bei Fehler
    """
    try:
        with managed_session(session) as db:
            # BUG-FIX: Validate FK exists
            project = db.query(Project).filter_by(id=project_id).first()
            if not project:
                logger.error(f"Cannot create pacing blueprint: Project {project_id} does not exist")
                return None

            bp = PacingBlueprint(project_id=project_id, name=name)
            bp.set_blueprint(blueprint)
            db.add(bp)
            db.flush()
            db.refresh(bp)
            logger.info(f"Created pacing blueprint: {name}")
            return bp
    except Exception as e:
        logger.error(f"Error creating pacing blueprint: {e}")
        return None


def get_pacing_blueprint(
    blueprint_id: int, session: Session | None = None
) -> PacingBlueprint | None:
    """
    Holt ein Pacing-Blueprint anhand der ID.

    Args:
        blueprint_id: Blueprint-ID
        session: SQLAlchemy Session (optional)

    Returns:
        PacingBlueprint-Objekt oder None
    """
    with managed_session(session) as db:
        return db.query(PacingBlueprint).filter_by(id=blueprint_id).first()


def delete_pacing_blueprint(blueprint_id: int, session: Session | None = None) -> bool:
    """
    Löscht ein Pacing-Blueprint.

    Args:
        blueprint_id: Blueprint-ID
        session: SQLAlchemy Session (optional)

    Returns:
        True bei Erfolg, False bei Fehler
    """
    try:
        with managed_session(session) as db:
            bp = db.query(PacingBlueprint).filter_by(id=blueprint_id).first()
            if not bp:
                return False
            db.delete(bp)
            logger.info(f"Deleted pacing blueprint: {blueprint_id}")
            return True
    except Exception as e:
        logger.error(f"Error deleting pacing blueprint: {e}")
        return False
