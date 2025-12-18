"""
Datenbank-Schema-Initialisierung und Migrations-Utilities für PB_studio

Stellt Funktionen bereit für:
- Schema-Initialisierung
- Schema-Updates
- Datenbank-Backup
"""

import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from ..utils.logger import get_logger
from .connection import get_db_manager, get_duckdb_manager
from .models import Base

logger = get_logger()

# Thread Pool für asynchrone Backups (max. 1 gleichzeitiges Backup)
_backup_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="db_backup")


def init_database(db_path: str | None = None, force: bool = False) -> bool:
    """
    Initialisiert die SQLite-Datenbank mit allen Tabellen.

    Args:
        db_path: Pfad zur Datenbank (optional)
        force: Falls True, werden bestehende Tabellen gelöscht

    Returns:
        True bei Erfolg, False bei Fehler
    """
    try:
        logger.info("Initialisiere Datenbank-Schema...")

        # Database Manager holen
        if db_path:
            from .connection import DatabaseManager

            # Erstelle neuen Manager und setze als globalen Manager
            db_manager = DatabaseManager(db_path)
            # Update globalen Manager
            import pb_studio.database.connection as conn_module

            conn_module._db_manager = db_manager
        else:
            db_manager = get_db_manager()

        # Falls force=True, lösche alle Tabellen
        if force:
            logger.warning("Force-Modus: Lösche bestehende Tabellen...")
            db_manager.drop_all(Base)

        # Erstelle alle Tabellen
        db_manager.init_db(Base)

        logger.info("Datenbank-Schema erfolgreich initialisiert")
        return True

    except Exception as e:
        logger.error(f"Fehler bei Datenbank-Initialisierung: {e}")
        return False


def init_analytics_database(db_path: str | None = None) -> bool:
    """
    Initialisiert die DuckDB-Analytics-Datenbank mit Views und Tabellen.

    Args:
        db_path: Pfad zur DuckDB-Datenbank (optional)

    Returns:
        True bei Erfolg, False bei Fehler
    """
    try:
        logger.info("Initialisiere DuckDB Analytics-Datenbank...")

        # DuckDB Manager holen
        if db_path:
            from .connection import DuckDBManager

            duckdb_manager = DuckDBManager(db_path)
        else:
            duckdb_manager = get_duckdb_manager()

        if duckdb_manager.connection is None:
            logger.warning("DuckDB nicht verfügbar - überspringe Analytics-Setup")
            return False

        # Erstelle Analytics-Tabellen
        duckdb_manager.execute(
            """
            CREATE TABLE IF NOT EXISTS clip_usage_stats (
                clip_id INTEGER,
                usage_count INTEGER DEFAULT 0,
                last_used TIMESTAMP,
                avg_duration FLOAT,
                PRIMARY KEY (clip_id)
            )
        """
        )

        duckdb_manager.execute(
            """
            CREATE TABLE IF NOT EXISTS project_stats (
                project_id INTEGER,
                total_clips INTEGER DEFAULT 0,
                total_duration FLOAT DEFAULT 0.0,
                avg_clip_duration FLOAT DEFAULT 0.0,
                last_rendered TIMESTAMP,
                PRIMARY KEY (project_id)
            )
        """
        )

        duckdb_manager.execute(
            """
            CREATE TABLE IF NOT EXISTS beatgrid_analysis (
                audio_track_id INTEGER,
                bpm FLOAT,
                beat_count INTEGER,
                analyzed_at TIMESTAMP,
                PRIMARY KEY (audio_track_id)
            )
        """
        )

        logger.info("DuckDB Analytics-Schema erfolgreich initialisiert")
        return True

    except Exception as e:
        logger.error(f"Fehler bei DuckDB-Initialisierung: {e}")
        return False


def verify_database_schema() -> bool:
    """
    Überprüft, ob das Datenbank-Schema vollständig ist.

    Returns:
        True wenn Schema vollständig, False sonst
    """
    session = None
    try:
        db_manager = get_db_manager()
        session = db_manager.get_session()

        # Prüfe ob wichtige Tabellen existieren
        from sqlalchemy import inspect

        inspector = inspect(db_manager.engine)
        tables = inspector.get_table_names()

        required_tables = [
            "projects",
            "audio_tracks",
            "video_clips",
            "beatgrids",
            "pacing_blueprints",
        ]

        missing_tables = [t for t in required_tables if t not in tables]

        if missing_tables:
            logger.warning(f"Fehlende Tabellen: {missing_tables}")
            return False

        logger.info("Datenbank-Schema vollständig")
        return True

    except Exception as e:
        logger.error(f"Fehler bei Schema-Verifikation: {e}")
        return False
    finally:
        # FIX #5: Session IMMER schließen, auch bei Exceptions
        if session is not None:
            session.close()


def backup_database(backup_path: str | None = None) -> bool:
    """
    Erstellt ein Backup der SQLite-Datenbank (synchron/blocking).

    Args:
        backup_path: Pfad für Backup-Datei (optional, default: db_name.backup.db)

    Returns:
        True bei Erfolg, False bei Fehler
    """
    try:
        db_manager = get_db_manager()
        source_path = db_manager.db_path

        if not source_path.exists():
            logger.error("Datenbank-Datei existiert nicht")
            return False

        # Backup-Pfad generieren
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = str(source_path.parent / f"{source_path.stem}_backup_{timestamp}.db")

        # Backup erstellen
        shutil.copy2(source_path, backup_path)
        logger.info(f"Datenbank-Backup erstellt: {backup_path}")
        return True

    except Exception as e:
        logger.error(f"Fehler bei Datenbank-Backup: {e}")
        return False


def _backup_with_progress(
    db_path: str, backup_dir: str | None, progress_callback: Callable[[int, int], None] | None
) -> str:
    """
    Erstellt Backup mit Progress-Reporting.

    Args:
        db_path: Pfad zur Datenbank
        backup_dir: Zielverzeichnis (optional)
        progress_callback: Callback(bytes_copied, total_bytes)

    Returns:
        Pfad zum erstellten Backup
    """
    src = Path(db_path)
    dst_dir = Path(backup_dir) if backup_dir else src.parent / "backups"
    dst_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = dst_dir / f"{src.stem}_{timestamp}.db"

    total_size = src.stat().st_size
    copied = 0
    chunk_size = 1024 * 1024  # 1MB chunks

    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        while True:
            chunk = fsrc.read(chunk_size)
            if not chunk:
                break
            fdst.write(chunk)
            copied += len(chunk)
            if progress_callback:
                progress_callback(copied, total_size)

    logger.info(f"Backup erstellt: {dst}")
    return str(dst)


def backup_database_async(
    db_path: str | None = None,
    backup_dir: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    completion_callback: Callable[[str, bool], None] | None = None,
) -> None:
    """
    Erstellt Datenbank-Backup asynchron (non-blocking).

    Args:
        db_path: Pfad zur Datenbank (optional, nutzt default DB wenn None)
        backup_dir: Zielverzeichnis für Backup (optional)
        progress_callback: Callback(bytes_copied, total_bytes) für Fortschritt
        completion_callback: Callback(backup_path, success) bei Fertigstellung

    Example:
        def on_progress(copied, total):
            percent = int(100 * copied / total)
            print(f"Backup: {percent}%")

        def on_complete(path, success):
            if success:
                print(f"Backup erstellt: {path}")
            else:
                print("Backup fehlgeschlagen!")

        backup_database_async(
            progress_callback=on_progress,
            completion_callback=on_complete
        )
    """
    # DB-Pfad ermitteln
    if db_path is None:
        db_manager = get_db_manager()
        db_path = str(db_manager.db_path)

    def _do_backup():
        try:
            backup_path = _backup_with_progress(db_path, backup_dir, progress_callback)
            if completion_callback:
                completion_callback(backup_path, True)
        except Exception as e:
            logger.error(f"Async backup fehlgeschlagen: {e}")
            if completion_callback:
                completion_callback(None, False)

    _backup_executor.submit(_do_backup)
    logger.info(f"Async Backup gestartet für {db_path}")


def get_database_stats() -> dict:
    """
    Gibt Statistiken über die Datenbank zurück.

    Returns:
        Dictionary mit Statistiken
    """
    session = None
    try:
        db_manager = get_db_manager()
        session = db_manager.get_session()

        from .models import AudioTrack, PacingBlueprint, Project, VideoClip

        stats = {
            "projects": session.query(Project).count(),
            "audio_tracks": session.query(AudioTrack).count(),
            "video_clips": session.query(VideoClip).count(),
            "pacing_blueprints": session.query(PacingBlueprint).count(),
            "database_path": str(db_manager.db_path),
            "database_size_mb": (
                round(db_manager.db_path.stat().st_size / (1024 * 1024), 2)
                if db_manager.db_path.exists()
                else 0
            ),
        }

        return stats

    except Exception as e:
        logger.error(f"Fehler bei Statistik-Abfrage: {e}")
        return {}
    finally:
        # FIX #5: Session IMMER schließen, auch bei Exceptions
        if session is not None:
            session.close()


# ========================================================================
# PARAMETRISIERTE ANALYTICS-QUERIES
# ========================================================================
# WICHTIG: Alle dynamischen DuckDB-Queries MÜSSEN parametrisiert sein!
# Grund: SQL-Injection-Prävention und Best Practice
#
# RICHTIG: execute(query, [param1, param2])
# FALSCH:  execute(f"... {param1} ...")
# ========================================================================


def insert_clip_usage(clip_id: int, usage_count: int = 1, avg_duration: float = 0.0) -> bool:
    """
    Fügt oder aktualisiert Clip-Nutzungsstatistiken hinzu (parametrisiert).

    Args:
        clip_id: ID des Video-Clips
        usage_count: Anzahl Verwendungen
        avg_duration: Durchschnittliche Clip-Dauer in Sekunden

    Returns:
        True bei Erfolg, False bei Fehler
    """
    try:
        duckdb_manager = get_duckdb_manager()
        if duckdb_manager.connection is None:
            return False

        # Parametrisierte Query mit ? Platzhaltern
        duckdb_manager.execute(
            """
            INSERT INTO clip_usage_stats (clip_id, usage_count, last_used, avg_duration)
            VALUES (?, ?, CURRENT_TIMESTAMP, ?)
            ON CONFLICT (clip_id) DO UPDATE SET
                usage_count = usage_count + ?,
                last_used = CURRENT_TIMESTAMP,
                avg_duration = (avg_duration + ?) / 2
            """,
            [clip_id, usage_count, avg_duration, usage_count, avg_duration],
        )
        return True

    except Exception as e:
        logger.error(f"Fehler beim Einfügen von Clip-Usage: {e}")
        return False


def get_clip_usage(clip_id: int) -> dict:
    """
    Holt Nutzungsstatistiken für einen Clip (parametrisiert).

    Args:
        clip_id: ID des Video-Clips

    Returns:
        Dictionary mit Statistiken oder leeres Dict bei Fehler
    """
    try:
        duckdb_manager = get_duckdb_manager()
        if duckdb_manager.connection is None:
            return {}

        result = duckdb_manager.execute(
            """
            SELECT usage_count, last_used, avg_duration
            FROM clip_usage_stats
            WHERE clip_id = ?
            """,
            [clip_id],
        ).fetchone()

        if result:
            return {
                "usage_count": result[0],
                "last_used": result[1],
                "avg_duration": result[2],
            }
        return {}

    except Exception as e:
        logger.error(f"Fehler beim Abrufen von Clip-Usage: {e}")
        return {}


def update_project_stats(
    project_id: int,
    total_clips: int = 0,
    total_duration: float = 0.0,
    avg_clip_duration: float = 0.0,
) -> bool:
    """
    Aktualisiert Projekt-Statistiken (parametrisiert).

    Args:
        project_id: ID des Projekts
        total_clips: Gesamtzahl Clips
        total_duration: Gesamtdauer in Sekunden
        avg_clip_duration: Durchschnittliche Clip-Dauer

    Returns:
        True bei Erfolg, False bei Fehler
    """
    try:
        duckdb_manager = get_duckdb_manager()
        if duckdb_manager.connection is None:
            return False

        duckdb_manager.execute(
            """
            INSERT INTO project_stats (project_id, total_clips, total_duration, avg_clip_duration, last_rendered)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT (project_id) DO UPDATE SET
                total_clips = ?,
                total_duration = ?,
                avg_clip_duration = ?,
                last_rendered = CURRENT_TIMESTAMP
            """,
            [
                project_id,
                total_clips,
                total_duration,
                avg_clip_duration,
                total_clips,
                total_duration,
                avg_clip_duration,
            ],
        )
        return True

    except Exception as e:
        logger.error(f"Fehler beim Aktualisieren von Project-Stats: {e}")
        return False


def get_project_stats(project_id: int) -> dict:
    """
    Holt Projekt-Statistiken (parametrisiert).

    Args:
        project_id: ID des Projekts

    Returns:
        Dictionary mit Statistiken oder leeres Dict bei Fehler
    """
    try:
        duckdb_manager = get_duckdb_manager()
        if duckdb_manager.connection is None:
            return {}

        result = duckdb_manager.execute(
            """
            SELECT total_clips, total_duration, avg_clip_duration, last_rendered
            FROM project_stats
            WHERE project_id = ?
            """,
            [project_id],
        ).fetchone()

        if result:
            return {
                "total_clips": result[0],
                "total_duration": result[1],
                "avg_clip_duration": result[2],
                "last_rendered": result[3],
            }
        return {}

    except Exception as e:
        logger.error(f"Fehler beim Abrufen von Project-Stats: {e}")
        return {}


def insert_beatgrid_analysis(audio_track_id: int, bpm: float, beat_count: int) -> bool:
    """
    Fügt Beatgrid-Analyse-Ergebnisse hinzu (parametrisiert).

    Args:
        audio_track_id: ID des Audio-Tracks
        bpm: Beats per Minute
        beat_count: Anzahl erkannter Beats

    Returns:
        True bei Erfolg, False bei Fehler
    """
    try:
        duckdb_manager = get_duckdb_manager()
        if duckdb_manager.connection is None:
            return False

        duckdb_manager.execute(
            """
            INSERT INTO beatgrid_analysis (audio_track_id, bpm, beat_count, analyzed_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT (audio_track_id) DO UPDATE SET
                bpm = ?,
                beat_count = ?,
                analyzed_at = CURRENT_TIMESTAMP
            """,
            [audio_track_id, bpm, beat_count, bpm, beat_count],
        )
        return True

    except Exception as e:
        logger.error(f"Fehler beim Einfügen von Beatgrid-Analyse: {e}")
        return False


def get_beatgrid_analysis(audio_track_id: int) -> dict:
    """
    Holt Beatgrid-Analyse für einen Audio-Track (parametrisiert).

    Args:
        audio_track_id: ID des Audio-Tracks

    Returns:
        Dictionary mit Analyse-Daten oder leeres Dict bei Fehler
    """
    try:
        duckdb_manager = get_duckdb_manager()
        if duckdb_manager.connection is None:
            return {}

        result = duckdb_manager.execute(
            """
            SELECT bpm, beat_count, analyzed_at
            FROM beatgrid_analysis
            WHERE audio_track_id = ?
            """,
            [audio_track_id],
        ).fetchone()

        if result:
            return {
                "bpm": result[0],
                "beat_count": result[1],
                "analyzed_at": result[2],
            }
        return {}

    except Exception as e:
        logger.error(f"Fehler beim Abrufen von Beatgrid-Analyse: {e}")
        return {}
