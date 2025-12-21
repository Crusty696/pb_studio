"""
Migration v2: Video-Analyse System

Diese Migration fuegt folgende Tabellen hinzu:
- clip_colors: Farb-Analyse
- clip_motion: Bewegungs-Analyse
- clip_scene_type: Szenentyp-Analyse
- clip_mood: Stimmungs-Analyse
- clip_objects: Objekt-Erkennung (YOLO)
- clip_style: Visual Style
- clip_fingerprints: Hashes fuer Wiedererkennung
- clip_analysis_status: Analyse-Status Tracking

Und erweitert video_clips um:
- content_fingerprint
- original_path
- is_available
- last_seen_at
- needs_reanalysis
"""

import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

from ...utils.logger import get_logger
from ..connection import get_db_manager
from .rollback import find_backup_for_version, print_backup_info, rollback_to_backup, verify_backup

logger = get_logger()

# Aktuelle Schema-Version
SCHEMA_VERSION = 2


def get_schema_version(db_path: Path) -> int:
    """Liest aktuelle Schema-Version aus der Datenbank."""
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Pruefe ob schema_info Tabelle existiert
        cursor.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='schema_info'
        """
        )

        if not cursor.fetchone():
            # Keine Version Info = Version 1
            conn.close()
            return 1

        cursor.execute("SELECT version FROM schema_info ORDER BY applied_at DESC LIMIT 1")
        result = cursor.fetchone()
        conn.close()

        return result[0] if result else 1

    except Exception as e:
        logger.error(f"Fehler beim Lesen der Schema-Version: {e}")
        return 1


def check_migration_needed(db_path: Path | None = None) -> tuple[bool, int, int]:
    """
    Prueft ob eine Migration noetig ist.

    Returns:
        Tuple[bool, int, int]: (migration_needed, current_version, target_version)
    """
    if db_path is None:
        db_manager = get_db_manager()
        db_path = db_manager.db_path

    current = get_schema_version(db_path)
    return (current < SCHEMA_VERSION, current, SCHEMA_VERSION)


def create_backup(db_path: Path) -> Path:
    """Erstellt ein Backup vor der Migration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.parent / f"{db_path.stem}_backup_v1_{timestamp}.db"
    shutil.copy2(db_path, backup_path)
    logger.info(f"Backup erstellt: {backup_path}")
    return backup_path


def migrate_to_v2(db_path: Path | None = None, create_backup_first: bool = True) -> bool:
    """
    Fuehrt Migration auf Schema v2 durch.

    Args:
        db_path: Pfad zur Datenbank (optional)
        create_backup_first: Erstellt Backup vor Migration

    Returns:
        True bei Erfolg
    """
    try:
        if db_path is None:
            db_manager = get_db_manager()
            db_path = db_manager.db_path

        # Pruefe ob Migration noetig
        needed, current, target = check_migration_needed(db_path)
        if not needed:
            logger.info(f"Keine Migration noetig. Schema ist bereits auf Version {current}")
            return True

        logger.info(f"Starte Migration von v{current} auf v{target}...")

        # Backup erstellen
        if create_backup_first:
            backup_path = create_backup(db_path)

        # SQLite Verbindung
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # =============================================
        # 1. Schema Info Tabelle erstellen
        # =============================================
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_info (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version INTEGER NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT
            )
        """
        )

        # =============================================
        # 2. Neue Spalten zu video_clips hinzufuegen
        # =============================================
        logger.info("Erweitere video_clips Tabelle...")

        # Pruefe existierende Spalten
        cursor.execute("PRAGMA table_info(video_clips)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        new_columns = [
            ("content_fingerprint", "TEXT"),  # UNIQUE via Index, nicht ALTER TABLE
            ("vision_description", "TEXT"),
            ("story_role", "TEXT"),
            ("original_path", "TEXT"),
            ("is_available", "BOOLEAN DEFAULT 1"),
            ("last_seen_at", "TIMESTAMP"),
            ("needs_reanalysis", "BOOLEAN DEFAULT 1"),
        ]

        for col_name, col_type in new_columns:
            if col_name not in existing_columns:
                # CRITICAL-05 FIX: Quote identifiers to prevent SQL injection
                # Although new_columns is hardcoded, this is best practice
                cursor.execute(f'ALTER TABLE video_clips ADD COLUMN "{col_name}" {col_type}')
                logger.info(f"  + Spalte '{col_name}' hinzugefuegt")

        # Index fuer content_fingerprint
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_clips_fingerprint
            ON video_clips(content_fingerprint)
        """
        )

        # =============================================
        # 3. Neue Analyse-Tabellen erstellen
        # =============================================

        # clip_colors
        logger.info("Erstelle clip_colors Tabelle...")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS clip_colors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                clip_id INTEGER NOT NULL REFERENCES video_clips(id) ON DELETE CASCADE,
                frame_position TEXT DEFAULT 'middle',
                dominant_colors TEXT,
                temperature TEXT,
                temperature_score REAL,
                brightness TEXT,
                brightness_value REAL,
                color_moods TEXT,
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_colors_clip ON clip_colors(clip_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_colors_temp ON clip_colors(temperature)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_colors_bright ON clip_colors(brightness)")

        # clip_motion
        logger.info("Erstelle clip_motion Tabelle...")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS clip_motion (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                clip_id INTEGER NOT NULL REFERENCES video_clips(id) ON DELETE CASCADE,
                motion_type TEXT,
                motion_score REAL,
                motion_rhythm TEXT,
                motion_variation REAL,
                camera_motion TEXT,
                camera_magnitude REAL,
                flow_magnitude_avg REAL,
                flow_direction_dominant REAL,
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_motion_clip ON clip_motion(clip_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_motion_type ON clip_motion(motion_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_motion_camera ON clip_motion(camera_motion)")

        # clip_scene_type
        logger.info("Erstelle clip_scene_type Tabelle...")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS clip_scene_type (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                clip_id INTEGER NOT NULL REFERENCES video_clips(id) ON DELETE CASCADE,
                frame_position TEXT DEFAULT 'middle',
                scene_types TEXT,
                edge_density REAL,
                texture_variance REAL,
                center_ratio REAL,
                depth_of_field REAL,
                has_face BOOLEAN DEFAULT 0,
                face_count INTEGER DEFAULT 0,
                face_size_ratio REAL,
                confidence_scores TEXT,
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_scene_clip ON clip_scene_type(clip_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_scene_face ON clip_scene_type(has_face)")

        # clip_mood
        logger.info("Erstelle clip_mood Tabelle...")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS clip_mood (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                clip_id INTEGER NOT NULL REFERENCES video_clips(id) ON DELETE CASCADE,
                frame_position TEXT DEFAULT 'middle',
                moods TEXT,
                mood_scores TEXT,
                brightness REAL,
                saturation REAL,
                contrast REAL,
                energy REAL,
                warm_ratio REAL,
                cool_ratio REAL,
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mood_clip ON clip_mood(clip_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mood_energy ON clip_mood(energy)")

        # clip_objects
        logger.info("Erstelle clip_objects Tabelle...")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS clip_objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                clip_id INTEGER NOT NULL REFERENCES video_clips(id) ON DELETE CASCADE,
                frame_position TEXT DEFAULT 'middle',
                detected_objects TEXT,
                object_counts TEXT,
                confidence_scores TEXT,
                content_tags TEXT,
                line_count INTEGER,
                green_ratio REAL,
                sky_ratio REAL,
                symmetry REAL,
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_objects_clip ON clip_objects(clip_id)")

        # clip_style
        logger.info("Erstelle clip_style Tabelle...")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS clip_style (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                clip_id INTEGER NOT NULL REFERENCES video_clips(id) ON DELETE CASCADE,
                frame_position TEXT DEFAULT 'middle',
                styles TEXT,
                unique_colors INTEGER,
                noise_level REAL,
                sharpness REAL,
                vignette_score REAL,
                saturation_mean REAL,
                saturation_std REAL,
                dynamic_range REAL,
                mean_brightness REAL,
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_style_clip ON clip_style(clip_id)")

        # clip_fingerprints
        logger.info("Erstelle clip_fingerprints Tabelle...")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS clip_fingerprints (
                clip_id INTEGER PRIMARY KEY REFERENCES video_clips(id) ON DELETE CASCADE,
                content_fingerprint TEXT UNIQUE,
                phash TEXT,
                dhash TEXT,
                ahash TEXT,
                vector_file TEXT,
                faiss_index_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_fp_content ON clip_fingerprints(content_fingerprint)"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fp_phash ON clip_fingerprints(phash)")

        # clip_analysis_status
        logger.info("Erstelle clip_analysis_status Tabelle...")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS clip_analysis_status (
                clip_id INTEGER PRIMARY KEY REFERENCES video_clips(id) ON DELETE CASCADE,
                colors_analyzed BOOLEAN DEFAULT 0,
                motion_analyzed BOOLEAN DEFAULT 0,
                scene_analyzed BOOLEAN DEFAULT 0,
                mood_analyzed BOOLEAN DEFAULT 0,
                objects_analyzed BOOLEAN DEFAULT 0,
                style_analyzed BOOLEAN DEFAULT 0,
                fingerprint_created BOOLEAN DEFAULT 0,
                vector_extracted BOOLEAN DEFAULT 0,
                colors_version INTEGER DEFAULT 0,
                motion_version INTEGER DEFAULT 0,
                scene_version INTEGER DEFAULT 0,
                mood_version INTEGER DEFAULT 0,
                objects_version INTEGER DEFAULT 0,
                style_version INTEGER DEFAULT 0,
                last_full_analysis TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # =============================================
        # 4. Initialisiere Analyse-Status fuer existierende Clips
        # =============================================
        logger.info("Initialisiere Analyse-Status fuer existierende Clips...")
        cursor.execute(
            """
            INSERT OR IGNORE INTO clip_analysis_status (clip_id)
            SELECT id FROM video_clips
        """
        )

        # Setze needs_reanalysis = 1 fuer alle existierenden Clips
        cursor.execute("UPDATE video_clips SET needs_reanalysis = 1 WHERE needs_reanalysis IS NULL")
        cursor.execute("UPDATE video_clips SET is_available = 1 WHERE is_available IS NULL")

        # =============================================
        # 5. Schema-Version speichern
        # =============================================
        cursor.execute(
            """
            INSERT INTO schema_info (version, description)
            VALUES (?, ?)
        """,
            (
                SCHEMA_VERSION,
                "Video-Analyse System: Farbe, Motion, Szene, Mood, Objects, Style, Fingerprints",
            ),
        )

        # Commit
        conn.commit()
        conn.close()

        logger.info(f"Migration auf v{SCHEMA_VERSION} erfolgreich abgeschlossen!")
        logger.info("Alle existierenden Clips sind fuer Re-Analyse markiert.")

        return True

    except Exception as e:
        logger.error(f"Migration fehlgeschlagen: {e}")
        import traceback

        traceback.print_exc()
        return False


def get_unanalyzed_clip_count(db_path: Path | None = None) -> int:
    """Zaehlt Clips die noch analysiert werden muessen."""
    try:
        if db_path is None:
            db_manager = get_db_manager()
            db_path = db_manager.db_path

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT COUNT(*) FROM video_clips
            WHERE needs_reanalysis = 1 OR needs_reanalysis IS NULL
        """
        )
        count = cursor.fetchone()[0]
        conn.close()

        return count

    except Exception as e:
        logger.error(f"Fehler beim Zaehlen: {e}")
        return 0


def rollback_v2_to_v1(db_path: Path | None = None) -> bool:
    """
    Rollback von Schema V2 zu V1.

    WARNUNG: Diese Operation stellt die Datenbank aus einem Backup wieder her.
    Alle Aenderungen seit dem Backup gehen verloren!

    Args:
        db_path: Pfad zur Datenbank (optional)

    Returns:
        True bei Erfolg
    """
    try:
        if db_path is None:
            db_manager = get_db_manager()
            db_path = db_manager.db_path

        db_path_str = str(db_path)

        # Pruefe aktuelle Version
        current_version = get_schema_version(db_path)

        if current_version != 2:
            logger.error(f"Kann nur V2 zurückrollen, aktuelle Version: {current_version}")
            return False

        logger.info(f"Starte Rollback von v{current_version} auf v1...")

        # Finde V1 Backup
        v1_backup = find_backup_for_version(db_path_str, target_version=1)

        if not v1_backup:
            logger.error("Kein V1 Backup gefunden - Rollback nicht moeglich")
            logger.info("Verfuegbare Backups:")
            print_backup_info(db_path_str)
            return False

        # Verifiziere Backup
        if not verify_backup(v1_backup):
            logger.error(f"Backup-Verifikation fehlgeschlagen: {v1_backup}")
            return False

        # Erstelle Sicherheitskopie der V2 DB vor Rollback
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safety_backup = db_path.parent / f"{db_path.stem}_v2_before_rollback_{timestamp}.db"
        shutil.copy2(db_path, safety_backup)
        logger.info(f"Sicherheitskopie der V2 DB erstellt: {safety_backup}")

        # Fuehre Rollback durch
        if rollback_to_backup(db_path_str, v1_backup):
            logger.info("Rollback auf v1 erfolgreich!")
            logger.info(f"V2 Backup gespeichert unter: {safety_backup}")
            return True
        else:
            logger.error("Rollback fehlgeschlagen")
            return False

    except Exception as e:
        logger.error(f"Rollback-Prozess fehlgeschlagen: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test-Migration
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "rollback":
        print("=== Rollback zu v1 ===")
        success = rollback_v2_to_v1()
        sys.exit(0 if success else 1)
    else:
        print("=== Migration zu v2 ===")
        migrate_to_v2()
