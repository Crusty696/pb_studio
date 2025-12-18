"""Database Migration Rollback Utilities."""
import logging
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def rollback_to_backup(db_path: str, backup_path: str) -> bool:
    """
    Stelle Datenbank aus Backup wieder her.

    Args:
        db_path: Pfad zur aktuellen Datenbank
        backup_path: Pfad zum Backup

    Returns:
        True bei Erfolg
    """
    db = Path(db_path)
    backup = Path(backup_path)

    if not backup.exists():
        logger.error(f"Backup nicht gefunden: {backup_path}")
        return False

    # Sicherheitskopie der aktuellen DB (falls Rollback fehlschlaegt)
    temp_backup = db.with_suffix(".rollback_temp")
    try:
        # Aktuelle DB sichern
        shutil.copy2(db, temp_backup)
        logger.info(f"Temporaeres Backup erstellt: {temp_backup}")

        # Backup wiederherstellen
        shutil.copy2(backup, db)
        logger.info(f"Rollback erfolgreich: {backup} → {db}")

        # Temp-Backup loeschen
        temp_backup.unlink()
        logger.info("Temporaeres Backup entfernt")

        return True

    except Exception as e:
        logger.error(f"Rollback fehlgeschlagen: {e}")

        # Versuche Wiederherstellung der vorherigen DB
        if temp_backup.exists():
            try:
                shutil.copy2(temp_backup, db)
                logger.warning("Rollback rueckgaengig gemacht - DB in vorherigem Zustand")
            except Exception as restore_error:
                logger.critical(f"Kritischer Fehler bei Wiederherstellung: {restore_error}")
                logger.critical(f"Manuelle Wiederherstellung erforderlich von: {temp_backup}")

        return False


def get_schema_version(db_path: str) -> int:
    """
    Hole aktuelle Schema-Version aus der Datenbank.

    Args:
        db_path: Pfad zur Datenbank

    Returns:
        Schema-Version (1 wenn keine Version-Info vorhanden)
    """
    try:
        conn = sqlite3.connect(db_path)
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

        # Hole neueste Version
        cursor.execute("SELECT version FROM schema_info ORDER BY applied_at DESC LIMIT 1")
        result = cursor.fetchone()
        conn.close()

        return result[0] if result else 1

    except Exception as e:
        logger.error(f"Fehler beim Lesen der Schema-Version: {e}")
        return 1


def list_backups(db_path: str) -> list[tuple[str, int, datetime]]:
    """
    Liste alle verfuegbaren Backups.

    Args:
        db_path: Pfad zur Datenbank

    Returns:
        List von (backup_path, version, timestamp) Tupeln
    """
    db = Path(db_path)
    backup_dir = db.parent

    if not backup_dir.exists():
        return []

    # Finde alle Backup-Dateien
    backups = []

    # Pattern: pb_studio_backup_v1_20241215_143022.db
    for backup_file in backup_dir.glob(f"{db.stem}_backup_v*_*.db"):
        try:
            # Extrahiere Version aus Dateinamen
            name_parts = backup_file.stem.split("_")

            # Finde Version (z.B. "v1")
            version_str = next((p for p in name_parts if p.startswith("v")), None)
            version = int(version_str[1:]) if version_str else 0

            # Finde Timestamp (YYYYMMDD_HHMMSS)
            date_part = None
            time_part = None
            for i, part in enumerate(name_parts):
                if len(part) == 8 and part.isdigit():  # Datum
                    date_part = part
                    if i + 1 < len(name_parts) and len(name_parts[i + 1]) == 6:
                        time_part = name_parts[i + 1]
                    break

            if date_part and time_part:
                timestamp = datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H%M%S")
            else:
                timestamp = datetime.fromtimestamp(backup_file.stat().st_mtime)

            backups.append((str(backup_file), version, timestamp))

        except Exception as e:
            logger.warning(f"Konnte Backup nicht parsen: {backup_file} - {e}")
            continue

    # Sortiere nach Timestamp (neueste zuerst)
    backups.sort(key=lambda x: x[2], reverse=True)

    return backups


def find_backup_for_version(db_path: str, target_version: int) -> str | None:
    """
    Finde Backup fuer spezifische Schema-Version.

    Args:
        db_path: Pfad zur Datenbank
        target_version: Gewuenschte Schema-Version

    Returns:
        Pfad zum Backup oder None
    """
    backups = list_backups(db_path)

    # Finde neuestes Backup mit passender Version
    for backup_path, version, timestamp in backups:
        if version == target_version:
            logger.info(f"Backup gefunden: {backup_path} (v{version}, {timestamp})")
            return backup_path

    logger.warning(f"Kein Backup fuer Version {target_version} gefunden")
    return None


def verify_backup(backup_path: str) -> bool:
    """
    Verifiziere dass Backup gueltig ist.

    Args:
        backup_path: Pfad zum Backup

    Returns:
        True wenn Backup gueltig
    """
    try:
        conn = sqlite3.connect(backup_path)
        cursor = conn.cursor()

        # Pruefe ob video_clips Tabelle existiert
        cursor.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='video_clips'
        """
        )

        if not cursor.fetchone():
            logger.error("Backup ungueltig: video_clips Tabelle fehlt")
            conn.close()
            return False

        # Pruefe Integritaet
        cursor.execute("PRAGMA integrity_check")
        result = cursor.fetchone()
        conn.close()

        if result[0] != "ok":
            logger.error(f"Backup-Integritaet fehlgeschlagen: {result[0]}")
            return False

        logger.info(f"Backup verifiziert: {backup_path}")
        return True

    except Exception as e:
        logger.error(f"Backup-Verifikation fehlgeschlagen: {e}")
        return False


def print_backup_info(db_path: str) -> None:
    """
    Zeige Info ueber verfuegbare Backups.

    Args:
        db_path: Pfad zur Datenbank
    """
    current_version = get_schema_version(db_path)
    print("\n=== Datenbank Backup Info ===")
    print(f"Aktuelle DB: {db_path}")
    print(f"Aktuelle Version: {current_version}")
    print()

    backups = list_backups(db_path)

    if not backups:
        print("Keine Backups gefunden.")
        return

    print(f"Verfuegbare Backups ({len(backups)}):")
    print("-" * 80)

    for backup_path, version, timestamp in backups:
        backup_file = Path(backup_path)
        size_mb = backup_file.stat().st_size / (1024 * 1024)
        print(f"  Version {version} | {timestamp.strftime('%Y-%m-%d %H:%M:%S')} | {size_mb:.1f} MB")
        print(f"  -> {backup_path}")
        print()
