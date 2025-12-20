"""
PB_studio - Datenbank bereinigen
Löscht alle Video-Clips und bereinigt die Datenbank.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def clear_database():
    """Lösche alle Video-Clips aus der Datenbank."""

    # Pfade zur Datenbank
    db_path = Path(__file__).parent.parent / "data" / "project.db"
    wal_path = db_path.with_suffix(".db-wal")
    shm_path = db_path.with_suffix(".db-shm")

    print(f"Datenbank-Pfad: {db_path}")

    if not db_path.exists():
        print("Keine Datenbank gefunden. Nichts zu löschen.")
        return

    try:
        import sqlite3

        # Lösche WAL/SHM Dateien zuerst (falls App nicht sauber beendet)
        for cache_file in [wal_path, shm_path]:
            if cache_file.exists():
                try:
                    cache_file.unlink()
                    print(f"Gelöscht: {cache_file.name}")
                except PermissionError:
                    print(f"WARNUNG: Kann {cache_file.name} nicht löschen (App läuft?)")

        # Verbinde zur Datenbank
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Zähle vorhandene Clips
        try:
            cursor.execute("SELECT COUNT(*) FROM video_clips")
            clip_count = cursor.fetchone()[0]
            print(f"Gefundene Video-Clips: {clip_count}")
        except sqlite3.OperationalError:
            print("Tabelle 'video_clips' nicht gefunden.")
            clip_count = 0

        # Lösche alle Video-Clips
        if clip_count > 0:
            cursor.execute("DELETE FROM video_clips")
            conn.commit()
            print(f"✓ {clip_count} Video-Clips gelöscht")

        # Lösche auch zugehörige Analyse-Daten falls vorhanden
        tables_to_clear = [
            "clip_analysis",
            "clip_motion",
            "clip_mood",
            "clip_embedding",
            "clip_fingerprint",
        ]

        for table in tables_to_clear:
            try:
                cursor.execute(f"DELETE FROM {table}")
                deleted = cursor.rowcount
                if deleted > 0:
                    print(f"✓ {deleted} Einträge aus '{table}' gelöscht")
            except sqlite3.OperationalError:
                pass  # Tabelle existiert nicht

        conn.commit()

        # VACUUM um Speicherplatz freizugeben
        print("Komprimiere Datenbank...")
        cursor.execute("VACUUM")
        conn.commit()

        conn.close()
        print("\n✓ Datenbank erfolgreich bereinigt!")

    except Exception as e:
        print(f"FEHLER: {e}")
        return False

    return True


def clear_thumbnails():
    """Lösche alle Thumbnail-Dateien."""
    thumb_dir = Path(__file__).parent.parent / "thumbnails"

    if not thumb_dir.exists():
        return

    count = 0
    for file in thumb_dir.glob("*.jpg"):
        try:
            file.unlink()
            count += 1
        except PermissionError:
            pass

    for file in thumb_dir.glob("*.png"):
        try:
            file.unlink()
            count += 1
        except PermissionError:
            pass

    if count > 0:
        print(f"✓ {count} Thumbnails gelöscht")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  PB_studio - Datenbank bereinigen")
    print("=" * 50 + "\n")

    clear_database()
    clear_thumbnails()

    print("\nFertig! Starte die App neu um mit leerer Bibliothek zu beginnen.")
