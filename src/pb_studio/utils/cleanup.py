"""
PB_studio - Startup Cleanup Module

Bereinigt die Datenbank und temporäre Dateien beim App-Start,
um keine Altlasten in der Anwendung zu haben.
"""

import logging
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Optional

# Logger - wird später initialisiert, daher print() für frühe Meldungen
logger = None


def _get_db_path(root_dir: Path) -> Optional[Path]:
    """Findet den Pfad zur Projekt-Datenbank."""
    # Prüfe beide möglichen Orte
    locations = [root_dir / "data" / "project.db", root_dir / "project.db"]
    for loc in locations:
        if loc.exists():
            return loc
    return None


def cleanup_database_orphans(root_dir: Path) -> dict:
    """
    Bereinigt verwaiste Einträge in der Datenbank:
    - Video-Clips deren Dateien nicht mehr existieren
    - Audio-Tracks deren Dateien nicht mehr existieren
    - Analyse-Daten ohne zugehörigen Clip
    
    Args:
        root_dir: Projekt-Root-Verzeichnis
        
    Returns:
        Dict mit Bereinigungsstatistiken
    """
    stats = {
        "orphaned_video_clips": 0,
        "orphaned_audio_tracks": 0,
        "orphaned_analysis": 0,
        "errors": []
    }
    
    db_path = _get_db_path(root_dir)
    if not db_path:
        print("Keine Datenbank gefunden - überspringe DB-Bereinigung.")
        return stats
    
    print(f"Bereinige Datenbank: {db_path}")
    
    try:
        conn = sqlite3.connect(str(db_path), timeout=10)
        cursor = conn.cursor()
        
        # 1. Verwaiste Video-Clips finden und löschen
        try:
            cursor.execute("SELECT id, file_path FROM video_clips")
            video_clips = cursor.fetchall()
            
            orphaned_video_ids = []
            for clip_id, file_path in video_clips:
                if file_path and not Path(file_path).exists():
                    orphaned_video_ids.append(clip_id)
                    print(f"  Verwaister Video-Clip gefunden: {file_path}")
            
            if orphaned_video_ids:
                # Lösche zugehörige Analyse-Daten zuerst (wegen FK-Constraints)
                analysis_tables = [
                    "clip_semantics", "clip_colors", "clip_motion", 
                    "clip_scene_type", "clip_mood", "clip_objects",
                    "clip_style", "clip_fingerprint", "clip_analysis_status"
                ]
                for table in analysis_tables:
                    try:
                        cursor.execute(
                            f"DELETE FROM {table} WHERE video_clip_id IN ({','.join('?' * len(orphaned_video_ids))})",
                            orphaned_video_ids
                        )
                    except sqlite3.OperationalError:
                        pass  # Tabelle existiert nicht
                
                # Lösche die verwaisten Clips
                cursor.execute(
                    f"DELETE FROM video_clips WHERE id IN ({','.join('?' * len(orphaned_video_ids))})",
                    orphaned_video_ids
                )
                stats["orphaned_video_clips"] = len(orphaned_video_ids)
                print(f"  ✓ {len(orphaned_video_ids)} verwaiste Video-Clips gelöscht")
                
        except sqlite3.OperationalError as e:
            stats["errors"].append(f"Video-Clips: {e}")
        
        # 2. Verwaiste Audio-Tracks finden und löschen
        try:
            cursor.execute("SELECT id, file_path FROM audio_tracks")
            audio_tracks = cursor.fetchall()
            
            orphaned_audio_ids = []
            for track_id, file_path in audio_tracks:
                if file_path and not Path(file_path).exists():
                    orphaned_audio_ids.append(track_id)
                    print(f"  Verwaister Audio-Track gefunden: {file_path}")
            
            if orphaned_audio_ids:
                # Lösche zugehörige Beatgrids zuerst
                try:
                    cursor.execute(
                        f"DELETE FROM beatgrids WHERE audio_track_id IN ({','.join('?' * len(orphaned_audio_ids))})",
                        orphaned_audio_ids
                    )
                except sqlite3.OperationalError:
                    pass
                
                # Lösche die verwaisten Tracks
                cursor.execute(
                    f"DELETE FROM audio_tracks WHERE id IN ({','.join('?' * len(orphaned_audio_ids))})",
                    orphaned_audio_ids
                )
                stats["orphaned_audio_tracks"] = len(orphaned_audio_ids)
                print(f"  ✓ {len(orphaned_audio_ids)} verwaiste Audio-Tracks gelöscht")
                
        except sqlite3.OperationalError as e:
            stats["errors"].append(f"Audio-Tracks: {e}")
        
        # 3. Verwaiste Analyse-Daten (ohne zugehörigen Clip) löschen
        analysis_tables = [
            ("clip_semantics", "video_clip_id"),
            ("clip_colors", "video_clip_id"),
            ("clip_motion", "video_clip_id"),
            ("clip_scene_type", "video_clip_id"),
            ("clip_mood", "video_clip_id"),
            ("clip_objects", "video_clip_id"),
            ("clip_style", "video_clip_id"),
            ("clip_fingerprint", "video_clip_id"),
            ("clip_analysis_status", "video_clip_id"),
        ]
        
        for table, fk_column in analysis_tables:
            try:
                cursor.execute(f"""
                    DELETE FROM {table} 
                    WHERE {fk_column} NOT IN (SELECT id FROM video_clips)
                """)
                deleted = cursor.rowcount
                if deleted > 0:
                    stats["orphaned_analysis"] += deleted
                    print(f"  ✓ {deleted} verwaiste Einträge aus '{table}' gelöscht")
            except sqlite3.OperationalError:
                pass  # Tabelle existiert nicht
        
        # Commit und Vacuum
        conn.commit()
        
        if stats["orphaned_video_clips"] > 0 or stats["orphaned_audio_tracks"] > 0 or stats["orphaned_analysis"] > 0:
            print("  Komprimiere Datenbank...")
            cursor.execute("VACUUM")
            conn.commit()
        
        conn.close()
        print("Datenbank-Bereinigung abgeschlossen.")
        
    except Exception as e:
        stats["errors"].append(str(e))
        print(f"FEHLER bei DB-Bereinigung: {e}")
    
    return stats


def cleanup_stale_cache_files(root_dir: Path) -> dict:
    """
    Bereinigt veraltete Cache-Dateien die keinem aktiven Clip zugeordnet sind.
    
    Args:
        root_dir: Projekt-Root-Verzeichnis
        
    Returns:
        Dict mit Bereinigungsstatistiken
    """
    stats = {
        "deleted_files": 0,
        "freed_bytes": 0,
    }
    
    cache_dirs = [
        root_dir / "audio_cache",
        root_dir / "scene_cache", 
        root_dir / "stem_cache",
        root_dir / "trigger_cache",
        root_dir / "video_cache",
        root_dir / "temp",
    ]
    
    for cache_dir in cache_dirs:
        if not cache_dir.exists():
            continue
            
        # Lösche alle Dateien die älter als 7 Tage sind
        import time
        now = time.time()
        max_age = 7 * 24 * 60 * 60  # 7 Tage in Sekunden
        
        for file_path in cache_dir.rglob("*"):
            if file_path.is_file():
                try:
                    file_age = now - file_path.stat().st_mtime
                    if file_age > max_age:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        stats["deleted_files"] += 1
                        stats["freed_bytes"] += file_size
                except Exception:
                    pass
    
    if stats["deleted_files"] > 0:
        freed_mb = stats["freed_bytes"] / (1024 * 1024)
        print(f"  ✓ {stats['deleted_files']} veraltete Cache-Dateien gelöscht ({freed_mb:.1f} MB freigegeben)")
    
    return stats


def perform_startup_cleanup(root_dir: Path, aggressive: bool = False):
    """
    Performs cleanup of temporary data, caches, and database orphans
    to ensure a clean state on application startup.
    
    Args:
        root_dir: Projekt-Root-Verzeichnis
        aggressive: Wenn True, wird die gesamte DB gelöscht (wie CLEAN_AND_START.bat)
    """
    print("Performing startup cleanup...")
    
    # ========================================
    # SCHRITT 1: Datenbank bereinigen (WICHTIGSTER SCHRITT!)
    # ========================================
    if aggressive:
        # Aggressive Mode: Komplette DB löschen (wie CLEAN_AND_START.bat)
        db_locations = [root_dir, root_dir / "data"]
        db_filenames = ["project.db", "project.db-shm", "project.db-wal"]
        
        for loc in db_locations:
            if not loc.exists():
                continue
                
            for db_file in db_filenames:
                db_path = loc / db_file
                if db_path.exists():
                    try:
                        os.remove(db_path)
                        print(f"Deleted database: {db_path}")
                    except Exception as e:
                        print(f"Failed to delete {db_path}: {e}")
    else:
        # Standard Mode: Nur verwaiste Einträge entfernen (Audio/Video Altlasten!)
        cleanup_database_orphans(root_dir)
    
    # ========================================
    # SCHRITT 2: Veraltete Cache-Dateien bereinigen (älter als 7 Tage)
    # ========================================
    cleanup_stale_cache_files(root_dir)
    
    # ========================================
    # SCHRITT 3: Logs bereinigen (optional, nur bei aggressive=True)
    # ========================================
    if aggressive:
        log_dir = root_dir / "logs"
        if log_dir.exists():
            try:
                for log_file in log_dir.glob("*.log"):
                    try:
                        os.remove(log_file)
                    except Exception as e:
                        print(f"Failed to delete log {log_file.name}: {e}")
                print("Cleaned logs directory.")
            except Exception as e:
                print(f"Failed to clean logs directory: {e}")

    # ========================================
    # SCHRITT 4: Cache-Verzeichnisse vorbereiten
    # ========================================
    # Stelle sicher, dass alle benötigten Verzeichnisse existieren
    dirs_to_ensure = [
        "audio_cache",
        "scene_cache",
        "stem_cache",
        "trigger_cache",
        "video_cache",
        "thumbnails",
        "temp",
        "cache"
    ]

    for dir_name in dirs_to_ensure:
        dir_path = root_dir / dir_name
        if aggressive and dir_path.exists():
            try:
                # Aggressive: Komplett leeren und neu erstellen
                shutil.rmtree(dir_path)
                dir_path.mkdir(exist_ok=True)
                print(f"Reset directory: {dir_name}")
            except Exception as e:
                print(f"Failed to reset {dir_name}: {e}")
        else:
            # Standard: Nur sicherstellen, dass Verzeichnis existiert
            try:
                dir_path.mkdir(exist_ok=True)
            except Exception as e:
                pass  # Nicht kritisch

    print("Startup cleanup complete.")

