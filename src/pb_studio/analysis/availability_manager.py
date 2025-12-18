"""
Clip-Verfuegbarkeits-Manager.

Verwaltet die Verfuegbarkeit von Clips ueber mehrere Laufwerke/Ordner:
- Prueft ob Clips noch existieren
- Findet verschobene/umbenannte Clips via Fingerprint
- Aktualisiert Pfade automatisch
- Unterstuetzt mehrere Quell-Ordner
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..database.connection import get_db_manager
from ..utils.logger import get_logger
from .fingerprint import ContentFingerprint

logger = get_logger()

# Unterstuetzte Video-Erweiterungen
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".m4v", ".flv"}


@dataclass
class ClipLocation:
    """Speichert Clip-Pfad-Informationen."""

    clip_id: int
    current_path: str
    original_path: str | None
    content_fingerprint: str | None
    is_available: bool
    last_seen_at: datetime | None


class AvailabilityManager:
    """Verwaltet Clip-Verfuegbarkeit ueber mehrere Laufwerke."""

    def __init__(self, db_path: Path | None = None):
        """
        Args:
            db_path: Optionaler Pfad zur Datenbank
        """
        if db_path is None:
            self.db_manager = get_db_manager()
            self.db_path = self.db_manager.db_path
        else:
            self.db_path = db_path

        self.fingerprinter = ContentFingerprint()
        self.search_paths: list[Path] = []  # Zusaetzliche Suchpfade

    def _get_connection(self) -> sqlite3.Connection:
        """Erstellt eine SQLite-Verbindung."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def add_search_path(self, path: str) -> bool:
        """
        Fuegt einen Suchpfad fuer Clip-Wiedererkennung hinzu.

        Args:
            path: Ordnerpfad

        Returns:
            True wenn Pfad gueltig
        """
        p = Path(path)
        if p.exists() and p.is_dir():
            if p not in self.search_paths:
                self.search_paths.append(p)
                logger.info(f"Suchpfad hinzugefuegt: {path}")
            return True
        logger.warning(f"Ungueltiger Suchpfad: {path}")
        return False

    def remove_search_path(self, path: str) -> bool:
        """Entfernt einen Suchpfad."""
        p = Path(path)
        if p in self.search_paths:
            self.search_paths.remove(p)
            return True
        return False

    def get_search_paths(self) -> list[str]:
        """Gibt alle konfigurierten Suchpfade zurueck."""
        return [str(p) for p in self.search_paths]

    def check_clip_availability(self, clip_id: int) -> tuple[bool, str | None]:
        """
        Prueft ob ein Clip verfuegbar ist.

        Args:
            clip_id: ID des Clips

        Returns:
            Tuple (verfuegbar, neuer_pfad_wenn_gefunden)
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT id, file_path, original_path, content_fingerprint, is_available
                FROM video_clips WHERE id = ?
            """,
                (clip_id,),
            )

            row = cursor.fetchone()
            conn.close()

            if row is None:
                return False, None

            current_path = row["file_path"]
            fingerprint = row["content_fingerprint"]

            # Pruefe aktuellen Pfad
            if current_path and Path(current_path).exists():
                self._update_availability(clip_id, True, current_path)
                return True, current_path

            # Clip nicht am aktuellen Pfad - suche mit Fingerprint
            if fingerprint:
                new_path = self._find_clip_by_fingerprint(fingerprint, row["original_path"])
                if new_path:
                    self._update_clip_path(clip_id, new_path)
                    return True, new_path

            # Nicht gefunden
            self._update_availability(clip_id, False)
            return False, None

        except Exception as e:
            logger.error(f"Fehler bei Verfuegbarkeitspruefung: {e}")
            return False, None

    def _find_clip_by_fingerprint(self, fingerprint: str, original_path: str | None) -> str | None:
        """Sucht Clip mit Fingerprint in allen Suchpfaden."""
        # Zuerst Original-Ordner pruefen
        if original_path:
            original_dir = Path(original_path).parent
            if original_dir.exists():
                found = self._search_in_folder(original_dir, fingerprint)
                if found:
                    return found

        # Dann alle Suchpfade durchsuchen
        for search_path in self.search_paths:
            found = self._search_in_folder(search_path, fingerprint, recursive=True)
            if found:
                return found

        return None

    def _search_in_folder(
        self, folder: Path, fingerprint: str, recursive: bool = False
    ) -> str | None:
        """Durchsucht einen Ordner nach einem Clip mit bestimmtem Fingerprint."""
        try:
            if recursive:
                files = folder.rglob("*")
            else:
                files = folder.glob("*")

            for file_path in files:
                if file_path.suffix.lower() in VIDEO_EXTENSIONS:
                    computed_fp = self.fingerprinter.compute(str(file_path))
                    if computed_fp == fingerprint:
                        logger.info(f"Clip via Fingerprint gefunden: {file_path}")
                        return str(file_path)

        except Exception as e:
            logger.error(f"Fehler bei Ordner-Suche in {folder}: {e}")

        return None

    def _update_availability(self, clip_id: int, is_available: bool, path: str | None = None):
        """Aktualisiert Verfuegbarkeits-Status in DB."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            if is_available:
                cursor.execute(
                    """
                    UPDATE video_clips
                    SET is_available = 1, last_seen_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """,
                    (clip_id,),
                )
            else:
                cursor.execute(
                    """
                    UPDATE video_clips
                    SET is_available = 0
                    WHERE id = ?
                """,
                    (clip_id,),
                )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Fehler beim Update der Verfuegbarkeit: {e}")

    def _update_clip_path(self, clip_id: int, new_path: str):
        """Aktualisiert Clip-Pfad nach Wiederfinden."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE video_clips
                SET file_path = ?,
                    is_available = 1,
                    last_seen_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """,
                (new_path, clip_id),
            )

            conn.commit()
            conn.close()

            logger.info(f"Clip {clip_id} Pfad aktualisiert: {new_path}")

        except Exception as e:
            logger.error(f"Fehler beim Pfad-Update: {e}")

    def check_all_clips(self, progress_callback=None) -> dict:
        """
        Prueft Verfuegbarkeit aller Clips.

        Args:
            progress_callback: Callback(current, total) fuer Fortschritt

        Returns:
            Dict mit Statistiken
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT id FROM video_clips")
            clip_ids = [row["id"] for row in cursor.fetchall()]
            conn.close()

            stats = {"total": len(clip_ids), "available": 0, "missing": 0, "relocated": 0}

            for i, clip_id in enumerate(clip_ids):
                is_available, new_path = self.check_clip_availability(clip_id)

                if is_available:
                    stats["available"] += 1
                    if new_path and new_path != self._get_original_path(clip_id):
                        stats["relocated"] += 1
                else:
                    stats["missing"] += 1

                if progress_callback:
                    progress_callback(i + 1, len(clip_ids))

            logger.info(
                f"Verfuegbarkeitspruefung: {stats['available']}/{stats['total']} verfuegbar, "
                f"{stats['missing']} fehlen, {stats['relocated']} umgezogen"
            )

            return stats

        except Exception as e:
            logger.error(f"Fehler bei Gesamt-Pruefung: {e}")
            return {"total": 0, "available": 0, "missing": 0, "relocated": 0}

    def _get_original_path(self, clip_id: int) -> str | None:
        """Holt Original-Pfad eines Clips."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT original_path FROM video_clips WHERE id = ?", (clip_id,))
            row = cursor.fetchone()
            conn.close()
            return row["original_path"] if row else None
        except Exception:
            return None

    def get_unavailable_clips(self) -> list[dict]:
        """
        Gibt Liste aller nicht verfuegbaren Clips zurueck.

        Returns:
            Liste von Dicts mit Clip-Infos
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT id, file_path, original_path, content_fingerprint, last_seen_at
                FROM video_clips
                WHERE is_available = 0
                ORDER BY last_seen_at DESC
            """
            )

            clips = []
            for row in cursor.fetchall():
                clips.append(
                    {
                        "id": row["id"],
                        "file_path": row["file_path"],
                        "original_path": row["original_path"],
                        "fingerprint": row["content_fingerprint"],
                        "last_seen": row["last_seen_at"],
                    }
                )

            conn.close()
            return clips

        except Exception as e:
            logger.error(f"Fehler beim Laden nicht verfuegbarer Clips: {e}")
            return []

    def get_available_clips(self) -> list[int]:
        """Gibt Liste aller verfuegbaren Clip-IDs zurueck."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT id FROM video_clips WHERE is_available = 1")
            clip_ids = [row["id"] for row in cursor.fetchall()]
            conn.close()

            return clip_ids

        except Exception as e:
            logger.error(f"Fehler beim Laden verfuegbarer Clips: {e}")
            return []

    def set_clip_fingerprint(self, clip_id: int, fingerprint: str) -> bool:
        """
        Setzt Fingerprint fuer einen Clip.

        Args:
            clip_id: ID des Clips
            fingerprint: Berechneter Fingerprint

        Returns:
            True bei Erfolg
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE video_clips
                SET content_fingerprint = ?
                WHERE id = ?
            """,
                (fingerprint, clip_id),
            )

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"Fehler beim Setzen des Fingerprints: {e}")
            return False

    def compute_and_store_fingerprint(self, clip_id: int) -> str | None:
        """
        Berechnet und speichert Fingerprint fuer einen Clip.

        Args:
            clip_id: ID des Clips

        Returns:
            Berechneter Fingerprint oder None
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT file_path FROM video_clips WHERE id = ?", (clip_id,))
            row = cursor.fetchone()
            conn.close()

            if row is None:
                return None

            file_path = row["file_path"]
            if not file_path or not Path(file_path).exists():
                return None

            fingerprint = self.fingerprinter.compute(file_path)
            if fingerprint:
                self.set_clip_fingerprint(clip_id, fingerprint)

                # Original-Pfad setzen wenn noch nicht vorhanden
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE video_clips
                    SET original_path = COALESCE(original_path, file_path)
                    WHERE id = ?
                """,
                    (clip_id,),
                )
                conn.commit()
                conn.close()

            return fingerprint

        except Exception as e:
            logger.error(f"Fehler beim Fingerprint-Berechnung: {e}")
            return None

    def find_duplicates(self) -> list[list[int]]:
        """
        Findet Duplikate basierend auf Content-Fingerprint.

        Returns:
            Liste von Listen mit duplizierten Clip-IDs
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT content_fingerprint, GROUP_CONCAT(id) as clip_ids
                FROM video_clips
                WHERE content_fingerprint IS NOT NULL
                GROUP BY content_fingerprint
                HAVING COUNT(*) > 1
            """
            )

            duplicates = []
            for row in cursor.fetchall():
                clip_ids = [int(id) for id in row["clip_ids"].split(",")]
                duplicates.append(clip_ids)

            conn.close()

            if duplicates:
                logger.info(f"{len(duplicates)} Duplikat-Gruppen gefunden")

            return duplicates

        except Exception as e:
            logger.error(f"Fehler bei Duplikat-Suche: {e}")
            return []

    def scan_folder_for_new_clips(self, folder: str, recursive: bool = True) -> list[str]:
        """
        Scannt einen Ordner nach neuen (noch nicht importierten) Videos.

        Args:
            folder: Zu scannender Ordner
            recursive: Auch Unterordner durchsuchen

        Returns:
            Liste neuer Video-Pfade
        """
        try:
            folder_path = Path(folder)
            if not folder_path.exists():
                return []

            # Existierende Fingerprints laden
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT content_fingerprint FROM video_clips WHERE content_fingerprint IS NOT NULL"
            )
            existing_fps = {row["content_fingerprint"] for row in cursor.fetchall()}
            conn.close()

            # Ordner durchsuchen
            new_videos = []
            if recursive:
                files = folder_path.rglob("*")
            else:
                files = folder_path.glob("*")

            for file_path in files:
                if file_path.suffix.lower() in VIDEO_EXTENSIONS:
                    # Fingerprint berechnen und pruefen
                    fp = self.fingerprinter.compute(str(file_path))
                    if fp and fp not in existing_fps:
                        new_videos.append(str(file_path))

            logger.info(f"Scan von {folder}: {len(new_videos)} neue Videos gefunden")
            return new_videos

        except Exception as e:
            logger.error(f"Fehler beim Ordner-Scan: {e}")
            return []

    def get_clips_by_folder(self) -> dict[str, list[int]]:
        """
        Gruppiert Clips nach ihrem aktuellen Ordner.

        Returns:
            Dict {ordner_pfad: [clip_ids]}
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT id, file_path FROM video_clips WHERE is_available = 1")

            folders: dict[str, list[int]] = {}
            for row in cursor.fetchall():
                if row["file_path"]:
                    folder = str(Path(row["file_path"]).parent)
                    if folder not in folders:
                        folders[folder] = []
                    folders[folder].append(row["id"])

            conn.close()
            return folders

        except Exception as e:
            logger.error(f"Fehler beim Gruppieren nach Ordner: {e}")
            return {}
