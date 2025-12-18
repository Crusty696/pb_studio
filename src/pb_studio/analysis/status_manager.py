"""
Analyse-Status Manager.

Verwaltet den Analyse-Status aller Clips:
- Welche Analysen wurden durchgefuehrt?
- Welche Versionen der Algorithmen wurden verwendet?
- Welche Clips muessen (neu) analysiert werden?
"""

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..database.connection import get_db_manager
from ..utils.logger import get_logger

logger = get_logger()


# Aktuelle Algorithmus-Versionen
ANALYSIS_VERSIONS = {
    "colors": 1,
    "motion": 1,
    "scene": 1,
    "mood": 1,
    "objects": 1,
    "style": 1,
    "fingerprint": 1,
    "vector": 1,
}


@dataclass
class ClipAnalysisStatus:
    """Status-Daten fuer einen Clip."""

    clip_id: int
    colors_analyzed: bool = False
    motion_analyzed: bool = False
    scene_analyzed: bool = False
    mood_analyzed: bool = False
    objects_analyzed: bool = False
    style_analyzed: bool = False
    fingerprint_created: bool = False
    vector_extracted: bool = False

    colors_version: int = 0
    motion_version: int = 0
    scene_version: int = 0
    mood_version: int = 0
    objects_version: int = 0
    style_version: int = 0

    last_full_analysis: datetime | None = None

    @property
    def is_fully_analyzed(self) -> bool:
        """Prueft ob alle Analysen durchgefuehrt wurden."""
        return all(
            [
                self.colors_analyzed,
                self.motion_analyzed,
                self.scene_analyzed,
                self.mood_analyzed,
                self.objects_analyzed,
                self.style_analyzed,
                self.fingerprint_created,
                self.vector_extracted,
            ]
        )

    @property
    def needs_update(self) -> bool:
        """Prueft ob Analysen aktualisiert werden muessen."""
        return any(
            [
                self.colors_version < ANALYSIS_VERSIONS["colors"],
                self.motion_version < ANALYSIS_VERSIONS["motion"],
                self.scene_version < ANALYSIS_VERSIONS["scene"],
                self.mood_version < ANALYSIS_VERSIONS["mood"],
                self.objects_version < ANALYSIS_VERSIONS["objects"],
                self.style_version < ANALYSIS_VERSIONS["style"],
            ]
        )

    def get_missing_analyses(self) -> list[str]:
        """Gibt Liste fehlender Analysen zurueck."""
        missing = []
        if not self.colors_analyzed:
            missing.append("colors")
        if not self.motion_analyzed:
            missing.append("motion")
        if not self.scene_analyzed:
            missing.append("scene")
        if not self.mood_analyzed:
            missing.append("mood")
        if not self.objects_analyzed:
            missing.append("objects")
        if not self.style_analyzed:
            missing.append("style")
        if not self.fingerprint_created:
            missing.append("fingerprint")
        if not self.vector_extracted:
            missing.append("vector")
        return missing

    def get_outdated_analyses(self) -> list[str]:
        """Gibt Liste veralteter Analysen zurueck."""
        outdated = []
        if self.colors_version < ANALYSIS_VERSIONS["colors"]:
            outdated.append("colors")
        if self.motion_version < ANALYSIS_VERSIONS["motion"]:
            outdated.append("motion")
        if self.scene_version < ANALYSIS_VERSIONS["scene"]:
            outdated.append("scene")
        if self.mood_version < ANALYSIS_VERSIONS["mood"]:
            outdated.append("mood")
        if self.objects_version < ANALYSIS_VERSIONS["objects"]:
            outdated.append("objects")
        if self.style_version < ANALYSIS_VERSIONS["style"]:
            outdated.append("style")
        return outdated


class AnalysisStatusManager:
    """Verwaltet Analyse-Status fuer alle Clips."""

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

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context Manager fuer SQLite-Verbindung.

        FIX: Connection Leak behoben durch Context Manager.
        Garantiert dass Connection auch bei Exceptions geschlossen wird.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def get_status(self, clip_id: int) -> ClipAnalysisStatus | None:
        """
        Holt Analyse-Status fuer einen Clip.

        Args:
            clip_id: ID des Clips

        Returns:
            ClipAnalysisStatus oder None
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT * FROM clip_analysis_status WHERE clip_id = ?
                """,
                    (clip_id,),
                )

                row = cursor.fetchone()

                if row is None:
                    return None

                return ClipAnalysisStatus(
                    clip_id=row["clip_id"],
                    colors_analyzed=bool(row["colors_analyzed"]),
                    motion_analyzed=bool(row["motion_analyzed"]),
                    scene_analyzed=bool(row["scene_analyzed"]),
                    mood_analyzed=bool(row["mood_analyzed"]),
                    objects_analyzed=bool(row["objects_analyzed"]),
                    style_analyzed=bool(row["style_analyzed"]),
                    fingerprint_created=bool(row["fingerprint_created"]),
                    vector_extracted=bool(row["vector_extracted"]),
                    colors_version=row["colors_version"] or 0,
                    motion_version=row["motion_version"] or 0,
                    scene_version=row["scene_version"] or 0,
                    mood_version=row["mood_version"] or 0,
                    objects_version=row["objects_version"] or 0,
                    style_version=row["style_version"] or 0,
                    last_full_analysis=row["last_full_analysis"],
                )

        except Exception as e:
            logger.error(f"Fehler beim Laden des Status fuer Clip {clip_id}: {e}")
            return None

    def create_status(self, clip_id: int) -> bool:
        """
        Erstellt initialen Status-Eintrag fuer einen neuen Clip.

        Args:
            clip_id: ID des Clips

        Returns:
            True bei Erfolg
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT OR IGNORE INTO clip_analysis_status (clip_id)
                    VALUES (?)
                """,
                    (clip_id,),
                )

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Fehler beim Erstellen des Status fuer Clip {clip_id}: {e}")
            return False

    def update_status(self, clip_id: int, analysis_type: str, success: bool = True) -> bool:
        """
        Aktualisiert Status fuer eine spezifische Analyse.

        Args:
            clip_id: ID des Clips
            analysis_type: Typ der Analyse (colors, motion, scene, mood, objects, style, fingerprint, vector)
            success: Ob die Analyse erfolgreich war

        Returns:
            True bei Erfolg
        """
        valid_types = [
            "colors",
            "motion",
            "scene",
            "mood",
            "objects",
            "style",
            "fingerprint",
            "vector",
        ]
        if analysis_type not in valid_types:
            logger.error(f"Ungueltiger Analyse-Typ: {analysis_type}")
            return False

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Stelle sicher dass Eintrag existiert
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO clip_analysis_status (clip_id)
                    VALUES (?)
                """,
                    (clip_id,),
                )

                # Analysiert-Flag setzen
                analyzed_col = f"{analysis_type}_analyzed"
                if analysis_type in ["fingerprint", "vector"]:
                    analyzed_col = (
                        f"{analysis_type}_created"
                        if analysis_type == "fingerprint"
                        else "vector_extracted"
                    )

                cursor.execute(
                    f"""
                    UPDATE clip_analysis_status
                    SET {analyzed_col} = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE clip_id = ?
                """,
                    (1 if success else 0, clip_id),
                )

                # Version aktualisieren (nur fuer Haupt-Analysen)
                if analysis_type in ANALYSIS_VERSIONS and analysis_type not in [
                    "fingerprint",
                    "vector",
                ]:
                    version_col = f"{analysis_type}_version"
                    cursor.execute(
                        f"""
                        UPDATE clip_analysis_status
                        SET {version_col} = ?
                        WHERE clip_id = ?
                    """,
                        (ANALYSIS_VERSIONS[analysis_type], clip_id),
                    )

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Fehler beim Update des Status fuer Clip {clip_id}: {e}")
            return False

    def mark_full_analysis_complete(self, clip_id: int) -> bool:
        """
        Markiert vollstaendige Analyse als abgeschlossen.

        Args:
            clip_id: ID des Clips

        Returns:
            True bei Erfolg
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    UPDATE clip_analysis_status
                    SET last_full_analysis = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE clip_id = ?
                """,
                    (clip_id,),
                )

                # Auch needs_reanalysis in video_clips zuruecksetzen
                cursor.execute(
                    """
                    UPDATE video_clips
                    SET needs_reanalysis = 0
                    WHERE id = ?
                """,
                    (clip_id,),
                )

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Fehler beim Markieren der vollstaendigen Analyse: {e}")
            return False

    def get_unanalyzed_clips(self) -> list[int]:
        """
        Gibt Liste aller Clips zurueck die analysiert werden muessen.

        Returns:
            Liste von Clip-IDs
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT vc.id FROM video_clips vc
                    LEFT JOIN clip_analysis_status cas ON vc.id = cas.clip_id
                    WHERE vc.needs_reanalysis = 1
                       OR vc.is_available = 1 AND (
                           cas.clip_id IS NULL
                           OR cas.colors_analyzed = 0
                           OR cas.motion_analyzed = 0
                           OR cas.scene_analyzed = 0
                           OR cas.mood_analyzed = 0
                           OR cas.style_analyzed = 0
                           OR cas.fingerprint_created = 0
                       )
                    ORDER BY vc.id
                """
                )

                clip_ids = [row["id"] for row in cursor.fetchall()]
                return clip_ids

        except Exception as e:
            logger.error(f"Fehler beim Laden unanalysierter Clips: {e}")
            return []

    def get_outdated_clips(self) -> list[int]:
        """
        Gibt Liste aller Clips mit veralteten Analysen zurueck.

        Returns:
            Liste von Clip-IDs
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT clip_id FROM clip_analysis_status
                    WHERE colors_version < ?
                       OR motion_version < ?
                       OR scene_version < ?
                       OR mood_version < ?
                       OR objects_version < ?
                       OR style_version < ?
                """,
                    (
                        ANALYSIS_VERSIONS["colors"],
                        ANALYSIS_VERSIONS["motion"],
                        ANALYSIS_VERSIONS["scene"],
                        ANALYSIS_VERSIONS["mood"],
                        ANALYSIS_VERSIONS["objects"],
                        ANALYSIS_VERSIONS["style"],
                    ),
                )

                clip_ids = [row["clip_id"] for row in cursor.fetchall()]
                return clip_ids

        except Exception as e:
            logger.error(f"Fehler beim Laden veralteter Clips: {e}")
            return []

    def get_analysis_statistics(self) -> dict:
        """
        Gibt Statistiken ueber den Analyse-Status zurueck.

        Returns:
            Dict mit Statistiken
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Gesamtanzahl Clips
                cursor.execute("SELECT COUNT(*) as total FROM video_clips WHERE is_available = 1")
                total = cursor.fetchone()["total"]

                # Vollstaendig analysierte Clips
                cursor.execute(
                    """
                    SELECT COUNT(*) as analyzed FROM clip_analysis_status
                    WHERE colors_analyzed = 1
                      AND motion_analyzed = 1
                      AND scene_analyzed = 1
                      AND mood_analyzed = 1
                      AND style_analyzed = 1
                      AND fingerprint_created = 1
                """
                )
                fully_analyzed = cursor.fetchone()["analyzed"]

                # Einzelne Analysen
                stats = {
                    "total_clips": total,
                    "fully_analyzed": fully_analyzed,
                    "pending": total - fully_analyzed,
                    "percentage": round(fully_analyzed / total * 100, 1) if total > 0 else 0,
                }

                # Details pro Analyse-Typ
                for analysis_type in ["colors", "motion", "scene", "mood", "objects", "style"]:
                    col = f"{analysis_type}_analyzed"
                    # Use parameterized query for the value 1
                    cursor.execute(
                        f"SELECT COUNT(*) as cnt FROM clip_analysis_status WHERE {col} = ?", (1,)
                    )
                    stats[f"{analysis_type}_count"] = cursor.fetchone()["cnt"]

                return stats

        except Exception as e:
            logger.error(f"Fehler beim Laden der Statistiken: {e}")
            return {"total_clips": 0, "fully_analyzed": 0, "pending": 0, "percentage": 0}

    def reset_clip_analysis(self, clip_id: int) -> bool:
        """
        Setzt Analyse-Status fuer einen Clip zurueck.

        Args:
            clip_id: ID des Clips

        Returns:
            True bei Erfolg
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Status zuruecksetzen
                cursor.execute(
                    """
                    UPDATE clip_analysis_status
                    SET colors_analyzed = 0,
                        motion_analyzed = 0,
                        scene_analyzed = 0,
                        mood_analyzed = 0,
                        objects_analyzed = 0,
                        style_analyzed = 0,
                        fingerprint_created = 0,
                        vector_extracted = 0,
                        colors_version = 0,
                        motion_version = 0,
                        scene_version = 0,
                        mood_version = 0,
                        objects_version = 0,
                        style_version = 0,
                        last_full_analysis = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE clip_id = ?
                """,
                    (clip_id,),
                )

                # needs_reanalysis setzen
                cursor.execute(
                    """
                    UPDATE video_clips
                    SET needs_reanalysis = 1
                    WHERE id = ?
                """,
                    (clip_id,),
                )

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Fehler beim Zuruecksetzen des Status: {e}")
            return False

    def reset_all_analyses(self) -> int:
        """
        Setzt Analyse-Status fuer alle Clips zurueck.

        Returns:
            Anzahl zurueckgesetzter Clips
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    UPDATE clip_analysis_status
                    SET colors_analyzed = 0,
                        motion_analyzed = 0,
                        scene_analyzed = 0,
                        mood_analyzed = 0,
                        objects_analyzed = 0,
                        style_analyzed = 0,
                        fingerprint_created = 0,
                        vector_extracted = 0,
                        colors_version = 0,
                        motion_version = 0,
                        scene_version = 0,
                        mood_version = 0,
                        objects_version = 0,
                        style_version = 0,
                        last_full_analysis = NULL,
                        updated_at = CURRENT_TIMESTAMP
                """
                )
                count = cursor.rowcount

                cursor.execute("UPDATE video_clips SET needs_reanalysis = 1")

                conn.commit()

                logger.info(f"Analyse-Status fuer {count} Clips zurueckgesetzt")
                return count

        except Exception as e:
            logger.error(f"Fehler beim Zuruecksetzen aller Analysen: {e}")
            return 0
