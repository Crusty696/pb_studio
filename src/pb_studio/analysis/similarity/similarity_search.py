"""
Similarity Search API - Aehnlichkeitssuche fuer Clips.

Kombiniert FAISS-Vektorsuche mit Perceptual Hashing fuer
umfassende Aehnlichkeitserkennung.
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from ...database.connection import get_db_manager
from ...utils.logger import get_logger
from ..analyzers.feature_extractor import FeatureExtractor, FeatureVector
from ..fingerprint import ContentFingerprint, PerceptualHash
from .faiss_manager import FAISSManager

logger = get_logger()


@dataclass
class SimilarityResult:
    """Vollstaendiges Aehnlichkeits-Ergebnis."""

    clip_id: int
    file_path: str
    similarity_score: float  # 0-1
    match_type: str  # 'exact', 'near_duplicate', 'similar'
    details: dict  # Zusaetzliche Infos


class SimilaritySearch:
    """Haupt-API fuer Aehnlichkeitssuche."""

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

        self.faiss_manager = FAISSManager()
        self.feature_extractor = FeatureExtractor()
        self.fingerprinter = ContentFingerprint()
        self.perceptual_hash = PerceptualHash()

        # Schwellwerte
        self.exact_threshold = 0.99
        self.near_duplicate_threshold = 0.90
        self.similar_threshold = 0.70

    def _get_connection(self) -> sqlite3.Connection:
        """Erstellt eine SQLite-Verbindung."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> bool:
        """
        Initialisiert die Similarity Search.

        Returns:
            True bei Erfolg
        """
        return self.faiss_manager.load()

    def find_similar(
        self, clip_id: int, k: int = 10, min_similarity: float = 0.5
    ) -> list[SimilarityResult]:
        """
        Findet aehnliche Clips zu einem gegebenen Clip.

        Args:
            clip_id: ID des Referenz-Clips
            k: Maximale Anzahl Ergebnisse
            min_similarity: Minimale Aehnlichkeit (0-1)

        Returns:
            Liste von SimilarityResult
        """
        try:
            # FAISS-Suche
            faiss_results = self.faiss_manager.search_by_clip_id(
                clip_id, k * 2
            )  # Mehr holen fuer Filter

            results = []
            for result in faiss_results:
                if result.similarity < min_similarity:
                    continue

                # Clip-Info laden
                clip_info = self._get_clip_info(result.clip_id)
                if clip_info is None:
                    continue

                # Match-Typ bestimmen
                match_type = self._classify_match(result.similarity)

                results.append(
                    SimilarityResult(
                        clip_id=result.clip_id,
                        file_path=clip_info["file_path"],
                        similarity_score=result.similarity,
                        match_type=match_type,
                        details={
                            "faiss_distance": result.distance,
                            "filename": clip_info.get("filename", ""),
                            "duration": clip_info.get("duration_seconds", 0),
                        },
                    )
                )

                if len(results) >= k:
                    break

            return results

        except Exception as e:
            logger.error(f"Fehler bei Aehnlichkeitssuche: {e}")
            return []

    def find_similar_to_video(self, video_path: str, k: int = 10) -> list[SimilarityResult]:
        """
        Findet aehnliche Clips zu einem externen Video (nicht in DB).

        Args:
            video_path: Pfad zum Video
            k: Anzahl Ergebnisse

        Returns:
            Liste von SimilarityResult
        """
        try:
            # Feature-Vektor extrahieren
            feature_vector = self.feature_extractor.extract_from_video(video_path)
            if feature_vector is None:
                logger.warning(f"Feature-Extraktion fehlgeschlagen: {video_path}")
                return []

            # FAISS-Suche
            faiss_results = self.faiss_manager.search(feature_vector.vector, k)

            results = []
            for result in faiss_results:
                clip_info = self._get_clip_info(result.clip_id)
                if clip_info is None:
                    continue

                match_type = self._classify_match(result.similarity)

                results.append(
                    SimilarityResult(
                        clip_id=result.clip_id,
                        file_path=clip_info["file_path"],
                        similarity_score=result.similarity,
                        match_type=match_type,
                        details={
                            "filename": clip_info.get("filename", ""),
                            "duration": clip_info.get("duration_seconds", 0),
                        },
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Fehler bei Video-Aehnlichkeitssuche: {e}")
            return []

    def find_duplicates(self, threshold: float = 0.95) -> list[list[int]]:
        """
        Findet Duplikate in der gesamten Clip-Bibliothek.

        Args:
            threshold: Minimale Aehnlichkeit fuer Duplikat

        Returns:
            Liste von Duplikat-Gruppen (Clip-IDs)
        """
        try:
            # Alle Clips laden
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM video_clips WHERE is_available = 1")
            clip_ids = [row["id"] for row in cursor.fetchall()]
            conn.close()

            if len(clip_ids) < 2:
                return []

            # Duplikat-Gruppen finden
            seen = set()
            duplicate_groups = []

            for clip_id in clip_ids:
                if clip_id in seen:
                    continue

                # Aehnliche suchen
                similar = self.faiss_manager.search_by_clip_id(clip_id, 10)

                # Duplikate filtern
                duplicates = [clip_id]
                for result in similar:
                    if result.similarity >= threshold and result.clip_id not in seen:
                        duplicates.append(result.clip_id)
                        seen.add(result.clip_id)

                if len(duplicates) > 1:
                    duplicate_groups.append(duplicates)
                    seen.add(clip_id)

            logger.info(f"{len(duplicate_groups)} Duplikat-Gruppen gefunden")
            return duplicate_groups

        except Exception as e:
            logger.error(f"Fehler bei Duplikat-Suche: {e}")
            return []

    def find_exact_duplicate(self, video_path: str) -> int | None:
        """
        Prueft ob exaktes Duplikat bereits existiert (via Fingerprint).

        Args:
            video_path: Pfad zum Video

        Returns:
            Clip-ID des Duplikats oder None
        """
        try:
            # Content Fingerprint berechnen
            fingerprint = self.fingerprinter.compute(video_path)
            if fingerprint is None:
                return None

            # In DB suchen
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id FROM video_clips
                WHERE content_fingerprint = ?
            """,
                (fingerprint,),
            )
            row = cursor.fetchone()
            conn.close()

            return row["id"] if row else None

        except Exception as e:
            logger.error(f"Fehler bei Duplikat-Pruefung: {e}")
            return None

    def add_clip_to_index(self, clip_id: int, video_path: str) -> bool:
        """
        Fuegt einen Clip zum Similarity-Index hinzu.

        Args:
            clip_id: ID des Clips
            video_path: Pfad zum Video

        Returns:
            True bei Erfolg
        """
        try:
            # Feature-Vektor extrahieren
            feature_vector = self.feature_extractor.extract_from_video(video_path)
            if feature_vector is None:
                logger.warning(f"Feature-Extraktion fehlgeschlagen: {video_path}")
                return False

            # Zu FAISS hinzufuegen
            success = self.faiss_manager.add_vector(clip_id, feature_vector.vector)

            if success:
                # Vektor auch in DB speichern (fuer fingerprints Tabelle)
                self._save_vector_to_db(clip_id, feature_vector)

            return success

        except Exception as e:
            logger.error(f"Fehler beim Hinzufuegen zum Index: {e}")
            return False

    def remove_clip_from_index(self, clip_id: int) -> bool:
        """Entfernt einen Clip aus dem Index."""
        return self.faiss_manager.remove_clip(clip_id)

    def save_index(self) -> bool:
        """Speichert den Index."""
        return self.faiss_manager.save()

    def get_statistics(self) -> dict:
        """Gibt Statistiken zurueck."""
        return self.faiss_manager.get_stats()

    def _classify_match(self, similarity: float) -> str:
        """Klassifiziert Match-Typ basierend auf Aehnlichkeit."""
        if similarity >= self.exact_threshold:
            return "exact"
        elif similarity >= self.near_duplicate_threshold:
            return "near_duplicate"
        elif similarity >= self.similar_threshold:
            return "similar"
        else:
            return "low_similarity"

    def _get_clip_info(self, clip_id: int) -> dict | None:
        """Laedt Clip-Informationen aus DB."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, file_path, duration_seconds
                FROM video_clips WHERE id = ?
            """,
                (clip_id,),
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                return {
                    "id": row["id"],
                    "file_path": row["file_path"],
                    "filename": Path(row["file_path"]).name if row["file_path"] else "",
                    "duration_seconds": row["duration_seconds"],
                }
            return None

        except Exception as e:
            logger.error(f"Fehler beim Laden der Clip-Info: {e}")
            return None

    def _save_vector_to_db(self, clip_id: int, vector: FeatureVector) -> bool:
        """Speichert Vektor-Referenz in DB."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # FAISS ID aus Manager
            faiss_id = self.faiss_manager._reverse_mapping.get(clip_id)

            cursor.execute(
                """
                INSERT OR REPLACE INTO clip_fingerprints
                (clip_id, faiss_index_id, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
                (clip_id, faiss_id),
            )

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"Fehler beim Speichern der Vektor-Referenz: {e}")
            return False

    def rebuild_index_from_db(self, progress_callback=None) -> bool:
        """
        Baut den Index aus allen Clips in der DB neu auf.

        Args:
            progress_callback: Callback(current, total)

        Returns:
            True bei Erfolg
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, file_path FROM video_clips
                WHERE is_available = 1
            """
            )
            clips = cursor.fetchall()
            conn.close()

            if not clips:
                logger.info("Keine Clips fuer Index-Aufbau")
                return True

            # Neuen Index initialisieren
            use_ivf = len(clips) > 5000
            self.faiss_manager.initialize(use_ivf=use_ivf)

            # Vektoren extrahieren und hinzufuegen
            for i, clip in enumerate(clips):
                try:
                    if clip["file_path"] and Path(clip["file_path"]).exists():
                        self.add_clip_to_index(clip["id"], clip["file_path"])
                except Exception as e:
                    logger.warning(f"Clip {clip['id']} uebersprungen: {e}")

                if progress_callback:
                    progress_callback(i + 1, len(clips))

            # Speichern
            self.save_index()

            logger.info(
                f"Index mit {self.faiss_manager._index.ntotal if self.faiss_manager._index else 0} Vektoren aufgebaut"
            )
            return True

        except Exception as e:
            logger.error(f"Fehler beim Index-Aufbau: {e}")
            return False

    def compare_clips(self, clip_id_1: int, clip_id_2: int) -> float | None:
        """
        Vergleicht zwei spezifische Clips.

        Args:
            clip_id_1: Erste Clip-ID
            clip_id_2: Zweite Clip-ID

        Returns:
            Aehnlichkeit 0-1 oder None bei Fehler
        """
        try:
            # Beide Clips muessen im Index sein
            if clip_id_1 not in self.faiss_manager._reverse_mapping:
                return None
            if clip_id_2 not in self.faiss_manager._reverse_mapping:
                return None

            # Vektoren holen
            faiss_id_1 = self.faiss_manager._reverse_mapping[clip_id_1]
            faiss_id_2 = self.faiss_manager._reverse_mapping[clip_id_2]

            v1 = self.faiss_manager._index.reconstruct(faiss_id_1)
            v2 = self.faiss_manager._index.reconstruct(faiss_id_2)

            # Cosine Similarity (bereits normalisiert)
            similarity = float(v1 @ v2)
            return max(0.0, min(1.0, similarity))

        except Exception as e:
            logger.error(f"Fehler beim Vergleich: {e}")
            return None
