"""
FAISS Manager - Vektorbasierte Aehnlichkeitssuche.

Verwendet FAISS (Facebook AI Similarity Search) fuer schnelle
Nearest-Neighbor-Suche in hochdimensionalen Feature-Vektoren.
"""

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...utils.logger import get_logger

logger = get_logger()

# Feature-Vektor Dimension
VECTOR_DIM = 512


@dataclass
class SearchResult:
    """Ergebnis einer Aehnlichkeitssuche."""

    clip_id: int
    similarity: float  # 0-1 (1 = identisch)
    distance: float  # Rohe Distanz


class FAISSManager:
    """Verwaltet FAISS Index fuer Aehnlichkeitssuche."""

    def __init__(self, index_path: str | None = None, mapping_path: str | None = None):
        """
        Args:
            index_path: Pfad zur FAISS Index-Datei
            mapping_path: Pfad zur ID-Mapping-Datei
        """
        self.index_path = Path(index_path) if index_path else Path("data/clip_vectors.faiss")
        self.mapping_path = (
            Path(mapping_path) if mapping_path else Path("data/faiss_clip_mapping.json")
        )

        self._faiss = None
        self._faiss_available = None
        self._index = None
        self._clip_mapping: dict[int, int] = {}  # faiss_id -> clip_id
        self._reverse_mapping: dict[int, int] = {}  # clip_id -> faiss_id
        self._lock = threading.Lock()
        # Lazy Update-Mechanismus fuer bestehende Vektoren
        self._pending_updates: dict[int, np.ndarray] = {}
        self._update_rebuild_threshold: int = 100  # nach 100 Updates Index neu aufbauen
        self._stale_entries = set()  # IDs von veralteten Eintraegen
        self._stale_threshold = 100  # Rebuild bei 100 stale entries
        self._update_count = 0  # Gesamtzahl der Updates

    def _check_faiss(self) -> bool:
        """Prueft ob FAISS verfuegbar ist."""
        if self._faiss_available is None:
            try:
                import faiss

                self._faiss = faiss
                self._faiss_available = True
                logger.info("FAISS verfuegbar")
            except ImportError:
                logger.warning("faiss-cpu nicht installiert - Aehnlichkeitssuche deaktiviert")
                self._faiss_available = False
        return self._faiss_available

    def is_available(self) -> bool:
        """Prueft ob FAISS Manager bereit ist."""
        return self._check_faiss()

    def initialize(self, use_ivf: bool = False, nlist: int = None) -> bool:
        """
        Initialisiert einen neuen FAISS Index.

        Args:
            use_ivf: Verwende IVF Index fuer grosse Datensaetze (>5000)
            nlist: Anzahl Cluster fuer IVF

        Returns:
            True bei Erfolg
        """
        if not self._check_faiss():
            return False

        try:
            with self._lock:
                if use_ivf and nlist:
                    # IVF Index fuer grosse Datensaetze
                    quantizer = self._faiss.IndexFlatIP(VECTOR_DIM)
                    self._index = self._faiss.IndexIVFFlat(
                        quantizer, VECTOR_DIM, nlist, self._faiss.METRIC_INNER_PRODUCT
                    )
                else:
                    # Flat Index (exakte Suche)
                    self._index = self._faiss.IndexFlatIP(VECTOR_DIM)

                self._clip_mapping = {}
                self._reverse_mapping = {}

                logger.info("FAISS Index initialisiert")
                return True

        except Exception as e:
            logger.error(f"Fehler bei Index-Initialisierung: {e}")
            return False

    def add_vector(self, clip_id: int, vector: np.ndarray) -> bool:
        """
        Fuegt einen Vektor zum Index hinzu.

        Args:
            clip_id: ID des Clips
            vector: Feature-Vektor (512-dimensional)

        Returns:
            True bei Erfolg
        """
        if not self._check_faiss():
            return False

        if self._index is None:
            self.initialize()

        try:
            # Vektor vorbereiten
            normalized = self._normalize_vector(vector)
            if normalized is None:
                return False
            vector = normalized.reshape(1, -1)

            with self._lock:
                # Clip bereits im Index?
                if clip_id in self._reverse_mapping:
                    # Update statt Add (lazy rebuild)
                    return self.update_vector(clip_id, normalized)

                # Hinzufuegen
                faiss_id = self._index.ntotal
                self._index.add(vector)

                # Mapping aktualisieren
                self._clip_mapping[faiss_id] = clip_id
                self._reverse_mapping[clip_id] = faiss_id

                return True

        except Exception as e:
            logger.error(f"Fehler beim Hinzufuegen zum Index: {e}")
            return False

    def add_vectors_batch(self, clip_ids: list[int], vectors: np.ndarray) -> int:
        """
        Fuegt mehrere Vektoren auf einmal hinzu.

        Args:
            clip_ids: Liste von Clip-IDs
            vectors: Matrix mit Vektoren (n x 512)

        Returns:
            Anzahl erfolgreich hinzugefuegter Vektoren
        """
        if not self._check_faiss():
            return 0

        if self._index is None:
            self.initialize()

        try:
            vectors = np.asarray(vectors, dtype=np.float32)

            # L2-Normalisierung (PERF-FIX: In-place Division)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vectors /= norms  # In-place statt neuer Allokation

            with self._lock:
                start_id = self._index.ntotal
                self._index.add(vectors)

            # Mapping aktualisieren
            for i, clip_id in enumerate(clip_ids):
                faiss_id = start_id + i
                self._clip_mapping[faiss_id] = clip_id
                self._reverse_mapping[clip_id] = faiss_id
                if clip_id in self._pending_updates:
                    self._pending_updates.pop(clip_id, None)

            return len(clip_ids)

        except Exception as e:
            logger.error(f"Fehler beim Batch-Hinzufuegen: {e}")
            return 0

    def search(self, query_vector: np.ndarray, k: int = 10) -> list[SearchResult]:
        """
        Sucht aehnliche Vektoren.

        Args:
            query_vector: Such-Vektor (512-dimensional)
            k: Anzahl Ergebnisse

        Returns:
            Liste von SearchResult
        """
        if not self._check_faiss() or self._index is None:
            return []

        try:
            # Query vorbereiten
            query = np.asarray(query_vector, dtype=np.float32)
            norm = np.linalg.norm(query)
            if norm > 0:
                query = query / norm
            query = query.reshape(1, -1)

            # Suche
            with self._lock:
                if self._index.ntotal == 0:
                    return []

                k = min(k, self._index.ntotal)
                distances, indices = self._index.search(query, k)

            # Ergebnisse formatieren
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < 0:  # Ungueltig
                    continue

                clip_id = self._clip_mapping.get(int(idx))
                if clip_id is None:
                    continue

                # Inner Product zu Similarity konvertieren (bereits normalisiert)
                similarity = float(dist)  # Bei IP ist dist = similarity
                similarity = max(0.0, min(1.0, similarity))

                results.append(
                    SearchResult(
                        clip_id=clip_id, similarity=round(similarity, 4), distance=float(dist)
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Fehler bei Suche: {e}")
            return []

    def search_by_clip_id(self, clip_id: int, k: int = 10) -> list[SearchResult]:
        """
        Sucht aehnliche Clips zu einem gegebenen Clip.

        Args:
            clip_id: ID des Referenz-Clips
            k: Anzahl Ergebnisse (ohne den Referenz-Clip)

        Returns:
            Liste von SearchResult (ohne Referenz-Clip)
        """
        if clip_id not in self._reverse_mapping:
            logger.warning(f"Clip {clip_id} nicht im Index")
            return []

        try:
            # Vektor des Clips holen
            faiss_id = self._reverse_mapping[clip_id]

            with self._lock:
                vector = self._index.reconstruct(faiss_id)

            # Suche mit k+1 (Referenz-Clip wird gefiltert)
            results = self.search(vector, k + 1)

            # Referenz-Clip entfernen
            results = [r for r in results if r.clip_id != clip_id]

            return results[:k]

        except Exception as e:
            logger.error(f"Fehler bei Clip-Suche: {e}")
            return []

    def remove_clip(self, clip_id: int) -> bool:
        """
        Entfernt einen Clip aus dem Index.

        Hinweis: FAISS unterstuetzt kein direktes Loeschen.
        Der Clip wird aus dem Mapping entfernt.

        Args:
            clip_id: ID des Clips

        Returns:
            True bei Erfolg
        """
        with self._lock:
            if clip_id not in self._reverse_mapping:
                return False

            faiss_id = self._reverse_mapping[clip_id]
            del self._reverse_mapping[clip_id]
            del self._clip_mapping[faiss_id]
            self._pending_updates.pop(clip_id, None)

            return True

    def save(self) -> bool:
        """
        Speichert Index und Mapping.

        Returns:
            True bei Erfolg
        """
        if not self._check_faiss() or self._index is None:
            return False

        try:
            # Verzeichnisse erstellen
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self.mapping_path.parent.mkdir(parents=True, exist_ok=True)

            with self._lock:
                # Ausstehende Updates anwenden bevor gespeichert wird
                if self._pending_updates:
                    self._apply_pending_updates(force=True)

                # Index speichern
                self._faiss.write_index(self._index, str(self.index_path))

                # Mapping speichern
                mapping_data = {
                    "clip_mapping": {str(k): v for k, v in self._clip_mapping.items()},
                    "reverse_mapping": {str(k): v for k, v in self._reverse_mapping.items()},
                }
                with open(self.mapping_path, "w") as f:
                    json.dump(mapping_data, f)

            logger.info(f"FAISS Index gespeichert: {self._index.ntotal} Vektoren")
            return True

        except Exception as e:
            logger.error(f"Fehler beim Speichern: {e}")
            return False

    def load(self) -> bool:
        """
        Laedt Index und Mapping.

        CRITICAL FIX: Dimension-Check beim Laden!
        Bei Dimension-Mismatch (z.B. alte 22D vs neue 27D Embeddings)
        wird der Index verworfen und neu initialisiert.

        Returns:
            True bei Erfolg
        """
        if not self._check_faiss():
            return False

        if not self.index_path.exists():
            logger.info("Kein FAISS Index gefunden - neuer Index wird erstellt")
            return self.initialize()

        try:
            with self._lock:
                # Index laden
                loaded_index = self._faiss.read_index(str(self.index_path))

                # CRITICAL: Dimension-Check vor Verwendung!
                if loaded_index.d != VECTOR_DIM:
                    logger.warning(
                        f"FAISS Index Dimension-Mismatch: "
                        f"Geladen={loaded_index.d}D, Erwartet={VECTOR_DIM}D. "
                        f"Index wird verworfen und neu erstellt."
                    )
                    # Alte Index-Datei loeschen
                    try:
                        self.index_path.unlink()
                        logger.info(f"Alter Index geloescht: {self.index_path}")
                    except OSError as e:
                        logger.warning(f"Konnte alten Index nicht loeschen: {e}")

                    # Neuen Index initialisieren
                    return self.initialize()

                self._index = loaded_index

                # Mapping laden
                if self.mapping_path.exists():
                    with open(self.mapping_path) as f:
                        mapping_data = json.load(f)

                    self._clip_mapping = {
                        int(k): v for k, v in mapping_data.get("clip_mapping", {}).items()
                    }
                    self._reverse_mapping = {
                        int(k): v for k, v in mapping_data.get("reverse_mapping", {}).items()
                    }
                else:
                    self._clip_mapping = {}
                    self._reverse_mapping = {}
                self._pending_updates = {}

            logger.info(f"FAISS Index geladen: {self._index.ntotal} Vektoren, {self._index.d}D")
            return True

        except Exception as e:
            logger.error(f"Fehler beim Laden: {e}")
            return self.initialize()

    def get_stats(self) -> dict:
        """Gibt Index-Statistiken zurueck."""
        if self._index is None:
            return {"total_vectors": 0, "dimension": VECTOR_DIM, "is_trained": False}

        return {
            "total_vectors": self._index.ntotal,
            "dimension": VECTOR_DIM,
            "is_trained": getattr(self._index, "is_trained", True),
            "index_path": str(self.index_path),
            "clips_mapped": len(self._clip_mapping),
            "pending_updates": len(self._pending_updates),
            "stale_entries": len(self._stale_entries),
            "update_count": self._update_count,
            "needs_rebuild": len(self._stale_entries) >= self._stale_threshold,
        }

    def get_index_stats(self) -> dict:
        """
        Erweiterte Index-Statistiken.

        Returns:
            Dict mit detaillierten Statistiken
        """
        return {
            "total_entries": self._index.ntotal if self._index else 0,
            "stale_entries": len(self._stale_entries),
            "update_count": self._update_count,
            "needs_rebuild": len(self._stale_entries) >= self._stale_threshold,
            "pending_updates": len(self._pending_updates),
            "clips_mapped": len(self._clip_mapping),
        }

    def update_vector(self, clip_id: int, vector: np.ndarray, apply_now: bool = False) -> bool:
        """
        Aktualisiert einen vorhandenen Vektor (lazy Rebuild).

        Args:
            clip_id: Clip-ID
            vector: Neuer Feature-Vektor (512D)
            apply_now: Bei True sofortigen Rebuild erzwingen
        """
        if not self._check_faiss() or self._index is None:
            return False

        normalized = self._normalize_vector(vector)
        if normalized is None:
            return False

        with self._lock:
            if clip_id not in self._reverse_mapping:
                # Nicht vorhanden -> wie Add behandeln
                return self.add_vector(clip_id, normalized)

            self._pending_updates[clip_id] = normalized

            threshold = self._update_rebuild_threshold
            if apply_now or len(self._pending_updates) >= threshold:
                return self._apply_pending_updates(force=True)

        return True

    def _apply_pending_updates(self, force: bool = False) -> bool:
        """
        Wendet alle pending updates an, indem der Index neu aufgebaut wird.
        """
        if not self._pending_updates:
            return True

        try:
            # Alle bestehenden Vektoren rekonstruieren
            vectors: dict[int, np.ndarray] = {}
            for faiss_id, clip_id in self._clip_mapping.items():
                # Wenn ein Update existiert, spaeter ueberschreiben
                if clip_id in self._pending_updates:
                    continue
                vec = self._index.reconstruct(faiss_id)
                vectors[clip_id] = vec

            # Pending-Updates einfuegen
            for clip_id, vec in self._pending_updates.items():
                vectors[clip_id] = vec

            return self.rebuild_index(vectors)
        finally:
            if force:
                self._pending_updates = {}

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray | None:
        """Validiert und normalisiert einen Vektor."""
        try:
            vector = np.asarray(vector, dtype=np.float32)
            if vector.shape != (VECTOR_DIM,):
                logger.warning(f"Ungueltige Vektor-Dimension: {vector.shape}")
                return None
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            return vector
        except Exception as exc:
            logger.warning(f"Vektor-Normalisierung fehlgeschlagen: {exc}")
            return None

    def rebuild_index(self, vectors: dict[int, np.ndarray], batch_size: int = 10000) -> bool:
        """
        Baut den Index komplett neu auf mit Batch-Processing.

        PERF-FIX: Verwendet Batch-Processing um Memory-Spitzen zu vermeiden.
        Bei 100.000 Clips mit 512D Vektoren: 200MB statt 2GB peak memory.

        Args:
            vectors: Dict {clip_id: vector}
            batch_size: Anzahl Vektoren pro Batch (default: 10000 = ~40MB)

        Returns:
            True bei Erfolg
        """
        try:
            # Neuen Index erstellen
            use_ivf = len(vectors) > 5000
            nlist = int(np.sqrt(len(vectors))) if use_ivf else None
            self.initialize(use_ivf=use_ivf, nlist=nlist)

            clip_ids = list(vectors.keys())
            total_vectors = len(clip_ids)

            # Training bei IVF (benötigt Sample der Daten)
            if use_ivf and nlist:
                # Sample für Training (max 50k Vektoren)
                sample_size = min(50000, total_vectors)
                sample_ids = clip_ids[:sample_size]
                training_vectors = np.array([vectors[cid] for cid in sample_ids], dtype=np.float32)
                norms = np.linalg.norm(training_vectors, axis=1, keepdims=True)
                norms[norms == 0] = 1
                training_vectors /= norms  # In-place division
                self._index.train(training_vectors)
                del training_vectors  # Explizites Cleanup

            # PERF-FIX: Batch-Processing für Vektoren hinzufügen
            for batch_start in range(0, total_vectors, batch_size):
                batch_end = min(batch_start + batch_size, total_vectors)
                batch_ids = clip_ids[batch_start:batch_end]

                # Batch-Array erstellen und normalisieren
                batch_vectors = np.array([vectors[cid] for cid in batch_ids], dtype=np.float32)

                self.add_vectors_batch(batch_ids, batch_vectors)

                # Explizites Cleanup nach jedem Batch
                del batch_vectors

                if batch_end < total_vectors:
                    logger.debug(f"Index rebuild progress: {batch_end}/{total_vectors}")

            self._pending_updates = {}
            logger.info(
                f"Index neu aufgebaut mit {total_vectors} Vektoren (batch_size={batch_size})"
            )
            return True

        except Exception as e:
            logger.error(f"Fehler beim Index-Rebuild: {e}")
            return False

    def add_or_update(self, clip_id: int, embedding: np.ndarray) -> bool:
        """
        Fuegt Clip hinzu oder markiert fuer Update.

        Verwendet Lazy Update-Strategie: Bestehende Vektoren werden als "stale"
        markiert und beim naechsten Rebuild ersetzt.

        Args:
            clip_id: Clip-ID
            embedding: Feature-Vektor (512D)

        Returns:
            True bei Erfolg
        """
        if not self._check_faiss():
            return False

        if self._index is None:
            self.initialize()

        normalized = self._normalize_vector(embedding)
        if normalized is None:
            return False

        with self._lock:
            if clip_id in self._reverse_mapping:
                # Markiere als stale fuer spaeteren Rebuild
                self._stale_entries.add(clip_id)
                self._pending_updates[clip_id] = normalized
                self._update_count += 1
                logger.debug(f"Clip {clip_id} marked as stale for update")

                # Trigger Rebuild wenn Threshold erreicht
                if len(self._stale_entries) >= self._stale_threshold:
                    logger.info(
                        f"Stale threshold reached ({len(self._stale_entries)}), rebuilding index"
                    )
                    return self._rebuild_index()

                return True

            # Normales Add fuer neue Clips
            return self._add_embedding(clip_id, normalized)

    def _add_embedding(self, clip_id: int, embedding: np.ndarray) -> bool:
        """
        Interner Helper zum Hinzufuegen eines neuen Embeddings.

        Args:
            clip_id: Clip-ID
            embedding: Normalisierter Feature-Vektor

        Returns:
            True bei Erfolg
        """
        try:
            vector = embedding.reshape(1, -1)

            # Hinzufuegen
            faiss_id = self._index.ntotal
            self._index.add(vector)

            # Mapping aktualisieren
            self._clip_mapping[faiss_id] = clip_id
            self._reverse_mapping[clip_id] = faiss_id

            return True

        except Exception as e:
            logger.error(f"Fehler beim Hinzufuegen zum Index: {e}")
            return False

    def _rebuild_index(self) -> bool:
        """
        Rebuild Index ohne stale entries.

        Returns:
            True bei Erfolg
        """
        if not self._stale_entries:
            return True

        logger.info(f"Rebuilding FAISS index ({len(self._stale_entries)} stale entries)")

        try:
            # Sammle alle aktuellen Embeddings (außer stale)
            vectors: dict[int, np.ndarray] = {}

            for faiss_id, clip_id in self._clip_mapping.items():
                if clip_id in self._stale_entries:
                    # Nimm das aktualisierte Embedding aus pending_updates
                    if clip_id in self._pending_updates:
                        vectors[clip_id] = self._pending_updates[clip_id]
                else:
                    # Behalte existierendes Embedding
                    vec = self._index.reconstruct(faiss_id)
                    vectors[clip_id] = vec

            # Rebuild mit allen Vektoren
            success = self.rebuild_index(vectors)

            if success:
                # Stats zuruecksetzen
                self._stale_entries.clear()
                logger.info(f"FAISS index rebuilt with {len(vectors)} entries")

            return success

        except Exception as e:
            logger.error(f"Fehler beim Index-Rebuild: {e}")
            return False

    def force_update(self, clip_id: int, embedding: np.ndarray) -> bool:
        """
        Erzwinge sofortiges Update eines Clips.

        Im Gegensatz zu add_or_update wird hier sofort ein Rebuild ausgeloest,
        unabhaengig vom Threshold.

        Args:
            clip_id: Clip-ID
            embedding: Feature-Vektor (512D)

        Returns:
            True bei Erfolg
        """
        if not self._check_faiss():
            return False

        if self._index is None:
            self.initialize()

        normalized = self._normalize_vector(embedding)
        if normalized is None:
            return False

        with self._lock:
            if clip_id in self._reverse_mapping:
                self._stale_entries.add(clip_id)
                self._pending_updates[clip_id] = normalized
                self._update_count += 1

            # Fuege neues Embedding hinzu (falls noch nicht vorhanden)
            if clip_id not in self._reverse_mapping:
                success = self._add_embedding(clip_id, normalized)
                if not success:
                    return False

            # Sofortiger Rebuild
            if len(self._stale_entries) > 0:
                return self._rebuild_index()

            return True
