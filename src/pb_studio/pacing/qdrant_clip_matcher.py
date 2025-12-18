"""
Qdrant-basierter Clip Matcher für AMD GPU / optimierte CPU Performance.

Alternative zu FAISS für Systeme ohne NVIDIA GPU.
Qdrant ist CPU-basiert aber 1.5-3x schneller als FAISS CPU dank HNSW-Index.

Performance:
- Index Build: O(n log n) - ~200ms für 203 Clips
- Query: O(log n) - ~10-30ms pro Query
- Total für 200 Cuts: ~2-6 Sekunden

Features:
- Schneller als FAISS CPU
- Funktioniert auf allen Plattformen
- In-Memory Mode (kein Server nötig)
- Optional: Persistenz für große Datasets

Author: PB_studio Development Team
"""

import random
import time
from typing import Any

import numpy as np

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointStruct,
        VectorParams,
    )

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

from ..utils.logger import get_logger
from .motion_embedding import (
    create_query_vector,
    extract_motion_embedding,
    get_embedding_dimension,
    validate_embedding,
)

logger = get_logger(__name__)


class QdrantClipMatcher:
    """
    Qdrant-basierter Clip Matcher für AMD GPU / optimierte CPU Performance.

    Vorteile vs FAISS CPU:
    - 1.5-3x schnellere Queries (HNSW vs IVF)
    - Bessere Speicher-Effizienz
    - Einfachere API
    - Funktioniert auf allen Plattformen

    Example:
        >>> matcher = QdrantClipMatcher(use_persistence=False)
        >>> matcher.build_index(clips)  # 203 clips
        >>> clip_id, file_path, dist = matcher.find_best_clip(
        ...     target_motion_score=0.8,
        ...     target_energy=0.7,
        ...     target_motion_type='FAST',
        ...     target_moods=['ENERGETIC']
        ... )
    """

    COLLECTION_NAME = "video_clips"

    def __init__(self, use_persistence: bool = False, persist_path: str = "./qdrant_data"):
        """
        Initialisiert Qdrant Matcher.

        Args:
            use_persistence: Speichere Index auf Disk (für große Datasets)
            persist_path: Pfad für persistente Speicherung

        Raises:
            ImportError: Wenn Qdrant nicht installiert
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "Qdrant not installed. Install with:\n"
                "  pip install qdrant-client\n"
                "  or: poetry install -E vector-amd"
            )

        self.dimension = get_embedding_dimension()
        self.clip_metadata: dict[int, dict[str, Any]] = {}
        self.use_persistence = use_persistence

        # Initialize Qdrant client
        if use_persistence:
            logger.info(f"Using persistent Qdrant storage at: {persist_path}")
            self.client = QdrantClient(path=persist_path)
        else:
            logger.info("Using in-memory Qdrant (fastest, recommended for <100k clips)")
            self.client = QdrantClient(":memory:")

        self.collection_exists = False

        logger.info(
            f"QdrantClipMatcher initialized "
            f"(persistence: {use_persistence}, dimension: {self.dimension})"
        )

    def build_index(self, clips: list[dict[str, Any]]) -> None:
        """
        Baut Qdrant Collection aus Clip-Daten.

        Workflow:
        1. Extract embeddings from all clips
        2. Create Qdrant collection with HNSW index
        3. Upload vectors in batch

        Args:
            clips: Liste von Clip-Dicts mit analysis data
                   Format: [{'id': int, 'analysis': {}, 'file_path': str, ...}, ...]

        Raises:
            ValueError: Wenn keine gültigen Embeddings extrahiert werden konnten

        Performance:
            ~200ms für 203 Clips (einmalig)
        """
        logger.info(f"Building Qdrant index for {len(clips)} clips...")

        # Extract embeddings
        points = []
        self.clip_metadata = {}

        for clip in clips:
            try:
                # Extract embedding
                analysis_data = clip.get("analysis", {}).copy()
                analysis_data["id"] = clip.get("id", 0)
                vector = extract_motion_embedding(analysis_data)

                # Validate
                if not validate_embedding(vector):
                    logger.warning(f"Invalid embedding for clip {clip.get('id')}, skipping")
                    continue

                clip_id = clip["id"]

                # Create Qdrant point
                point = PointStruct(
                    id=clip_id,
                    vector=vector.tolist(),
                    payload={
                        "clip_id": clip_id,
                        "file_path": clip.get("file_path", ""),
                        "name": clip.get("name", f"Clip {clip_id}"),
                        "duration": clip.get("duration", 0.0),
                    },
                )
                points.append(point)

                # Store metadata
                self.clip_metadata[clip_id] = {
                    "file_path": clip.get("file_path", ""),
                    "name": clip.get("name", f"Clip {clip_id}"),
                    "duration": clip.get("duration", 0.0),
                }

            except Exception as e:
                logger.warning(f"Failed to extract embedding for clip {clip.get('id')}: {e}")
                continue

        if not points:
            raise ValueError("No valid embeddings extracted from clips")

        logger.info(f"Extracted {len(points)} valid embeddings")

        # Recreate collection (delete if exists)
        try:
            self.client.delete_collection(collection_name=self.COLLECTION_NAME)
            logger.debug("Deleted existing collection")
        except Exception:
            pass  # Collection didn't exist

        # Create collection with HNSW index
        self.client.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=self.dimension,
                distance=Distance.COSINE,  # Cosine similarity (best for embeddings)
            ),
        )

        # Upload vectors in batch
        self.client.upsert(collection_name=self.COLLECTION_NAME, points=points)

        self.collection_exists = True

        logger.info(
            f"Qdrant index built: {len(points)} vectors, "
            f"dimension={self.dimension}, persistence={self.use_persistence}"
        )

    def find_best_clip(
        self,
        target_motion_score: float,
        target_energy: float,
        target_motion_type: str = "MEDIUM",
        target_moods: list[str] | None = None,
        k: int = 5,
        exclude_ids: set | list | None = None,  # PERF-FIX: Accept set or list
        previous_clip_id: int | None = None,  # CONTINUITY: For visual flow
        continuity_weight: float = 0.4,  # How much to prefer similar clips (0=none, 1=max)
    ) -> tuple[int, str, float]:
        """
        Findet besten Clip via Qdrant Similarity Search.

        Args:
            target_motion_score: Ziel-Motion-Intensität (0-1)
            target_energy: Ziel-Energie-Level (0-1)
            target_motion_type: Motion-Typ (STATIC, SLOW, MEDIUM, FAST, EXTREME)
            target_moods: Liste von Ziel-Moods
            k: Anzahl Top-Ergebnisse (für Filtering)
            exclude_ids: Clip-IDs die ausgeschlossen werden sollen
            previous_clip_id: ID des vorherigen Clips für visuelle Kontinuität
            continuity_weight: Gewichtung der visuellen Ähnlichkeit (0-1)

        Returns:
            (clip_id, file_path, distance)

        Raises:
            ValueError: Wenn Index nicht gebaut wurde

        Performance:
            ~10-30ms pro Query (O(log n) Komplexität)

        Example:
            >>> clip_id, path, dist = matcher.find_best_clip(
            ...     target_motion_score=0.8,
            ...     target_energy=0.7,
            ...     target_motion_type='FAST',
            ...     target_moods=['ENERGETIC', 'CHEERFUL'],
            ...     exclude_ids=[1, 2, 3],
            ...     previous_clip_id=42  # For visual continuity
            ... )
        """
        if not self.collection_exists:
            raise ValueError("Index not built. Call build_index() first.")

        # Profiling
        func_start = time.time()
        query_start = time.time()

        # Create query vector
        query = create_query_vector(
            target_motion_score=target_motion_score,
            target_energy=target_energy,
            target_motion_type=target_motion_type,
            target_moods=target_moods or [],
        )

        # Validate
        if not validate_embedding(query):
            logger.warning("Invalid query vector, using defaults")
            query = create_query_vector()

        query_time = time.time() - query_start

        # Search
        search_start = time.time()

        # PERFORMANCE FIX: Use post-filtering instead of query-filter
        # Query-filter with many must_not conditions is O(n) per condition
        # Post-filtering is O(1) set lookup - MUCH faster!
        # PERF-FIX: Accept set directly (avoid O(n) conversion each call!)
        if exclude_ids is None:
            exclude_set = set()
        elif isinstance(exclude_ids, set):
            exclude_set = exclude_ids  # Already a set - use directly (O(1)!)
        else:
            exclude_set = set(exclude_ids)  # Convert list to set (O(n) but only once)

        # Request more results to account for exclusions, then filter in Python
        request_limit = min(k + len(exclude_set) + 10, len(self.clip_metadata))

        # Qdrant query WITHOUT filter (much faster!)
        response = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query.tolist(),
            limit=request_limit,
            with_payload=True,
        )
        all_results = response.points

        search_time = time.time() - search_start

        # Post-filter excluded IDs (O(1) per check with set)
        filter_start = time.time()
        results = [r for r in all_results if r.id not in exclude_set][:k]

        if not results:
            # FIX #11: All results were excluded - search for ANY non-excluded clip
            # Old behavior: Raise ValueError (caused pacing to fail)
            # New behavior: Search ALL clips to find one not excluded
            all_clip_ids = set(self.clip_metadata.keys())
            available_clips = all_clip_ids - exclude_set

            if available_clips:
                # Found at least one non-excluded clip - pick a random one for variety
                random_clip_id = random.choice(list(available_clips))
                metadata = self.clip_metadata[random_clip_id]
                file_path = metadata.get("file_path", "")
                logger.debug(
                    f"All search results excluded, picked random from "
                    f"{len(available_clips)} available clips"
                )
                # Return with neutral distance (0.5)
                filter_time = time.time() - filter_start
                total_time = time.time() - func_start
                logger.debug(
                    f"🔬 Qdrant find_best_clip (fallback): " f"Total={total_time*1000:.3f}ms"
                )
                return random_clip_id, file_path, 0.5

            # Truly no clips available (all excluded)
            logger.warning(
                f"All {len(self.clip_metadata)} clips were excluded! "
                f"Consider resetting exclusion list earlier."
            )
            # Return first result from original search as last resort
            if all_results:
                best = all_results[0]
                clip_id = best.id
                metadata = self.clip_metadata.get(clip_id, {})
                file_path = metadata.get("file_path", best.payload.get("file_path", ""))
                return clip_id, file_path, 1.0 - best.score

            raise ValueError("No clips available - collection is empty")

        # VARIETY FIX: Pick randomly from top results
        # VISUAL CONTINUITY: Prefer clips similar to previous clip ("roter Faden")
        top_candidates = results[: min(10, len(results))]  # Get more candidates for continuity

        if previous_clip_id is not None and continuity_weight > 0 and len(top_candidates) > 1:
            # Get previous clip's embedding via Qdrant retrieve
            try:
                prev_points = self.client.retrieve(
                    collection_name=self.COLLECTION_NAME, ids=[previous_clip_id], with_vectors=True
                )
                if prev_points and prev_points[0].vector:
                    prev_embedding = np.array(prev_points[0].vector)

                    # Score candidates by similarity to previous clip
                    scored_candidates = []
                    for candidate in top_candidates:
                        # Get this candidate's embedding
                        cand_points = self.client.retrieve(
                            collection_name=self.COLLECTION_NAME,
                            ids=[candidate.id],
                            with_vectors=True,
                        )
                        if cand_points and cand_points[0].vector:
                            cand_embedding = np.array(cand_points[0].vector)
                            similarity_dist = float(np.linalg.norm(prev_embedding - cand_embedding))
                            # Combined score: balance between target match and continuity
                            target_dist = 1.0 - candidate.score  # Convert score to distance
                            combined = (
                                1 - continuity_weight
                            ) * target_dist + continuity_weight * similarity_dist
                            scored_candidates.append((candidate, combined))

                    if scored_candidates:
                        # Sort by combined score and pick from top 3
                        scored_candidates.sort(key=lambda x: x[1])
                        top_3 = scored_candidates[:3]
                        best, _ = random.choice(top_3)
                    else:
                        best = random.choice(top_candidates)
                else:
                    best = random.choice(top_candidates)
            except Exception as e:
                logger.debug(f"Continuity lookup failed: {e}, using random selection")
                best = random.choice(top_candidates)
        else:
            best = random.choice(top_candidates[:5])  # Original behavior

        clip_id = best.id
        metadata = self.clip_metadata.get(clip_id, {})
        file_path = metadata.get("file_path", best.payload.get("file_path", ""))

        # Qdrant returns similarity score (0-1), convert to distance for consistency
        # distance = 1 - similarity (lower is better)
        distance = 1.0 - best.score

        # Profiling
        filter_time = time.time() - filter_start
        total_time = time.time() - func_start

        logger.info(
            f"🔬 Qdrant find_best_clip: "
            f"Total={total_time*1000:.3f}ms "
            f"(Query={query_time*1000:.3f}ms, "
            f"Search={search_time*1000:.3f}ms, "
            f"Filter={filter_time*1000:.3f}ms)"
        )

        return clip_id, file_path, distance

    def find_multiple_clips(
        self,
        target_motion_score: float,
        target_energy: float,
        target_motion_type: str = "MEDIUM",
        target_moods: list[str] | None = None,
        k: int = 5,
        exclude_ids: list[int] | None = None,
    ) -> list[tuple[int, str, float]]:
        """
        Findet Top-K beste Clips via Qdrant Search.

        Args:
            target_motion_score: Ziel-Motion (0-1)
            target_energy: Ziel-Energie (0-1)
            target_motion_type: Motion-Typ
            target_moods: Liste von Moods
            k: Anzahl Ergebnisse
            exclude_ids: Auszuschließende Clip-IDs

        Returns:
            Liste von (clip_id, file_path, distance) Tupeln
        """
        if not self.collection_exists:
            raise ValueError("Index not built. Call build_index() first.")

        # Create query
        query = create_query_vector(
            target_motion_score=target_motion_score,
            target_energy=target_energy,
            target_motion_type=target_motion_type,
            target_moods=target_moods or [],
        )

        # PERFORMANCE FIX: Use post-filtering instead of query-filter
        exclude_set = set(exclude_ids) if exclude_ids else set()
        request_limit = min(k + len(exclude_set) + 10, len(self.clip_metadata))

        # Qdrant query WITHOUT filter (much faster!)
        response = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query.tolist(),
            limit=request_limit,
            with_payload=True,
        )
        all_results = response.points

        # Post-filter excluded IDs
        results = [r for r in all_results if r.id not in exclude_set][:k]

        # Build results list
        output = []
        for hit in results:
            clip_id = hit.id
            metadata = self.clip_metadata.get(clip_id, {})
            file_path = metadata.get("file_path", hit.payload.get("file_path", ""))
            distance = 1.0 - hit.score  # Convert similarity to distance
            output.append((clip_id, file_path, distance))

        return output

    def get_index_stats(self) -> dict[str, Any]:
        """
        Returns statistics about the Qdrant index.

        Returns:
            Dict with index statistics
        """
        if not self.collection_exists:
            return {"status": "not_built"}

        try:
            info = self.client.get_collection(collection_name=self.COLLECTION_NAME)
            return {
                "status": "built",
                "total_vectors": info.points_count,
                "dimension": self.dimension,
                "use_persistence": self.use_persistence,
                "clip_count": len(self.clip_metadata),
                "backend": "Qdrant",
                "index_type": "HNSW",
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"status": "error", "error": str(e)}

    def clear_index(self) -> None:
        """Clears the Qdrant collection and metadata."""
        try:
            self.client.delete_collection(collection_name=self.COLLECTION_NAME)
            logger.info("Qdrant collection deleted")
        except Exception as e:
            logger.debug(f"Failed to delete collection: {e}")

        self.collection_exists = False
        self.clip_metadata = {}
        logger.info("Qdrant index cleared")

    def is_ready(self) -> bool:
        """Returns True if index is built and ready for queries."""
        if not self.collection_exists:
            return False

        try:
            info = self.client.get_collection(collection_name=self.COLLECTION_NAME)
            return info.points_count > 0
        except Exception:
            return False


# Utility functions
def is_qdrant_available() -> bool:
    """
    Checks if Qdrant is available.

    Returns:
        True if qdrant-client is installed, False otherwise
    """
    return QDRANT_AVAILABLE


def get_qdrant_version() -> str | None:
    """
    Returns Qdrant client version string.

    Returns:
        Version string or None if Qdrant not available
    """
    if not QDRANT_AVAILABLE:
        return None

    try:
        import qdrant_client

        return qdrant_client.__version__
    except AttributeError:
        return "unknown"
