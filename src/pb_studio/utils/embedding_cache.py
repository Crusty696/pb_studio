"""
Embedding Cache für PB_studio

Persistent Cache für Motion-Embeddings (27D numpy Arrays).
Beschleunigt FAISS Index-Build durch Vermeidung von Neuberechnungen.

Performance:
- Ohne Cache: 553 Clips × 1-2ms = 553-1100ms
- Mit Cache: 553 Clips × 0.01ms = ~5ms (100x Speedup)

Author: PB_studio Development Team
"""

import logging
import threading
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Embedding-Dimension (muss mit motion_embedding.py übereinstimmen)
EMBEDDING_DIM = 27


class EmbeddingCache:
    """
    Persistent Cache für Motion-Embedding Vektoren.

    Speichert 27D float32 numpy Arrays als .npy Dateien.
    Thread-safe für paralleles Laden/Speichern.

    Args:
        cache_dir: Verzeichnis für Cache-Dateien
        enabled: Cache aktiviert (default: True)

    Example:
        >>> cache = EmbeddingCache()
        >>> cache.set(123, np.zeros(27, dtype=np.float32))
        >>> embedding = cache.get(123)
        >>> print(embedding.shape)
        (27,)
    """

    def __init__(self, cache_dir: str = "video_cache/embeddings", enabled: bool = True):
        """Initialize embedding cache."""
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"EmbeddingCache initialized: {self.cache_dir}")

    def _get_cache_path(self, clip_id: int) -> Path:
        """Get cache file path for clip ID."""
        return self.cache_dir / f"clip_{clip_id}.npy"

    def get(self, clip_id: int) -> np.ndarray | None:
        """
        Load embedding from cache.

        Args:
            clip_id: Clip ID

        Returns:
            27D numpy array or None if not cached
        """
        if not self.enabled:
            return None

        cache_path = self._get_cache_path(clip_id)

        if not cache_path.exists():
            return None

        try:
            embedding = np.load(cache_path)

            # Validiere Dimension
            if embedding.shape != (EMBEDDING_DIM,):
                logger.warning(
                    f"Cache corrupt for clip {clip_id}: "
                    f"expected ({EMBEDDING_DIM},), got {embedding.shape}"
                )
                cache_path.unlink()  # Lösche korrupten Cache
                return None

            return embedding.astype(np.float32)

        except Exception as e:
            logger.warning(f"Cache load failed for clip {clip_id}: {e}")
            return None

    def set(self, clip_id: int, embedding: np.ndarray) -> bool:
        """
        Save embedding to cache.

        Args:
            clip_id: Clip ID
            embedding: 27D numpy array

        Returns:
            True if saved successfully
        """
        if not self.enabled:
            return False

        if embedding.shape != (EMBEDDING_DIM,):
            logger.error(
                f"Invalid embedding for clip {clip_id}: "
                f"expected ({EMBEDDING_DIM},), got {embedding.shape}"
            )
            return False

        cache_path = self._get_cache_path(clip_id)

        try:
            np.save(cache_path, embedding.astype(np.float32))
            return True

        except Exception as e:
            logger.error(f"Cache save failed for clip {clip_id}: {e}")
            return False

    def delete(self, clip_id: int) -> bool:
        """
        Delete cached embedding.

        Args:
            clip_id: Clip ID

        Returns:
            True if deleted (or didn't exist)
        """
        cache_path = self._get_cache_path(clip_id)

        if cache_path.exists():
            try:
                cache_path.unlink()
                return True
            except Exception as e:
                logger.error(f"Cache delete failed for clip {clip_id}: {e}")
                return False

        return True

    def get_batch(self, clip_ids: list[int]) -> dict[int, np.ndarray | None]:
        """
        Load multiple embeddings from cache.

        Args:
            clip_ids: List of clip IDs

        Returns:
            Dict mapping clip_id -> embedding (or None if not cached)
        """
        return {clip_id: self.get(clip_id) for clip_id in clip_ids}

    def set_batch(self, embeddings: dict[int, np.ndarray]) -> dict[int, bool]:
        """
        Save multiple embeddings to cache.

        Args:
            embeddings: Dict mapping clip_id -> embedding

        Returns:
            Dict mapping clip_id -> success
        """
        return {clip_id: self.set(clip_id, emb) for clip_id, emb in embeddings.items()}

    def precompute_batch(
        self,
        clip_ids: list[int],
        compute_fn: Callable[[int], np.ndarray],
        max_workers: int = 4,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> int:
        """
        Pre-compute embeddings für mehrere Clips parallel.

        Args:
            clip_ids: Liste von Clip IDs ohne gecachte Embeddings
            compute_fn: Funktion (clip_id) -> 27D numpy array
            max_workers: Anzahl paralleler Worker
            progress_callback: Optional (current, total) -> None

        Returns:
            Anzahl neu berechneter Embeddings
        """
        # Filtere bereits gecachte
        uncached = [cid for cid in clip_ids if self.get(cid) is None]

        if not uncached:
            logger.info("All embeddings already cached")
            return 0

        logger.info(f"Pre-computing {len(uncached)} embeddings with {max_workers} workers")

        computed = 0

        # Parallele Berechnung
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(compute_fn, clip_id): clip_id for clip_id in uncached}

            for i, future in enumerate(as_completed(futures)):
                clip_id = futures[future]

                try:
                    embedding = future.result()
                    if self.set(clip_id, embedding):
                        computed += 1

                except Exception as e:
                    logger.warning(f"Embedding compute failed for clip {clip_id}: {e}")

                if progress_callback:
                    progress_callback(i + 1, len(uncached))

        logger.info(f"Pre-computed {computed}/{len(uncached)} embeddings")
        return computed

    def get_stats(self) -> dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dict with 'cached_count', 'total_size_mb'
        """
        if not self.cache_dir.exists():
            return {"cached_count": 0, "total_size_mb": 0}

        files = list(self.cache_dir.glob("clip_*.npy"))
        total_size = sum(f.stat().st_size for f in files)

        return {"cached_count": len(files), "total_size_mb": round(total_size / (1024 * 1024), 2)}

    def clear(self) -> int:
        """
        Clear all cached embeddings.

        Returns:
            Number of deleted files
        """
        if not self.cache_dir.exists():
            return 0

        files = list(self.cache_dir.glob("clip_*.npy"))
        deleted = 0

        for f in files:
            try:
                f.unlink()
                deleted += 1
            except Exception as e:
                logger.error(f"Failed to delete {f}: {e}")

        logger.info(f"Cleared {deleted} cached embeddings")
        return deleted


# Singleton-Instanz für globalen Zugriff
_embedding_cache: EmbeddingCache | None = None
# FIX: Thread-safe singleton using Lock
_embedding_cache_lock = threading.Lock()


def get_embedding_cache() -> EmbeddingCache:
    """
    Get singleton EmbeddingCache instance.

    Returns:
        Global EmbeddingCache instance
    """
    global _embedding_cache

    # FIX: Double-checked locking pattern for thread-safe singleton
    if _embedding_cache is None:
        with _embedding_cache_lock:
            if _embedding_cache is None:
                _embedding_cache = EmbeddingCache()

    return _embedding_cache
