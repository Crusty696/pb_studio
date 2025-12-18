"""
Cache Manager für PB_studio

Zentralisierte JSON-basierte Cache-Verwaltung mit MD5-Hashing.
Eliminiert Code-Duplikation in motion_analyzer.py und structure_analyzer.py.

Author: PB_studio Development Team
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Generic JSON cache manager with MD5-based file naming.

    Args:
        cache_dir: Directory for cache files
        prefix: Prefix for cache file names (default: "cache")
        ttl_seconds: Time-to-live in seconds (None = infinite)

    Example:
        >>> cache = CacheManager(Path("cache"), prefix="motion")
        >>> cache.save("video.mp4", {"motion_data": [1, 2, 3]})
        >>> data = cache.load("video.mp4")
        >>> print(data)
        {'motion_data': [1, 2, 3]}
    """

    def __init__(
        self,
        cache_dir: Path,
        prefix: str = "cache",
        ttl_seconds: int | None = None,
        max_size_mb: int | None = None,
    ):
        """Initialize cache manager."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.ttl_seconds = ttl_seconds
        self.max_size_mb = max_size_mb
        logger.debug(
            f"CacheManager initialized: {self.cache_dir}, prefix={prefix}, max_size={max_size_mb}MB"
        )

    def _get_cache_path(self, key: str) -> Path:
        """
        Get cache file path for given key.

        Args:
            key: Cache key (e.g., file path)

        Returns:
            Path to cache file
        """
        hash_str = hashlib.md5(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{self.prefix}_{hash_str}.json"

    def load(self, key: str) -> dict[str, Any] | None:
        """
        Load data from cache.

        Args:
            key: Cache key

        Returns:
            Cached data or None if not found/expired

        Example:
            >>> cache = CacheManager(Path("cache"))
            >>> data = cache.load("video.mp4")
            >>> if data:
            ...     print(f"Cache hit: {data}")
        """
        cache_file = self._get_cache_path(key)

        if not cache_file.exists():
            logger.debug(f"Cache miss: {key}")
            return None

        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))

            # Check TTL if configured
            if self.ttl_seconds is not None and "_timestamp" in data:
                timestamp = datetime.fromisoformat(data["_timestamp"])
                age = (datetime.now() - timestamp).total_seconds()
                if age > self.ttl_seconds:
                    logger.debug(f"Cache expired: {key} (age={age:.1f}s)")
                    cache_file.unlink()
                    return None

            logger.debug(f"Cache hit: {key}")

            # LRU Update: Touch file to update mtime
            try:
                cache_file.touch()
            except Exception:
                pass  # Ignore permission errors

            return data

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Cache file corrupted: {cache_file}, error: {e}")
            cache_file.unlink()
            return None

    def save(self, key: str, data: dict[str, Any]) -> None:
        """
        Save data to cache.

        Args:
            key: Cache key
            data: Data to cache (must be JSON-serializable)

        Raises:
            TypeError: If data is not JSON-serializable

        Example:
            >>> cache = CacheManager(Path("cache"))
            >>> cache.save("video.mp4", {"frames": 1000})
        """
        cache_file = self._get_cache_path(key)

        # Add timestamp if TTL is configured
        if self.ttl_seconds is not None:
            data = data.copy()
            data["_timestamp"] = datetime.now().isoformat()

        try:
            cache_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.debug(f"Cache saved: {key}")

            # Enforce size limit
            if self.max_size_mb is not None:
                self._enforce_size_limit()

        except TypeError as e:
            logger.error(f"Cannot serialize data for {key}: {e}")
            raise

    def _enforce_size_limit(self) -> None:
        """Enforce maximum cache size by deleting LRU files."""
        if self.max_size_mb is None:
            return

        try:
            pattern = f"{self.prefix}_*.json"
            files = list(self.cache_dir.glob(pattern))

            total_size = sum(f.stat().st_size for f in files)
            max_bytes = self.max_size_mb * 1024 * 1024

            if total_size <= max_bytes:
                return

            # Sort by mtime (oldest first - LRU)
            files.sort(key=lambda f: f.stat().st_mtime)

            deleted_count = 0
            freed_bytes = 0

            for f in files:
                if total_size <= max_bytes:
                    break

                try:
                    size = f.stat().st_size
                    f.unlink()
                    total_size -= size
                    freed_bytes += size
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {f}: {e}")

            if deleted_count > 0:
                logger.info(
                    f"Cache cleanup: Deleted {deleted_count} files, "
                    f"freed {freed_bytes/1024/1024:.2f} MB"
                )

        except Exception as e:
            logger.error(f"Error enforcing cache size limit: {e}")

    def exists(self, key: str) -> bool:
        """
        Check if cache entry exists and is valid.

        Args:
            key: Cache key

        Returns:
            True if cache exists and is valid
        """
        return self.load(key) is not None

    def invalidate(self, key: str) -> bool:
        """
        Remove cache entry.

        Args:
            key: Cache key

        Returns:
            True if cache was removed, False if not found
        """
        cache_file = self._get_cache_path(key)
        if cache_file.exists():
            cache_file.unlink()
            logger.debug(f"Cache invalidated: {key}")
            return True
        return False

    def clear_all(self) -> int:
        """
        Clear all cache files with this prefix.

        Returns:
            Number of files deleted

        Example:
            >>> cache = CacheManager(Path("cache"), prefix="motion")
            >>> count = cache.clear_all()
            >>> print(f"Deleted {count} cache files")
        """
        pattern = f"{self.prefix}_*.json"
        cache_files = list(self.cache_dir.glob(pattern))

        for cache_file in cache_files:
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete {cache_file}: {e}")

        logger.info(f"Cleared {len(cache_files)} cache files")
        return len(cache_files)

    def get_cache_size(self) -> int:
        """
        Get total size of cache files in bytes.

        Returns:
            Total size in bytes
        """
        pattern = f"{self.prefix}_*.json"
        cache_files = list(self.cache_dir.glob(pattern))
        total_size = sum(f.stat().st_size for f in cache_files)
        return total_size

    def get_cache_info(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache info (file_count, total_size_bytes, total_size_mb)
        """
        pattern = f"{self.prefix}_*.json"
        cache_files = list(self.cache_dir.glob(pattern))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "file_count": len(cache_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
            "prefix": self.prefix,
            "max_size_mb": self.max_size_mb,
        }
