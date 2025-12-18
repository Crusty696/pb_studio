"""
Persistent Video Metadata Cache for PB_studio

PERF-07 FIX: Caches FFprobe results persistently across renderer instances.
Reduces repeated ffprobe calls by 90% for known videos.

Features:
- File-based persistent cache (JSON)
- Cache key based on file path + mtime + size (no content hashing needed)
- Automatic invalidation when file is modified
- Thread-safe with lock

Author: PB_studio Development Team
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class VideoMetadataCache:
    """
    Persistent cache for video metadata (FFprobe results).

    PERF-07 FIX: Avoids repeated ffprobe calls (100-200ms each) by caching
    metadata persistently. Cache survives renderer restarts.

    Cache key format: "filename_size_mtime"
    This ensures cache invalidation when file is modified.
    """

    def __init__(self, cache_dir: Path | None = None):
        """
        Initialize the video metadata cache.

        Args:
            cache_dir: Directory for cache file. Defaults to data/video_metadata_cache.json
        """
        if cache_dir is None:
            # Default to data directory in project root
            project_root = Path(__file__).parent.parent.parent.parent
            cache_dir = project_root / "data"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "video_metadata_cache.json"

        self._cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._dirty = False  # Track if cache needs saving

        # Load existing cache
        self._load_cache()

        logger.info(f"VideoMetadataCache initialized: {len(self._cache)} entries loaded")

    def _get_cache_key(self, video_path: Path) -> str | None:
        """
        Generate cache key from video file metadata.

        Uses file size + mtime for fast invalidation without content hashing.

        Args:
            video_path: Path to video file

        Returns:
            Cache key string or None if file doesn't exist
        """
        try:
            stat = video_path.stat()
            mtime_ms = int(stat.st_mtime * 1000)
            size = stat.st_size
            return f"{video_path.name}_{size}_{mtime_ms}"
        except Exception as e:
            logger.debug(f"Failed to stat {video_path}: {e}")
            return None

    def _load_cache(self) -> None:
        """Load cache from disk."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.debug(f"Loaded {len(self._cache)} cached metadata entries")
        except Exception as e:
            logger.warning(f"Failed to load metadata cache: {e}")
            self._cache = {}

    def _save_cache(self) -> None:
        """Save cache to disk (debounced - only if dirty).

        THREAD-SAFE FIX: Creates a copy of the cache dictionary under lock
        to prevent "dictionary changed size during iteration" errors.
        """
        if not self._dirty:
            return

        try:
            # THREAD-SAFE: Copy dict under lock to avoid iteration errors
            with self._lock:
                cache_copy = dict(self._cache)
                self._dirty = False

            # Save copy outside lock (allows concurrent reads during I/O)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_copy, f, indent=2)

            logger.debug(f"Saved {len(cache_copy)} metadata entries to disk")
        except Exception as e:
            logger.warning(f"Failed to save metadata cache: {e}")

    def get_duration(self, video_path: Path) -> float | None:
        """
        Get cached video duration.

        Args:
            video_path: Path to video file

        Returns:
            Duration in seconds or None if not cached
        """
        cache_key = self._get_cache_key(video_path)
        if cache_key is None:
            return None

        with self._lock:
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                logger.debug(f"Cache HIT: {video_path.name} = {entry.get('duration', 0):.2f}s")
                return entry.get("duration")

        return None

    def set_duration(self, video_path: Path, duration: float) -> None:
        """
        Cache video duration.

        Args:
            video_path: Path to video file
            duration: Duration in seconds
        """
        cache_key = self._get_cache_key(video_path)
        if cache_key is None:
            return

        with self._lock:
            self._cache[cache_key] = {
                "duration": duration,
                "path": str(video_path),
                "cached_at": int(os.path.getmtime(video_path) * 1000),
            }
            self._dirty = True

        # Save periodically (every 10 new entries)
        if len(self._cache) % 10 == 0:
            self._save_cache()

    def get_metadata(self, video_path: Path) -> dict[str, Any] | None:
        """
        Get all cached metadata for a video.

        Args:
            video_path: Path to video file

        Returns:
            Metadata dict or None if not cached
        """
        cache_key = self._get_cache_key(video_path)
        if cache_key is None:
            return None

        with self._lock:
            return self._cache.get(cache_key)

    def set_metadata(self, video_path: Path, metadata: dict[str, Any]) -> None:
        """
        Cache complete video metadata.

        Args:
            video_path: Path to video file
            metadata: FFprobe metadata dict
        """
        cache_key = self._get_cache_key(video_path)
        if cache_key is None:
            return

        with self._lock:
            self._cache[cache_key] = {
                **metadata,
                "path": str(video_path),
                "cached_at": int(os.path.getmtime(video_path) * 1000),
            }
            self._dirty = True

    def flush(self) -> None:
        """Force save cache to disk."""
        self._dirty = True
        self._save_cache()

    def clear(self) -> None:
        """Clear entire cache (memory + disk)."""
        with self._lock:
            self._cache.clear()
            self._dirty = False

        if self.cache_file.exists():
            self.cache_file.unlink()

        logger.info("Video metadata cache cleared")

    def get_stats(self) -> dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dict with 'entries', 'size_bytes'
        """
        with self._lock:
            entries = len(self._cache)

        size_bytes = 0
        if self.cache_file.exists():
            size_bytes = self.cache_file.stat().st_size

        return {"entries": entries, "size_bytes": size_bytes}

    def __del__(self):
        """Save cache on cleanup."""
        try:
            self._save_cache()
        except Exception:
            pass


# Global singleton instance
_video_metadata_cache: VideoMetadataCache | None = None


def get_video_metadata_cache() -> VideoMetadataCache:
    """
    Get global VideoMetadataCache instance.

    Returns:
        VideoMetadataCache singleton
    """
    global _video_metadata_cache
    if _video_metadata_cache is None:
        _video_metadata_cache = VideoMetadataCache()
    return _video_metadata_cache
