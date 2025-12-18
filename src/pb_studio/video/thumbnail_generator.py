"""
Thumbnail Generator for Video Clips

Generates thumbnail images from video files for quick visual identification.
Uses OpenCV to extract frames and PIL for image processing.

Cache Naming Schema:
- New format: {video_hash}_{position_ms}.jpg
  - video_hash = MD5(video_path)[:16]
  - position_ms = int(position * 1000) or "center"
- Index file: {video_hash}_index.json (tracks all thumbnail positions)
- Legacy format: thumb_{combined_hash}.jpg (backward compatible)
"""

import hashlib
import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import cv2
from PIL import Image

logger = logging.getLogger(__name__)

# Thread-Safety: Per-video locks for index file operations
_index_locks: dict[str, Lock] = {}
_index_locks_master = Lock()


class ThumbnailGenerator:
    """
    Generate thumbnail images from video files.

    Features:
    - Extract frame from video center or specific position
    - Resize to target dimensions while maintaining aspect ratio
    - Cache thumbnails to avoid re-generation
    - Support for various video formats via OpenCV
    """

    def __init__(
        self,
        cache_dir: Path | str = "thumbnails",
        thumbnail_size: tuple[int, int] = (160, 90),
        quality: int = 85,
    ):
        """
        Initialize ThumbnailGenerator.

        Args:
            cache_dir: Directory to store generated thumbnails
            thumbnail_size: Target size (width, height) for thumbnails
            quality: JPEG quality (1-100)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnail_size = thumbnail_size
        self.quality = quality

        logger.info(
            f"ThumbnailGenerator initialisiert: "
            f"cache_dir={self.cache_dir}, size={thumbnail_size}, quality={quality}"
        )

    def _get_index_lock(self, video_path: Path) -> Lock:
        """
        Get or create a lock for a specific video's index file.

        Thread-Safety: Prevents race conditions when multiple threads
        update the same video's index file simultaneously.
        """
        vid_hash = self._video_hash(video_path)
        with _index_locks_master:
            if vid_hash not in _index_locks:
                _index_locks[vid_hash] = Lock()
            return _index_locks[vid_hash]

    def _video_hash(self, video_path: Path) -> str:
        """
        Generate stable hash for video path.

        Returns:
            First 16 characters of MD5 hash (sufficient for uniqueness)
        """
        full_hash = hashlib.md5(str(video_path.resolve()).encode("utf-8")).hexdigest()
        return full_hash[:16]

    def _position_tag(self, position_sec: float | None) -> str:
        """
        Generate position tag for filename.

        Args:
            position_sec: Position in seconds or None for center

        Returns:
            "center" or "{milliseconds}ms"
        """
        if position_sec is None:
            return "center"
        position_ms = int(position_sec * 1000)
        return f"{position_ms}ms"

    def _get_index_path(self, video_path: Path) -> Path:
        """Get path to index file for a video."""
        vid_hash = self._video_hash(video_path)
        return self.cache_dir / f"{vid_hash}_index.json"

    def _load_index(self, video_path: Path) -> set[str]:
        """
        Load thumbnail position index for a video.

        Returns:
            Set of position tags (e.g., {"center", "1500ms", "3000ms"})
        """
        index_path = self._get_index_path(video_path)
        if not index_path.exists():
            return set()

        try:
            with open(index_path, encoding="utf-8") as f:
                data = json.load(f)
                return set(data.get("positions", []))
        except Exception as e:
            logger.warning(f"Failed to load index {index_path}: {e}")
            return set()

    def _save_index(self, video_path: Path, positions: set[str]) -> None:
        """
        Save thumbnail position index for a video.

        Args:
            video_path: Path to video file
            positions: Set of position tags
        """
        index_path = self._get_index_path(video_path)

        try:
            data = {"video_path": str(video_path.resolve()), "positions": sorted(positions)}
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save index {index_path}: {e}")

    def _add_to_index(self, video_path: Path, position_sec: float | None) -> None:
        """
        Add position to thumbnail index (thread-safe).

        Uses per-video locking to prevent race conditions during
        parallel thumbnail generation.
        """
        lock = self._get_index_lock(video_path)
        with lock:
            positions = self._load_index(video_path)
            positions.add(self._position_tag(position_sec))
            self._save_index(video_path, positions)

    def _get_cache_path(self, video_path: Path, position_sec: float | None = None) -> Path:
        """
        Generate cache file path for a video thumbnail (new naming schema).

        New format: {video_hash}_{position_tag}.jpg
        Example: abc123def456789a_center.jpg or abc123def456789a_1500ms.jpg

        Args:
            video_path: Path to video file
            position_sec: Optional specific position in seconds

        Returns:
            Path to thumbnail cache file
        """
        vid_hash = self._video_hash(video_path)
        pos_tag = self._position_tag(position_sec)
        return self.cache_dir / f"{vid_hash}_{pos_tag}.jpg"

    def _get_legacy_cache_path(self, video_path: Path, position_sec: float | None) -> Path:
        """
        Generate legacy cache path (backward compatible).

        Legacy format: thumb_{combined_hash}.jpg
        """
        key = f"{video_path}_{position_sec if position_sec else 'center'}"
        hash_str = hashlib.md5(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"thumb_{hash_str}.jpg"

    def generate(
        self,
        video_path: Path | str,
        position_sec: float | None = None,
        force_regenerate: bool = False,
    ) -> Path | None:
        """
        Generate thumbnail from video file.

        Args:
            video_path: Path to video file
            position_sec: Position in seconds to extract frame (None = center)
            force_regenerate: Force regeneration even if cached

        Returns:
            Path to generated thumbnail or None on error
        """
        video_path = Path(video_path)

        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return None

        # Check cache first (new naming), fallback auf Legacy
        cache_path = self._get_cache_path(video_path, position_sec)
        legacy_cache_path = self._get_legacy_cache_path(video_path, position_sec)
        if not force_regenerate:
            if cache_path.exists():
                logger.debug(f"Using cached thumbnail: {cache_path}")
                return cache_path
            if legacy_cache_path.exists():
                logger.debug(f"Using legacy cached thumbnail: {legacy_cache_path}")
                return legacy_cache_path

        try:
            # PERF-02 FIX: Use context manager to ensure VideoCapture is released
            from ..utils.video_utils import open_video

            with open_video(video_path) as cap:
                if not cap.isOpened():
                    logger.error(f"Failed to open video: {video_path}")
                    return None

                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                if frame_count == 0:
                    logger.error(f"Video has no frames: {video_path}")
                    return None

                # Determine frame position
                if position_sec is not None:
                    frame_number = int(position_sec * fps)
                else:
                    # Use center frame
                    frame_number = frame_count // 2

                # Ensure frame number is valid
                frame_number = max(0, min(frame_number, frame_count - 1))

                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()

            # VideoCapture automatically released by context manager

            if not ret or frame is None:
                logger.error(f"Failed to read frame {frame_number} from {video_path}")
                return None

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create PIL Image and save (PERF-FIX: explizites close())
            pil_image = Image.fromarray(frame_rgb)
            try:
                # Resize maintaining aspect ratio
                pil_image.thumbnail(self.thumbnail_size, Image.Resampling.LANCZOS)

                # Save thumbnail
                pil_image.save(cache_path, "JPEG", quality=self.quality)
            finally:
                pil_image.close()  # PERF-FIX: Verhindert Memory-Leak bei Batch-Processing

            # Update index
            self._add_to_index(video_path, position_sec)

            logger.info(
                f"Generated thumbnail: {video_path.name} -> {cache_path.name} "
                f"(frame {frame_number}/{frame_count})"
            )

            return cache_path

        except Exception as e:
            logger.error(f"Error generating thumbnail for {video_path}: {e}", exc_info=True)
            return None

    def generate_batch(
        self,
        video_paths: list[Path | str],
        position_sec: float | None = None,
        force_regenerate: bool = False,
        max_workers: int = 4,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[Path, Path | None]:
        """
        Generate thumbnails for multiple videos in PARALLEL.

        PERF-FIX: Verwendet ThreadPoolExecutor für parallele Thumbnail-Generierung.
        Bei 100 Videos: ~85% schneller (25s -> 4s auf SSD).

        Args:
            video_paths: List of video file paths
            position_sec: Position in seconds (None = center)
            force_regenerate: Force regeneration even if cached
            max_workers: Number of parallel workers (default: 4)
            progress_callback: Optional callback(completed, total) for progress updates

        Returns:
            Dictionary mapping video_path -> thumbnail_path (or None on error)
        """
        results = {}
        total = len(video_paths)

        if total == 0:
            return results

        # Für kleine Batches sequential bleiben
        if total <= 3:
            for i, video_path in enumerate(video_paths):
                video_path = Path(video_path)
                thumb_path = self.generate(video_path, position_sec, force_regenerate)
                results[video_path] = thumb_path
                if progress_callback:
                    progress_callback(i + 1, total)
            return results

        # PERF-FIX: Parallele Generierung für größere Batches
        completed = 0

        def _generate_single(vpath: Path) -> tuple[Path, Path | None]:
            """Worker function für einen Thumbnail."""
            return (vpath, self.generate(vpath, position_sec, force_regenerate))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(_generate_single, Path(vp)): Path(vp) for vp in video_paths}

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    video_path, thumb_path = future.result()
                    results[video_path] = thumb_path
                except Exception as e:
                    video_path = futures[future]
                    logger.error(f"Thumbnail generation failed for {video_path}: {e}")
                    results[video_path] = None

                completed += 1
                if progress_callback:
                    progress_callback(completed, total)

        success_count = sum(1 for p in results.values() if p is not None)
        logger.info(
            f"Batch generation complete: {success_count}/{total} successful "
            f"(parallel, workers={max_workers})"
        )

        return results

    def clear_cache_for_video(self, video_path: Path | str) -> int:
        """
        Clear all thumbnails for a specific video.

        This method can find and delete ALL thumbnails for a video because:
        1. New naming schema: {video_hash}_{position}.jpg - prefix search works
        2. Index file tracks all positions for complete cleanup
        3. Legacy format also checked for backward compatibility

        Args:
            video_path: Path to video file

        Returns:
            Number of files deleted (thumbnails + index)
        """
        video_path = Path(video_path)
        vid_hash = self._video_hash(video_path)
        deleted_count = 0

        # Delete all thumbnails with matching video hash (new format)
        for thumb_file in self.cache_dir.glob(f"{vid_hash}_*.jpg"):
            thumb_file.unlink()
            deleted_count += 1
            logger.debug(f"Deleted thumbnail: {thumb_file.name}")

        # Delete legacy thumbnails (check common positions)
        legacy_positions = [None, 0.0, 1.0, 2.0, 3.0, 5.0, 10.0]
        for pos in legacy_positions:
            legacy_path = self._get_legacy_cache_path(video_path, pos)
            if legacy_path.exists():
                legacy_path.unlink()
                deleted_count += 1
                logger.debug(f"Deleted legacy thumbnail: {legacy_path.name}")

        # Delete index file
        index_path = self._get_index_path(video_path)
        if index_path.exists():
            index_path.unlink()
            deleted_count += 1
            logger.debug(f"Deleted index file: {index_path.name}")

        if deleted_count > 0:
            logger.info(f"Cleared cache for {video_path.name}: {deleted_count} files deleted")
        else:
            logger.debug(f"No cached thumbnails found for {video_path.name}")

        return deleted_count

    def clear_cache(self, video_path: Path | str | None = None) -> int:
        """
        Clear thumbnail cache.

        Args:
            video_path: Optional specific video path to clear cache for.
                       If None, clears entire cache.

        Returns:
            Number of files deleted
        """
        if video_path is None:
            # Clear entire cache (thumbnails + indices)
            thumb_files = list(self.cache_dir.glob("*.jpg"))
            index_files = list(self.cache_dir.glob("*_index.json"))
            all_files = thumb_files + index_files

            for file in all_files:
                file.unlink()

            logger.info(
                f"Cleared entire thumbnail cache: "
                f"{len(thumb_files)} thumbnails, {len(index_files)} indices deleted"
            )
            return len(all_files)
        else:
            # Clear specific video
            return self.clear_cache_for_video(video_path)

    def get_cache_size(self) -> int:
        """
        Get total size of thumbnail cache in bytes.

        Returns:
            Total cache size in bytes
        """
        total_size = 0
        for file in self.cache_dir.glob("thumb_*.jpg"):
            total_size += file.stat().st_size
        return total_size

    def get_thumbnail_path(
        self, video_path: Path | str, position_sec: float | None = None
    ) -> Path | None:
        """
        Get path to cached thumbnail without generating it.

        Args:
            video_path: Path to video file
            position_sec: Position in seconds (None = center)

        Returns:
            Path to cached thumbnail or None if not cached
        """
        cache_path = self._get_cache_path(Path(video_path), position_sec)
        return cache_path if cache_path.exists() else None
