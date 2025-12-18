"""
Video Analyzer for PB_studio

Production-ready scene detection using PySceneDetect.

Features:
- Scene detection with ContentDetector (official recommended)
- Stats caching for 10-100x performance improvement
- MD5-based cache invalidation
- Database integration with SQLAlchemy
- Comprehensive error handling
- Memory-efficient processing

Dependencies:
- scenedetect[opencv] >= 0.6.3
- SQLAlchemy >= 2.0

Usage:
    analyzer = VideoAnalyzer(session=db_session, cache_dir="scene_cache")

    # Detect scenes and store in database
    scenes = analyzer.analyze_scenes(
        video_path="video.mp4",
        video_clip_id=1,
        threshold=27.0
    )

    # Get scene timestamps only (no database)
    timestamps = analyzer.get_scene_timestamps("video.mp4")
"""

import hashlib
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, Optional, Tuple, TypedDict

import cv2  # Added for video properties analysis
from scenedetect import AdaptiveDetector, ContentDetector, detect, open_video
from scenedetect.scene_detector import SceneDetector
from scenedetect.video_stream import VideoOpenFailure
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ..database.models import VideoClip
from ..database.models_analysis import ClipSemantics

logger = logging.getLogger(__name__)


class SceneTimestamp(TypedDict):
    """Single scene timestamp with start/end in seconds."""

    scene_index: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    start_frame: int
    end_frame: int


class SceneDetectionResult(TypedDict):
    """Complete scene detection result."""

    video_path: str
    total_scenes: int
    scene_timestamps: list[float]  # Flattened list: [start1, end1, start2, end2, ...]
    scenes: list[SceneTimestamp]
    detector_type: str
    threshold: float


class VideoResolution(Enum):
    """Standard Video Resolutions."""

    # Ultra HD
    UHD_4K = (3840, 2160)  # 4K UHD
    UHD_8K = (7680, 4320)  # 8K UHD

    # HD Resolutions
    FULL_HD = (1920, 1080)  # 1080p Full HD
    HD_READY = (1280, 720)  # 720p HD Ready

    # Standard Resolutions
    DVD = (720, 480)  # DVD Standard
    VCD = (352, 288)  # VCD

    # AI Model Optimized
    CLIP_224 = (224, 224)  # CLIP Standard
    CLIP_336 = (336, 336)  # CLIP Large
    YOLO_640 = (640, 640)  # YOLO Standard


class VideoFrameRate(Enum):
    """Common Video Frame Rates."""

    CINEMA_24 = 24.0  # Cinema Standard
    PAL_25 = 25.0  # PAL Standard
    NTSC_30 = 29.97  # NTSC Standard
    SMOOTH_30 = 30.0  # Smooth 30fps
    GAMING_60 = 60.0  # Gaming/High-Quality
    ULTRA_120 = 120.0  # Ultra-High FPS
    VARIABLE = 0.0  # Variable Frame Rate


@dataclass
class VideoProperties:
    """Video Properties Container."""

    # Basic Properties
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float

    # Codec Information
    codec: str
    bitrate: int | None = None

    # Quality Metrics
    quality_score: float = 0.0  # 0.0-1.0

    # Processing Optimization
    optimal_sample_rate: float = 1.0  # Frames to process (1.0 = all, 0.5 = every 2nd)
    recommended_resize: tuple[int, int] | None = None  # For AI processing

    # Memory Requirements
    memory_per_frame_mb: float = 0.0
    total_memory_estimate_mb: float = 0.0

    # Compatibility Flags
    supports_gpu_decode: bool = True
    supports_fast_seek: bool = True
    is_variable_fps: bool = False

    @property
    def resolution_class(self) -> str:
        """Get resolution classification."""
        if self.width >= 3840:
            return "4K+"
        elif self.width >= 1920:
            return "1080p"
        elif self.width >= 1280:
            return "720p"
        elif self.width >= 720:
            return "480p"
        else:
            return "low_res"

    @property
    def aspect_ratio(self) -> float:
        """Get aspect ratio."""
        return self.width / self.height if self.height > 0 else 0.0

    @property
    def fps_class(self) -> str:
        """Get FPS classification."""
        if self.fps >= 60:
            return "high_fps"
        elif self.fps >= 30:
            return "standard_fps"
        elif self.fps >= 24:
            return "cinema_fps"
        else:
            return "low_fps"


class VideoAnalyzer:
    """
    Production-ready video scene analyzer using PySceneDetect.

    Implements official best practices:
    - ContentDetector with threshold=27.0 (official default)
    - Stats caching for performance (10-100x speedup)
    - MD5-based cache keys with file size validation
    - Float seconds for precise timestamps
    - SQLAlchemy 2.0 database integration
    """

    def __init__(
        self,
        session: Session | None = None,
        cache_dir: str | Path = "scene_cache",
        use_cache: bool = True,
    ) -> None:
        """
        Initialize the VideoAnalyzer.

        Args:
            session: SQLAlchemy session for database operations (optional)
            cache_dir: Directory for storing stats cache files
            use_cache: Enable stats caching for performance
        """
        self.session = session
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache

        # Create cache directory
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        """String representation for debugging."""
        session_status = "configured" if self.session else "not configured"
        return f"<VideoAnalyzer(session={session_status}, cache_dir={self.cache_dir})>"

    def _validate_video_path(self, video_path: str | Path) -> Path | None:
        """
        Validate video file path exists.

        Args:
            video_path: Path to validate

        Returns:
            Path object if valid, None if invalid
        """
        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return None
        return video_path

    def _get_cache_key(self, video_path: Path) -> str:
        """
        Generate MD5 cache key from video path and file metadata.

        Args:
            video_path: Path to video file

        Returns:
            MD5 hash string for cache key
        """
        hasher = hashlib.md5()
        hasher.update(str(video_path.absolute()).encode())
        stat = video_path.stat()
        hasher.update(f"{stat.st_size}_{stat.st_mtime}".encode())
        return hasher.hexdigest()

    def _get_stats_cache_path(self, video_path: Path) -> Path:
        """
        Get stats cache file path for video.

        Args:
            video_path: Path to video file

        Returns:
            Path to stats CSV cache file
        """
        cache_key = self._get_cache_key(video_path)
        return self.cache_dir / f"{cache_key}.stats.csv"

    def _create_detector(
        self,
        detector_type: Literal["content", "adaptive"] = "content",
        threshold: float = 27.0,
        min_scene_len: int = 15,
    ) -> SceneDetector:
        """
        Create scene detector with official recommended parameters.

        Args:
            detector_type: Type of detector ("content" or "adaptive")
            threshold: Detection threshold (content: 27.0 default; adaptive: 3.0 default)
            min_scene_len: Minimum frames between cuts (default: 15)

        Returns:
            Configured SceneDetector instance
        """
        if detector_type == "adaptive":
            # AdaptiveDetector: Best for fast camera motion
            return AdaptiveDetector(
                adaptive_threshold=threshold if threshold != 27.0 else 3.0,
                min_scene_len=min_scene_len,
                min_content_val=15.0,
                window_width=2,
                luma_only=False,
            )
        else:
            # ContentDetector: Official recommended for general use
            return ContentDetector(
                threshold=threshold, min_scene_len=min_scene_len, luma_only=False
            )

    def get_scene_timestamps(
        self,
        video_path: str | Path,
        threshold: float = 27.0,
        detector_type: Literal["content", "adaptive"] = "content",
        min_scene_len: int = 15,
    ) -> list[float] | None:
        """
        Get scene timestamps only (no database storage).

        Args:
            video_path: Path to video file
            threshold: Detection threshold (default: 27.0)
            detector_type: Detector type ("content" or "adaptive")
            min_scene_len: Minimum frames between cuts

        Returns:
            List of scene timestamps [start1, end1, start2, end2, ...] or None on error
        """
        video_path = self._validate_video_path(video_path)
        if video_path is None:
            return None

        try:
            logger.info(f"Detecting scenes: {video_path.name}")

            # Get cache path
            stats_path = None
            if self.use_cache:
                stats_path = self._get_stats_cache_path(video_path)
                if stats_path.exists():
                    logger.debug(f"Using cached stats: {stats_path.name}")

            # Create detector
            detector = self._create_detector(detector_type, threshold, min_scene_len)

            # Detect scenes (official high-level API)
            scenes = detect(
                str(video_path),
                detector,
                stats_file_path=str(stats_path) if stats_path else None,
                show_progress=False,
            )

            if not scenes:
                logger.warning(f"No scenes detected in {video_path.name}")
                return []

            # Convert to flat list of timestamps (seconds)
            timestamps: list[float] = []
            for start, end in scenes:
                timestamps.append(start.get_seconds())
                timestamps.append(end.get_seconds())

            logger.info(
                f"Detected {len(scenes)} scenes in {video_path.name} "
                f"(detector: {detector_type}, threshold: {threshold})"
            )

            return timestamps

        except VideoOpenFailure as e:
            logger.error(f"Failed to open video {video_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error detecting scenes: {e}", exc_info=True)
            return None

    def analyze_scenes(
        self,
        video_path: str | Path,
        video_clip_id: int | None = None,
        threshold: float = 27.0,
        detector_type: Literal["content", "adaptive"] = "content",
        min_scene_len: int = 15,
    ) -> SceneDetectionResult | None:
        """
        Analyze video for scenes and optionally store in database.

        Args:
            video_path: Path to video file
            video_clip_id: VideoClip ID for database storage (optional)
            threshold: Detection threshold (default: 27.0)
            detector_type: Detector type ("content" or "adaptive")
            min_scene_len: Minimum frames between cuts

        Returns:
            SceneDetectionResult dict or None on error
        """
        video_path = self._validate_video_path(video_path)
        if video_path is None:
            return None

        try:
            logger.info(f"Analyzing scenes: {video_path.name}")

            # Get cache path
            stats_path = None
            if self.use_cache:
                stats_path = self._get_stats_cache_path(video_path)
                if stats_path.exists():
                    logger.debug(f"Using cached stats: {stats_path.name}")

            # Create detector
            detector = self._create_detector(detector_type, threshold, min_scene_len)

            # Detect scenes
            scenes = detect(
                str(video_path),
                detector,
                stats_file_path=str(stats_path) if stats_path else None,
                show_progress=False,
            )

            if not scenes:
                logger.warning(f"No scenes detected in {video_path.name}")
                scenes = []

            # Convert to structured format
            scene_list: list[SceneTimestamp] = []
            timestamps: list[float] = []

            for idx, (start, end) in enumerate(scenes):
                scene_info: SceneTimestamp = {
                    "scene_index": idx,
                    "start_seconds": start.get_seconds(),
                    "end_seconds": end.get_seconds(),
                    "duration_seconds": end.get_seconds() - start.get_seconds(),
                    "start_frame": start.get_frames(),
                    "end_frame": end.get_frames(),
                }
                scene_list.append(scene_info)

                # Flat list for database storage
                timestamps.append(start.get_seconds())
                timestamps.append(end.get_seconds())

            # Create result
            result: SceneDetectionResult = {
                "video_path": str(video_path.absolute()),
                "total_scenes": len(scenes),
                "scene_timestamps": timestamps,
                "scenes": scene_list,
                "detector_type": detector_type,
                "threshold": threshold,
            }

            logger.info(
                f"Scene analysis complete: {len(scenes)} scenes detected " f"({video_path.name})"
            )

            # Store in database if video_clip_id provided
            if video_clip_id is not None and self.session is not None:
                self._store_scenes_in_db(video_clip_id, result)

            return result

        except VideoOpenFailure as e:
            logger.error(f"Failed to open video {video_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error analyzing scenes: {e}", exc_info=True)
            return None

    def _store_scenes_in_db(self, video_clip_id: int, result: SceneDetectionResult) -> bool:
        """
        Store scene detection results in database.

        Args:
            video_clip_id: VideoClip ID
            result: SceneDetectionResult to store

        Returns:
            True if successful, False otherwise
        """
        if self.session is None:
            logger.warning("No database session configured, skipping storage")
            return False

        try:
            # Get video clip from database
            stmt = select(VideoClip).where(VideoClip.id == video_clip_id)
            video_clip = self.session.execute(stmt).scalar_one_or_none()

            if video_clip is None:
                logger.warning(f"VideoClip ID {video_clip_id} not found")
                return False

            # Store scene timestamps
            video_clip.set_scene_timestamps(result["scene_timestamps"])

            self.session.commit()

            logger.info(f"Stored {result['total_scenes']} scenes for VideoClip ID {video_clip_id}")
            return True

        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Database error storing scenes: {e}", exc_info=True)
            return False
        except Exception as e:
            self.session.rollback()
            logger.error(f"Unexpected error storing scenes: {e}", exc_info=True)
            return False

    def clear_cache(self, video_path: str | Path | None = None) -> bool:
        """
        Clear stats cache for specific video or all cached stats.

        Args:
            video_path: Path to video (None = clear all cache)

        Returns:
            True if successful, False otherwise
        """
        try:
            if video_path is not None:
                # Clear specific video cache
                video_path = Path(video_path)
                stats_path = self._get_stats_cache_path(video_path)
                if stats_path.exists():
                    stats_path.unlink()
                    logger.info(f"Cleared cache for {video_path.name}")
                else:
                    logger.debug(f"No cache found for {video_path.name}")
            else:
                # Clear all cache
                cache_files = list(self.cache_dir.glob("*.stats.csv"))
                for cache_file in cache_files:
                    cache_file.unlink()
                logger.info(f"Cleared {len(cache_files)} cached stats files")

            return True

        except Exception as e:
            logger.error(f"Error clearing cache: {e}", exc_info=True)
            return False

    def get_video_info(self, video_path: str | Path) -> dict | None:
        """
        Get basic video information using PySceneDetect's VideoStream.

        Args:
            video_path: Path to video file

        Returns:
            Dict with video info or None on error
        """
        video_path = self._validate_video_path(video_path)
        if video_path is None:
            return None

        video = None
        try:
            video = open_video(str(video_path))

            info = {
                "duration_seconds": video.duration.get_seconds(),
                "total_frames": video.duration.get_frames(),
                "frame_rate": video.frame_rate,
                "resolution": f"{video.frame_size[0]}x{video.frame_size[1]}",
                "width": video.frame_size[0],
                "height": video.frame_size[1],
            }

            return info

        except VideoOpenFailure as e:
            logger.error(f"Failed to open video {video_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting video info: {e}", exc_info=True)
            return None
        finally:
            # BUG FIX: Explicitly release video stream to prevent resource leak
            if video is not None:
                try:
                    video.release()
                except Exception as e:
                    logger.debug(f"Error releasing video stream: {e}")

    def analyze_video_properties(self, video_path: str | Path) -> VideoProperties | None:
        """
        Analyze video for technical properties using OpenCV.
        Transferred from VideoCompatibilityManager.

        Args:
            video_path: Path to video file

        Returns:
            VideoProperties object or None if failed
        """
        video_path = self._validate_video_path(video_path)
        if video_path is None:
            return None

        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")

            # Extract basic properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Handle variable FPS
            is_variable_fps = fps <= 0 or fps > 1000
            if is_variable_fps:
                fps = 25.0  # Default assumption for VFR
                logger.warning(f"Variable FPS detected in {video_path.name}, assuming {fps} fps")

            duration = frame_count / fps if fps > 0 else 0.0

            # Get codec information
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)]).strip()

            cap.release()

            # Logic originally from CompatibilityChecker

            # Simple quality score calculation
            resolution_score = min(width * height / (1920 * 1080), 1.0) * 0.6
            fps_score = min(fps / 60.0, 1.0) * 0.4
            quality_score = min(resolution_score + fps_score, 1.0)

            # Optimal sample rate (simplified)
            optimal_sample_rate = 1.0
            if fps >= 60:
                optimal_sample_rate = 0.25
            elif fps >= 30:
                optimal_sample_rate = 0.5

            # Recommended resize (simplified logic)
            recommended_resize = None
            max_w, max_h = (1920, 1080)  # Default safe max
            if width > max_w or height > max_h:
                scale = min(max_w / width, max_h / height)
                new_w = (int(width * scale) // 2) * 2
                new_h = (int(height * scale) // 2) * 2
                recommended_resize = (new_w, new_h)

            memory_per_frame_mb = (width * height * 3) / (1024 * 1024)
            total_memory_estimate_mb = memory_per_frame_mb * min(frame_count, 100)

            supports_gpu_decode = codec.lower() in ["h264", "h265", "vp9", "av1"]
            supports_fast_seek = supports_gpu_decode

            return VideoProperties(
                width=width,
                height=height,
                fps=fps,
                frame_count=frame_count,
                duration=duration,
                codec=codec,
                quality_score=quality_score,
                optimal_sample_rate=optimal_sample_rate,
                recommended_resize=recommended_resize,
                memory_per_frame_mb=memory_per_frame_mb,
                total_memory_estimate_mb=total_memory_estimate_mb,
                supports_gpu_decode=supports_gpu_decode,
                supports_fast_seek=supports_fast_seek,
                is_variable_fps=is_variable_fps,
            )

        except Exception as e:
            logger.error(f"Video property analysis failed: {e}")
            return None

    def _get_cached_semantics(self, clip_id: int) -> dict | None:
        """
        Lädt gecachte semantische Analyse aus DB.

        Args:
            clip_id: VideoClip ID

        Returns:
            Dict mit gespeicherten Ergebnissen oder None
        """
        if not self.session:
            return None

        try:
            semantics = self.session.get(ClipSemantics, clip_id)
            if semantics and semantics.raw_results:
                import json

                results = json.loads(semantics.raw_results)
                logger.debug(f"Cache-Hit für Clip {clip_id} semantische Analyse")
                return results
        except Exception as e:
            logger.debug(f"Keine gecachten Semantics für Clip {clip_id}: {e}")

        return None

    def _save_semantics_to_db(self, clip_id: int, results: dict) -> bool:
        """
        Speichert semantische Analyse-Ergebnisse in DB.

        Args:
            clip_id: VideoClip ID
            results: Analyse-Ergebnisse

        Returns:
            True bei Erfolg, False bei Fehler
        """
        if not self.session:
            return False

        try:
            semantics = self.session.get(ClipSemantics, clip_id)
            if semantics is None:
                semantics = ClipSemantics(clip_id=clip_id)
                self.session.add(semantics)

            semantics.set_results(results)
            self.session.commit()

            logger.info(
                f"CLIP-Analyse für Clip {clip_id} gespeichert "
                f"(Scene={results.get('scene_type')}, Mood={results.get('mood')})"
            )
            return True

        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"DB-Fehler beim Speichern der Semantics: {e}", exc_info=True)
            return False
        except Exception as e:
            self.session.rollback()
            logger.error(f"Fehler beim Speichern der Semantics: {e}", exc_info=True)
            return False

    def analyze_clip_semantics(
        self,
        video_path: str | Path,
        video_clip_id: int | None = None,
        sample_interval: float = 2.0,
        force_reanalyze: bool = False,
    ) -> dict | None:
        """
        Führt semantische CLIP-Analyse für einen Video-Clip durch.

        Kategorisiert Video nach Szenentyp, Stimmung und Inhalt.

        Args:
            video_path: Pfad zur Video-Datei
            video_clip_id: Optional VideoClip ID für DB-Update
            sample_interval: Sekunden zwischen Frame-Samples
            force_reanalyze: Erzwingt Neu-Analyse auch bei vorhandenen Ergebnissen

        Returns:
            Dict mit Kategorisierungs-Ergebnissen oder None
        """
        from .clip_analyzer import CLIPVideoAnalyzer, is_clip_available

        if not is_clip_available():
            logger.warning("CLIP nicht verfügbar - semantische Analyse übersprungen")
            return None

        # Cache-Check: Existierende Ergebnisse laden
        if not force_reanalyze and video_clip_id:
            cached_results = self._get_cached_semantics(video_clip_id)
            if cached_results:
                return cached_results

        try:
            # CLIP-Analyse durchführen
            analyzer = CLIPVideoAnalyzer()
            results = analyzer.analyze_video(
                video_path, sample_interval=sample_interval, max_frames=30
            )

            if "error" in results:
                logger.error(f"CLIP-Analyse Fehler: {results['error']}")
                return None

            # DB-Speicherung
            if video_clip_id:
                self._save_semantics_to_db(video_clip_id, results)

            return results

        except Exception as e:
            logger.error(f"Semantische Analyse fehlgeschlagen: {e}", exc_info=True)
            return None
