"""
Video Manager for PB_studio

Production-ready video file management with FFmpeg integration.

Features:
- Video metadata extraction (duration, resolution, codec, fps)
- Thumbnail generation with caching
- Thread-safe operations
- Comprehensive error handling
- Database integration

Dependencies:
- ffmpeg-python
- Pillow (for thumbnail processing)
- SQLAlchemy (for database)

Usage:
    manager = VideoManager(session=db_session)

    # Import video
    video_clip = manager.import_video('video.mp4', project_id=1)

    # Get metadata
    metadata = manager.get_metadata('video.mp4')

    # Generate thumbnail
    thumbnail_path = manager.generate_thumbnail('video.mp4', time_offset=5.0)
"""

import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Literal, TypedDict

import ffmpeg
from PIL import Image
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ..database.models import VideoClip
from .thumbnail_generator import ThumbnailGenerator


def ffprobe_with_timeout(video_path: str, timeout: int = 30) -> dict[str, Any] | None:
    """
    FIX #9: ffmpeg.probe() mit Timeout um hängende Prozesse zu vermeiden.

    Bei beschädigten Dateien oder Netzwerk-Problemen kann ffprobe hängen.
    Diese Funktion setzt ein Timeout um das zu verhindern.

    FIX: Uses Popen with explicit kill() to prevent zombie processes on all Python versions.

    Args:
        video_path: Pfad zur Video-Datei
        timeout: Timeout in Sekunden (default: 30)

    Returns:
        Probe-Ergebnis als Dict oder None bei Fehler/Timeout
    """
    process = None
    try:
        # Verwende subprocess direkt mit Timeout statt ffmpeg-python
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]

        # FIX: Use Popen instead of run() for explicit process control
        creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=creationflags,
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # FIX: Explicitly kill zombie process on timeout
            logging.getLogger(__name__).error(f"ffprobe timeout after {timeout}s, killing process...")
            process.kill()
            process.wait()  # Ensure process is fully terminated
            return None

        if process.returncode != 0:
            logging.getLogger(__name__).error(f"ffprobe failed: {stderr}")
            return None

        return json.loads(stdout)

    except json.JSONDecodeError as e:
        logging.getLogger(__name__).error(f"ffprobe output parse error: {e}")
        return None
    except FileNotFoundError:
        logging.getLogger(__name__).error("ffprobe not found in PATH")
        return None
    except Exception as e:
        logging.getLogger(__name__).error(f"ffprobe error: {e}")
        # FIX: Ensure process is killed on any exception
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        return None


# SECURITY: python-magic for MIME type validation (File Upload Protection)
try:
    import magic

    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    logging.warning(
        "python-magic not installed - MIME type validation DISABLED. "
        "Install with: pip install python-magic python-magic-bin (Windows)"
    )

logger = logging.getLogger(__name__)

# SECURITY: Allowed video MIME types (File Upload Validation)
ALLOWED_VIDEO_MIME_TYPES = {
    "video/mp4",
    "video/mpeg",
    "video/quicktime",  # .mov
    "video/x-msvideo",  # .avi
    "video/x-matroska",  # .mkv
    "video/webm",
    "video/x-flv",
    "video/x-m4v",
}


class VideoMetadata(TypedDict):
    """Video file metadata."""

    duration: float
    width: int
    height: int
    codec: str
    fps: float
    bitrate: int
    size_bytes: int
    format: str


class VideoManager:
    """
    Production-ready video file manager with FFmpeg integration.

    Handles video import, metadata extraction, and thumbnail generation
    with caching and database integration.
    """

    def __init__(
        self,
        session: Session | None = None,
        thumbnail_dir: str | Path = "thumbnails",
        thumbnail_width: int = 320,
    ) -> None:
        """
        Initialize the VideoManager.

        Args:
            session: SQLAlchemy session for database operations (optional)
            thumbnail_dir: Directory for storing thumbnails
            thumbnail_width: Width of generated thumbnails (maintains aspect ratio)
        """
        self.session = session
        self.thumbnail_dir = Path(thumbnail_dir)
        self.thumbnail_width = thumbnail_width

        # Create thumbnail directory
        self.thumbnail_dir.mkdir(parents=True, exist_ok=True)

        # Initialize thumbnail generator
        self.thumbnail_generator = ThumbnailGenerator(
            cache_dir=thumbnail_dir,
            thumbnail_size=(thumbnail_width, int(thumbnail_width * 9 / 16)),  # 16:9 aspect ratio
            quality=85,
        )

    def __repr__(self) -> str:
        """String representation for debugging."""
        session_status = "configured" if self.session else "not configured"
        return f"<VideoManager(session={session_status}, thumbnail_dir={self.thumbnail_dir})>"

    def _validate_mime_type(self, file_path: Path) -> bool:
        """
        Validate file MIME type using python-magic (Magic Bytes validation).

        Security Features:
        - Checks actual file content (Magic Bytes), not extension
        - Protects against malicious files renamed to .mp4
        - CWE-434 (Unrestricted Upload of File with Dangerous Type)

        Args:
            file_path: Path to file

        Returns:
            True if MIME type is allowed, False otherwise
        """
        if not MAGIC_AVAILABLE:
            logger.warning(
                f"MIME validation skipped (python-magic not installed): {file_path.name}"
            )
            return True  # Graceful fallback wenn python-magic fehlt

        try:
            # Get MIME type from magic bytes (not extension!)
            mime = magic.Magic(mime=True)
            mime_type = mime.from_file(str(file_path))

            if mime_type not in ALLOWED_VIDEO_MIME_TYPES:
                logger.error(
                    f"Security: Invalid MIME type '{mime_type}' for {file_path.name}. "
                    f"Allowed: {', '.join(sorted(ALLOWED_VIDEO_MIME_TYPES))}"
                )
                return False

            logger.debug(f"MIME type validated: {mime_type} ({file_path.name})")
            return True

        except Exception as e:
            # SEC-02 FIX: Use DEBUG for full paths, ERROR shows only filename
            logger.error(f"Error validating MIME type for {file_path.name}: {e}")
            logger.debug(f"Full path: {file_path}")
            return False

    def _validate_video_path(
        self, video_path: str | Path, validate_mime: bool = True
    ) -> Path | None:
        """
        Validate video file path exists and optionally check MIME type.

        Security Features:
        - File existence check
        - Optional MIME type validation (Magic Bytes)
        - CWE-434 Protection (File Upload Validation)

        Args:
            video_path: Path to validate
            validate_mime: Enable MIME type validation (default: True)

        Returns:
            Path object if valid, None if invalid
        """
        video_path = Path(video_path)

        # Check file exists
        if not video_path.exists():
            # SEC-02 FIX: Don't expose full paths in error messages
            logger.error(f"Video file not found: {video_path.name}")
            logger.debug(f"Full path: {video_path}")
            return None

        # SECURITY: Validate MIME type (Magic Bytes)
        if validate_mime and not self._validate_mime_type(video_path):
            # SEC-02 FIX: Don't expose full paths in error messages
            logger.error(f"File upload validation failed: {video_path.name}")
            logger.debug(f"Full path: {video_path}")
            return None

        return video_path

    def _get_file_hash(self, file_path: Path) -> str:
        """
        Generate MD5 hash for file (for caching).

        Args:
            file_path: Path to file

        Returns:
            MD5 hash string
        """
        hasher = hashlib.md5()
        hasher.update(str(file_path).encode())
        stat = file_path.stat()
        hasher.update(f"{stat.st_size}_{stat.st_mtime}".encode())
        return hasher.hexdigest()

    def get_metadata(self, video_path: str | Path) -> VideoMetadata | None:
        """
        Extract video metadata using FFmpeg.

        Args:
            video_path: Path to video file

        Returns:
            VideoMetadata dict or None on error
        """
        video_path = self._validate_video_path(video_path)
        if video_path is None:
            return None

        try:
            logger.info(f"Extracting metadata: {video_path.name}")

            # FIX #9: Probe video file mit Timeout
            probe = ffprobe_with_timeout(str(video_path), timeout=30)
            if probe is None:
                logger.error(f"Failed to probe video: {video_path.name}")
                return None

            # FIX: Defensive validation of probe structure before accessing
            if not isinstance(probe, dict):
                logger.error(f"Invalid probe output type for {video_path.name}: {type(probe)}")
                return None

            if "streams" not in probe or not isinstance(probe["streams"], list):
                logger.error(f"Malformed probe output (no streams) for {video_path.name}")
                return None

            if "format" not in probe or not isinstance(probe["format"], dict):
                logger.error(f"Malformed probe output (no format) for {video_path.name}")
                return None

            # Get video stream
            video_stream = next(
                (s for s in probe["streams"]
                 if isinstance(s, dict) and s.get("codec_type") == "video"),
                None
            )

            if video_stream is None:
                # SEC-02 FIX: Don't expose full paths in error messages
                logger.error(f"No video stream found in {video_path.name}")
                return None

            # FIX: Validate required fields exist before extraction
            format_data = probe["format"]
            if "duration" not in format_data:
                logger.error(f"No duration in probe output for {video_path.name}")
                return None

            # Extract metadata with defensive .get() for optional fields
            duration = float(format_data["duration"])
            width = int(video_stream.get("width", 0))
            height = int(video_stream.get("height", 0))
            codec = video_stream.get("codec_name", "unknown")
            bitrate = int(format_data.get("bit_rate", 0))
            format_name = format_data.get("format_name", "unknown")

            # FIX: Validate width/height are positive
            if width <= 0 or height <= 0:
                logger.error(f"Invalid video dimensions ({width}x{height}) for {video_path.name}")
                return None

            # Calculate FPS with validation
            fps_str = video_stream.get("r_frame_rate", "0/1")
            parts = fps_str.split("/")
            if len(parts) != 2:
                logger.warning(
                    f"Invalid FPS format '{fps_str}' for {video_path}, defaulting to 0.0"
                )
                fps = 0.0
            else:
                fps_num, fps_den = map(int, parts)
                fps = fps_num / fps_den if fps_den != 0 else 0.0

            # Get file size
            size_bytes = video_path.stat().st_size

            metadata: VideoMetadata = {
                "duration": duration,
                "width": width,
                "height": height,
                "codec": codec,
                "fps": fps,
                "bitrate": bitrate,
                "size_bytes": size_bytes,
                "format": format_name,
            }

            logger.info(
                f"Metadata extracted: {width}x{height} @ {fps:.2f}fps, "
                f"{duration:.2f}s ({video_path.name})"
            )

            return metadata

        except ffmpeg.Error as e:
            # SEC-02 FIX: Don't expose full paths in error messages
            logger.error(
                f"FFmpeg error extracting metadata from {video_path.name}: {e.stderr.decode()}"
            )
            return None
        except Exception as e:
            # SEC-02 FIX: Don't expose full paths in error messages
            logger.error(f"Unexpected error extracting metadata for {video_path.name}: {e}")
            logger.debug(f"Full path: {video_path}", exc_info=True)
            return None

    def generate_thumbnail(
        self,
        video_path: str | Path,
        time_offset: float = 1.0,
        format: Literal["jpg", "png"] = "jpg",
        quality: int = 85,
    ) -> Path | None:
        """
        Generate thumbnail from video at specified time.

        Args:
            video_path: Path to video file
            time_offset: Time offset in seconds for thumbnail
            format: Image format ('jpg' or 'png')
            quality: JPEG quality (1-100, ignored for PNG)

        Returns:
            Path to generated thumbnail or None on error
        """
        video_path = self._validate_video_path(video_path)
        if video_path is None:
            return None

        try:
            # Generate thumbnail filename
            file_hash = self._get_file_hash(video_path)
            thumbnail_name = f"{file_hash}_{time_offset:.1f}.{format}"
            thumbnail_path = self.thumbnail_dir / thumbnail_name

            # Return cached thumbnail if exists
            if thumbnail_path.exists():
                logger.debug(f"Using cached thumbnail: {thumbnail_path.name}")
                return thumbnail_path

            logger.info(f"Generating thumbnail: {video_path.name} @ {time_offset}s")

            # Extract frame with ffmpeg
            temp_output = self.thumbnail_dir / f"temp_{file_hash}.png"

            (
                ffmpeg.input(str(video_path), ss=time_offset)
                .filter("scale", self.thumbnail_width, -1)  # Maintain aspect ratio
                .output(str(temp_output), vframes=1, format="image2", vcodec="png")
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )

            # Convert to desired format if needed
            if format == "jpg":
                # Convert PNG to JPEG with quality
                img = Image.open(temp_output)
                img = img.convert("RGB")  # Remove alpha channel
                img.save(thumbnail_path, "JPEG", quality=quality, optimize=True)
                temp_output.unlink()  # Remove temp file
            else:
                # Rename PNG
                temp_output.rename(thumbnail_path)

            logger.info(f"Thumbnail generated: {thumbnail_path.name}")
            return thumbnail_path

        except ffmpeg.Error as e:
            # SEC-02 FIX: Don't expose full paths in error messages
            logger.error(
                f"FFmpeg error generating thumbnail for {video_path.name}: {e.stderr.decode()}"
            )
            return None
        except Exception as e:
            # SEC-02 FIX: Don't expose full paths in error messages
            logger.error(f"Unexpected error generating thumbnail for {video_path.name}: {e}")
            logger.debug(f"Full path: {video_path}", exc_info=True)
            return None

    def import_video(
        self,
        video_path: str | Path,
        project_id: int,
        tags: str = "",
        generate_thumbnail: bool = True,
    ) -> VideoClip | None:
        """
        Import video file into database with metadata and thumbnail.

        Args:
            video_path: Path to video file
            project_id: ID of the project
            tags: Optional tags (comma-separated)
            generate_thumbnail: Generate thumbnail automatically

        Returns:
            VideoClip database object or None on error

        Raises:
            ValueError: If session is not configured
        """
        if self.session is None:
            raise ValueError("Database session not configured for this manager instance")

        video_path = self._validate_video_path(video_path)
        if video_path is None:
            return None

        try:
            logger.info(f"Importing video: {video_path.name}")

            # Extract metadata
            metadata = self.get_metadata(video_path)
            if metadata is None:
                # SEC-02 FIX: Don't expose full paths in error messages
                logger.error(f"Failed to extract metadata for {video_path.name}")
                return None

            # Generate thumbnail using ThumbnailGenerator
            thumbnail_path: Path | None = None
            if generate_thumbnail:
                thumbnail_path = self.thumbnail_generator.generate(video_path)

            # Create VideoClip database entry
            video_clip = VideoClip(
                project_id=project_id,
                file_path=str(video_path.absolute()),
                filename=video_path.name,
                duration=metadata["duration"],
                width=metadata["width"],
                height=metadata["height"],
                fps=metadata["fps"],
                codec=metadata["codec"],
                format=metadata["format"],
                size_bytes=metadata["size_bytes"],
                thumbnail_path=str(thumbnail_path) if thumbnail_path else None,
                tags=tags,
            )

            self.session.add(video_clip)
            self.session.commit()
            self.session.refresh(video_clip)

            logger.info(f"Video imported: ID {video_clip.id}, {video_path.name}")
            return video_clip

        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Database error importing video: {e}", exc_info=True)
            return None
        except Exception as e:
            self.session.rollback()
            logger.error(f"Unexpected error importing video: {e}", exc_info=True)
            return None

    def update_metadata(self, video_clip_id: int) -> bool:
        """
        Update metadata for existing video clip.

        Args:
            video_clip_id: ID of VideoClip to update

        Returns:
            True if successful, False otherwise

        Raises:
            ValueError: If session is not configured
        """
        if self.session is None:
            raise ValueError("Database session not configured for this manager instance")

        try:
            # Get video clip from database
            stmt = select(VideoClip).where(VideoClip.id == video_clip_id)
            video_clip = self.session.execute(stmt).scalar_one_or_none()

            if video_clip is None:
                logger.warning(f"VideoClip ID {video_clip_id} not found")
                return False

            # Extract fresh metadata
            metadata = self.get_metadata(video_clip.file_path)
            if metadata is None:
                return False

            # Update video clip
            video_clip.duration = metadata["duration"]
            video_clip.width = metadata["width"]
            video_clip.height = metadata["height"]
            video_clip.fps = metadata["fps"]
            video_clip.codec = metadata["codec"]
            video_clip.format = metadata["format"]
            video_clip.size_bytes = metadata["size_bytes"]

            self.session.commit()

            logger.info(f"Metadata updated: VideoClip ID {video_clip_id}")
            return True

        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Database error updating metadata: {e}", exc_info=True)
            return False
        except Exception as e:
            self.session.rollback()
            logger.error(f"Unexpected error updating metadata: {e}", exc_info=True)
            return False

    def regenerate_thumbnail(self, video_clip_id: int, time_offset: float = 1.0) -> bool:
        """
        Regenerate thumbnail for existing video clip.

        Args:
            video_clip_id: ID of VideoClip
            time_offset: Time offset for thumbnail

        Returns:
            True if successful, False otherwise

        Raises:
            ValueError: If session is not configured
        """
        if self.session is None:
            raise ValueError("Database session not configured for this manager instance")

        try:
            # Get video clip from database
            stmt = select(VideoClip).where(VideoClip.id == video_clip_id)
            video_clip = self.session.execute(stmt).scalar_one_or_none()

            if video_clip is None:
                logger.warning(f"VideoClip ID {video_clip_id} not found")
                return False

            # Generate new thumbnail
            thumbnail_path = self.generate_thumbnail(video_clip.file_path, time_offset=time_offset)

            if thumbnail_path is None:
                return False

            # Update database
            video_clip.thumbnail_path = str(thumbnail_path)
            self.session.commit()

            logger.info(f"Thumbnail regenerated: VideoClip ID {video_clip_id}")
            return True

        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Database error regenerating thumbnail: {e}", exc_info=True)
            return False
        except Exception as e:
            self.session.rollback()
            logger.error(f"Unexpected error regenerating thumbnail: {e}", exc_info=True)
            return False

    def verify_video_file(self, video_path: str | Path) -> bool:
        """
        Verify video file is readable and valid.

        Args:
            video_path: Path to video file

        Returns:
            True if valid, False otherwise
        """
        video_path = self._validate_video_path(video_path)
        if video_path is None:
            return False

        try:
            # FIX #9: Try to probe the file mit Timeout
            probe = ffprobe_with_timeout(str(video_path), timeout=15)
            return probe is not None
        except Exception as e:
            logger.debug(f"Video validation failed for {video_path}: {e}")
            return False

    def get_video_codec_info(self, video_path: str | Path) -> dict | None:
        """
        Get detailed codec information for video file.

        Args:
            video_path: Path to video file

        Returns:
            Dict with codec details or None on error
        """
        video_path = self._validate_video_path(video_path)
        if video_path is None:
            return None

        try:
            # FIX #9: Probe mit Timeout
            probe = ffprobe_with_timeout(str(video_path), timeout=30)
            if probe is None:
                return None

            video_stream = next((s for s in probe["streams"] if s["codec_type"] == "video"), None)

            if video_stream is None:
                return None

            return {
                "codec_name": video_stream.get("codec_name"),
                "codec_long_name": video_stream.get("codec_long_name"),
                "profile": video_stream.get("profile"),
                "pix_fmt": video_stream.get("pix_fmt"),
                "level": video_stream.get("level"),
                "bit_rate": video_stream.get("bit_rate"),
            }

        except Exception as e:
            logger.error(f"Error getting codec info: {e}")
            return None
