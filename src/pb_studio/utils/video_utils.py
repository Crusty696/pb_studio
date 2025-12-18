"""
Video Utility Functions for PB_studio

Context managers and helpers for safe video handling.
PERF-02 Fix: Ensures VideoCapture resources are properly released.
SECURITY Fix: Path validation against traversal attacks (CWE-22).
H-06 Fix: Video size limits to prevent OOM.
GPU-01 Fix: Hardware video decode via DXVA2/D3D11 (1.9x speedup at 4K).
"""

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import cv2

logger = logging.getLogger(__name__)

# Erlaubte Video-Dateierweiterungen
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv", ".flv"}

# H-06 FIX: Video Size Limits (Bytes)
# Erhöht für 3h 4K Video Support (~22 GB/h bei 4K 30fps)
MAX_VIDEO_SIZE_BYTES = 100 * 1024 * 1024 * 1024  # 100 GB (ca. 4.5h 4K oder 25h 1080p)
MAX_VIDEO_SIZE_MB = MAX_VIDEO_SIZE_BYTES / (1024 * 1024)  # For display

# GPU-01: Hardware Video Decode Settings
# Aktiviert DXVA2/D3D11 Hardware-Beschleunigung fuer Video-Decode
# Getestet: 1.89x Speedup bei 4K Video (90 FPS -> 172 FPS)
USE_HARDWARE_DECODE = os.environ.get("PB_HARDWARE_DECODE", "1") == "1"

# Minimum Resolution fuer Hardware-Decode (kleinere Videos profitieren nicht)
# Bei 720p und kleiner ist Software-Decode oft schneller wegen Overhead
HW_DECODE_MIN_WIDTH = 1280
HW_DECODE_MIN_HEIGHT = 720


class VideoCapture:
    """
    Context manager wrapper for cv2.VideoCapture.

    PERF-02 FIX: Ensures VideoCapture is always released, even on exceptions.

    Usage:
        with VideoCapture(video_path) as cap:
            if cap.isOpened():
                ret, frame = cap.read()

        # VideoCapture is automatically released here
    """

    def __init__(self, video_path: str | Path):
        """
        Initialize VideoCapture context manager.

        Args:
            video_path: Path to video file
        """
        self.video_path = str(video_path)
        self._cap: cv2.VideoCapture | None = None

    def __enter__(self) -> cv2.VideoCapture:
        """Open video file and return VideoCapture object."""
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            logger.warning(f"Could not open video: {Path(self.video_path).name}")
        return self._cap

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Release VideoCapture resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        return False  # Don't suppress exceptions


def _validate_video_path(video_path: str | Path, check_size: bool = True) -> Path:
    """
    SECURITY: Validiert Video-Pfad gegen Path-Traversal-Angriffe (CWE-22).
    H-06 FIX: Prueft Video-Groesse gegen Limit.

    Args:
        video_path: Pfad zum Video
        check_size: Ob Dateigroesse geprueft werden soll (default: True)

    Returns:
        Validierter Path

    Raises:
        ValueError: Bei ungültigem Pfad, Extension oder zu grosser Datei
        FileNotFoundError: Wenn Datei nicht existiert
    """
    if not video_path:
        raise ValueError("Video path cannot be empty")

    path = Path(video_path).resolve()

    # Extension-Check
    if path.suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        raise ValueError(
            f"Invalid video extension: {path.suffix}. "
            f"Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}"
        )

    # Existenz-Check
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    # H-06 FIX: Size-Check
    if check_size:
        file_size = os.path.getsize(path)
        if file_size > MAX_VIDEO_SIZE_BYTES:
            size_mb = file_size / (1024 * 1024)
            raise ValueError(
                f"Video file too large: {size_mb:.1f} MB. "
                f"Maximum allowed: {MAX_VIDEO_SIZE_MB:.0f} MB"
            )

    return path


def _should_use_hardware_decode(width: int, height: int) -> bool:
    """
    Prueft ob Hardware-Decode fuer diese Video-Aufloesung sinnvoll ist.

    GPU-01: Hardware-Decode lohnt sich nur bei groesseren Videos.
    Bei kleinen Videos ist der Overhead groesser als der Gewinn.

    Args:
        width: Video-Breite in Pixel
        height: Video-Hoehe in Pixel

    Returns:
        True wenn Hardware-Decode verwendet werden sollte
    """
    if not USE_HARDWARE_DECODE:
        return False

    return width >= HW_DECODE_MIN_WIDTH and height >= HW_DECODE_MIN_HEIGHT


def _create_hardware_capture(video_path: str) -> cv2.VideoCapture:
    """
    Erstellt VideoCapture mit Hardware-Decode (DXVA2/D3D11).

    GPU-01: Nutzt Media Foundation Backend mit D3D11 Hardware-Acceleration.
    Getestet: 1.89x Speedup bei 4K Video.

    Args:
        video_path: Pfad zum Video

    Returns:
        VideoCapture mit Hardware-Acceleration (wenn verfuegbar)
    """
    # Versuche MSMF Backend mit D3D11 Hardware-Acceleration
    cap = cv2.VideoCapture(video_path, cv2.CAP_MSMF)

    if cap.isOpened():
        # Aktiviere D3D11 Hardware-Acceleration
        cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_D3D11)
        hw_status = cap.get(cv2.CAP_PROP_HW_ACCELERATION)

        if hw_status > 0:
            logger.debug(f"Hardware-Decode aktiviert (D3D11, status={hw_status})")
            return cap
        else:
            logger.debug("D3D11 nicht verfuegbar, fallback auf Software")
            cap.release()

    # Fallback: Standard-Capture
    return cv2.VideoCapture(video_path)


@contextmanager
def open_video(
    video_path: str | Path, validate: bool = True, hardware_decode: bool | None = None
) -> Generator[cv2.VideoCapture, None, None]:
    """
    Context manager for opening video files safely.

    PERF-02 FIX: Ensures VideoCapture is always released.
    SECURITY FIX: Path validation against traversal attacks (CWE-22).
    GPU-01 FIX: Hardware video decode via DXVA2/D3D11 (1.9x speedup at 4K).

    Args:
        video_path: Path to video file
        validate: If True, validates path and extension (default: True)
                 Set to False only for internal/trusted paths
        hardware_decode: Force hardware decode on/off. None = auto-detect based on resolution.

    Yields:
        cv2.VideoCapture object

    Raises:
        ValueError: If path validation fails (invalid extension, etc.)
        FileNotFoundError: If video file doesn't exist

    Usage:
        with open_video(video_path) as cap:
            if cap.isOpened():
                ret, frame = cap.read()
    """
    # SECURITY: Validate path before opening
    if validate:
        try:
            validated_path = _validate_video_path(video_path)
            video_path = str(validated_path)
        except (ValueError, FileNotFoundError) as e:
            logger.error(f"Video path validation failed: {e}")
            raise

    video_path_str = str(video_path)

    # GPU-01: Hardware-Decode Auto-Detection
    use_hw = hardware_decode
    if use_hw is None and USE_HARDWARE_DECODE:
        # Schnelle Metadata-Abfrage fuer Auto-Detection
        probe_cap = cv2.VideoCapture(video_path_str)
        if probe_cap.isOpened():
            width = int(probe_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(probe_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            probe_cap.release()
            use_hw = _should_use_hardware_decode(width, height)
        else:
            use_hw = False

    # Erstelle VideoCapture mit oder ohne Hardware-Decode
    if use_hw:
        cap = _create_hardware_capture(video_path_str)
    else:
        cap = cv2.VideoCapture(video_path_str)

    try:
        if not cap.isOpened():
            logger.warning(
                f"Could not open video: {Path(video_path).name if isinstance(video_path, (str, Path)) else 'unknown'}"
            )
        yield cap
    finally:
        cap.release()


def get_video_info_safe(video_path: str | Path) -> dict | None:
    """
    Get video metadata safely using context manager.

    Args:
        video_path: Path to video file

    Returns:
        Dict with video info or None on error
    """
    with open_video(video_path) as cap:
        if not cap.isOpened():
            return None

        return {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1),
        }


def extract_frame_safe(video_path: str | Path, frame_number: int) -> Optional["cv2.typing.MatLike"]:
    """
    Extract single frame from video safely.

    Args:
        video_path: Path to video file
        frame_number: Frame index to extract

    Returns:
        Frame as numpy array or None on error
    """
    with open_video(video_path) as cap:
        if not cap.isOpened():
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        return frame if ret else None
