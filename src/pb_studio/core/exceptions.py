"""
Custom Exceptions für PB_Studio.

Hierarchie:
    PBStudioError (Base)
    ├── ConfigurationError
    ├── DatabaseError
    │   └── SessionError
    ├── AudioError
    │   ├── AudioAnalysisError
    │   └── InvalidAudioFormatError
    ├── VideoError
    │   ├── ClipNotFoundError
    │   ├── InvalidVideoFormatError
    │   └── RenderError
    │       ├── FFmpegError
    │       └── RenderTimeoutError
    └── PacingError
        ├── BeatgridError
        └── ClipMatchError
"""


class PBStudioError(Exception):
    """
    Base exception für alle PB_Studio Fehler.

    Alle custom exceptions erben von dieser Klasse.
    Ermöglicht spezifisches Exception-Handling für die Anwendung.
    """

    def __init__(self, message: str = "", details: dict = None):
        """
        Initialize exception with message and optional details.

        Args:
            message: Human-readable error message
            details: Optional dict with additional context
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(PBStudioError):
    """
    Konfigurations-Fehler.

    Raised when:
    - Config file is missing or invalid
    - Required settings are not set
    - Invalid configuration values
    """

    pass


# =============================================================================
# Database Errors
# =============================================================================


class DatabaseError(PBStudioError):
    """
    Datenbank-Fehler.

    Raised when:
    - Database connection fails
    - Query execution fails
    - Data integrity issues
    """

    pass


class SessionError(DatabaseError):
    """
    SQLAlchemy Session-Fehler.

    Raised when:
    - Session cannot be created
    - Session commit/rollback fails
    - Connection pool exhausted
    """

    pass


# =============================================================================
# Audio Errors
# =============================================================================


class AudioError(PBStudioError):
    """
    Audio-bezogene Fehler.

    Base class for all audio-related errors.
    """

    pass


class AudioAnalysisError(AudioError):
    """
    Audio-Analyse Fehler.

    Raised when:
    - Beat detection fails
    - BPM calculation fails
    - Waveform extraction fails
    """

    pass


class InvalidAudioFormatError(AudioError):
    """
    Ungültiges Audio-Format.

    Raised when:
    - Audio file format is not supported
    - Audio file is corrupted
    - Cannot decode audio stream
    """

    pass


# =============================================================================
# Video Errors
# =============================================================================


class VideoError(PBStudioError):
    """
    Video-bezogene Fehler.

    Base class for all video-related errors.
    """

    pass


class ClipNotFoundError(VideoError):
    """
    Clip nicht gefunden.

    Raised when:
    - Clip ID not found in database
    - Video file path doesn't exist
    - Clip was deleted but still referenced
    """

    def __init__(self, clip_id: int | str = None, path: str = None, **kwargs):
        message = "Clip not found"
        if clip_id:
            message = f"Clip with ID {clip_id} not found"
        if path:
            message = f"Clip at path '{path}' not found"
        super().__init__(message, details={"clip_id": clip_id, "path": path, **kwargs})


class InvalidVideoFormatError(VideoError):
    """
    Ungültiges Video-Format.

    Raised when:
    - Video file format is not supported
    - Video codec is not supported
    - Video file is corrupted
    """

    pass


class RenderError(VideoError):
    """
    Fehler beim Video-Rendering.

    Base class for rendering-related errors.

    Raised when:
    - Segment extraction fails
    - Video concatenation fails
    - Audio merge fails
    - Output file cannot be written
    """

    pass


class FFmpegError(RenderError):
    """
    FFmpeg-spezifischer Fehler.

    Raised when:
    - FFmpeg command fails
    - FFmpeg not found in PATH
    - FFmpeg codec not available
    """

    def __init__(self, message: str = "", stderr: str = None, return_code: int = None, **kwargs):
        details = {"stderr": stderr, "return_code": return_code, **kwargs}
        super().__init__(message, details=details)


class RenderTimeoutError(RenderError):
    """
    Render-Timeout überschritten.

    Raised when:
    - FFmpeg operation exceeds timeout
    - Segment extraction takes too long
    - Video concatenation takes too long
    """

    def __init__(self, timeout_seconds: int = None, operation: str = None, **kwargs):
        message = "Render operation timed out"
        if operation:
            message = f"Render operation '{operation}' timed out"
        if timeout_seconds:
            message += f" after {timeout_seconds}s"
        super().__init__(
            message, details={"timeout": timeout_seconds, "operation": operation, **kwargs}
        )


# =============================================================================
# Pacing Errors
# =============================================================================


class PacingError(PBStudioError):
    """
    Pacing-bezogene Fehler.

    Base class for pacing engine errors.
    """

    pass


class BeatgridError(PacingError):
    """
    Beatgrid-Fehler.

    Raised when:
    - Beatgrid cannot be created
    - Beatgrid data is invalid
    - Beat times are inconsistent
    """

    pass


class ClipMatchError(PacingError):
    """
    Clip-Matching Fehler.

    Raised when:
    - No suitable clip found for trigger
    - FAISS/Qdrant index not available
    - Clip selection algorithm fails
    """

    pass


# =============================================================================
# Utility Functions
# =============================================================================


def wrap_exception(original: Exception, wrapper_class: type) -> PBStudioError:
    """
    Wrap a standard exception in a PBStudio exception.

    Args:
        original: The original exception
        wrapper_class: The PBStudio exception class to use

    Returns:
        Wrapped PBStudioError instance

    Example:
        try:
            do_something()
        except FileNotFoundError as e:
            raise wrap_exception(e, ClipNotFoundError) from e
    """
    return wrapper_class(
        message=str(original),
        details={
            "original_type": type(original).__name__,
            "original_args": original.args,
        },
    )


# =============================================================================
# ARCH-03 FIX: Einheitliche Error Handler
# =============================================================================

import logging
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

T = TypeVar("T")


def handle_errors(
    default_return: Any = None,
    log_level: int = logging.ERROR,
    reraise: bool = False,
    wrap_as: type = None,
) -> Callable:
    """
    Decorator für einheitliches Error Handling.

    ARCH-03 FIX: Vereinheitlicht Exception-Handling in der gesamten Codebase.

    Args:
        default_return: Rückgabewert bei Fehler (default: None)
        log_level: Log-Level für Fehler (default: ERROR)
        reraise: Ob Exception nach Logging erneut geworfen werden soll
        wrap_as: Optional: Wrapper-Klasse für Exception

    Example:
        @handle_errors(default_return=[], reraise=False)
        def load_clips():
            return database.query_clips()

        @handle_errors(wrap_as=RenderError, reraise=True)
        def render_video():
            # Fehler werden als RenderError gewrapped
            pass
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            logger = logging.getLogger(func.__module__)
            try:
                return func(*args, **kwargs)
            except PBStudioError:
                # Already a PBStudio error - log and optionally reraise
                logger.log(log_level, f"{func.__name__} failed", exc_info=True)
                if reraise:
                    raise
                return default_return
            except Exception as e:
                # External exception - wrap and handle
                logger.log(
                    log_level, f"{func.__name__} failed: {type(e).__name__}: {e}", exc_info=True
                )
                if wrap_as:
                    wrapped = wrap_exception(e, wrap_as)
                    if reraise:
                        raise wrapped from e
                elif reraise:
                    raise
                return default_return

        return wrapper

    return decorator


def safe_cleanup(func: Callable) -> Callable:
    """
    Decorator für sichere Cleanup-Operationen.

    Fängt alle Exceptions ab und loggt sie, wirft aber nie.
    Ideal für __del__, close(), cleanup() Methoden.

    Example:
        @safe_cleanup
        def cleanup(self):
            self.close_connections()
            self.delete_temp_files()
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Cleanup in {func.__name__} failed (ignored): {e}")
            return None

    return wrapper
