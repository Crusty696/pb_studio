"""
Core module for PB_Studio.

Contains fundamental components like configuration, exceptions, and utilities.
"""

from .exceptions import (
    AudioAnalysisError,
    # Audio
    AudioError,
    BeatgridError,
    ClipMatchError,
    ClipNotFoundError,
    # Configuration
    ConfigurationError,
    # Database
    DatabaseError,
    FFmpegError,
    InvalidAudioFormatError,
    InvalidVideoFormatError,
    # Pacing
    PacingError,
    # Base
    PBStudioError,
    RenderError,
    RenderTimeoutError,
    SessionError,
    # Video
    VideoError,
    # Utility
    wrap_exception,
)

__all__ = [
    # Base
    "PBStudioError",
    # Configuration
    "ConfigurationError",
    # Database
    "DatabaseError",
    "SessionError",
    # Audio
    "AudioError",
    "AudioAnalysisError",
    "InvalidAudioFormatError",
    # Video
    "VideoError",
    "ClipNotFoundError",
    "InvalidVideoFormatError",
    "RenderError",
    "FFmpegError",
    "RenderTimeoutError",
    # Pacing
    "PacingError",
    "BeatgridError",
    "ClipMatchError",
    # Utility
    "wrap_exception",
]
