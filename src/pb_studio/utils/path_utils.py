"""
Path Utilities für PB_studio

Zentralisierte Path-Validierung und -Normalisierung.
Eliminiert Code-Duplikation in video_renderer.py, motion_analyzer.py, etc.

Security:
- Path Traversal Protection (CWE-22)
- Directory Whitelist enforcement
- Safe path resolution

Author: PB_studio Development Team
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# =============================================================================
# SECURITY: Allowed Base Directories (Path Traversal Protection)
# =============================================================================
# Define allowed base directories for file access
# Any file outside these directories will be rejected
ALLOWED_BASE_DIRS = [
    Path.cwd(),  # Project directory
    Path.home() / "Videos",  # User Videos directory
    Path.home() / "Documents",  # User Documents directory
    Path.home() / "Downloads",  # User Downloads directory
    Path.home() / "Music",  # User Music directory (für Audio-Dateien)
    Path.home() / "Desktop",  # User Desktop directory
    # Add more allowed directories as needed
]


def validate_file_path(
    path: str | Path,
    must_exist: bool = True,
    extensions: list[str] | None = None,
    allowed_base_dirs: list[Path] | None = None,
    enforce_whitelist: bool = True,
) -> Path:
    """
    Validates and normalizes file path with security checks.

    Security Features:
    - Path Traversal Protection (CWE-22)
    - Directory Whitelist enforcement
    - Safe path resolution

    Args:
        path: File path to validate
        must_exist: If True, raises if file doesn't exist
        extensions: Allowed extensions (e.g., ['.mp4', '.avi'])
        allowed_base_dirs: List of allowed base directories
                          (defaults to ALLOWED_BASE_DIRS)
        enforce_whitelist: If True, enforces directory whitelist
                          (disable for testing only)

    Returns:
        Normalized Path object

    Raises:
        ValueError: If path is invalid, outside allowed dirs, or extension not allowed
        FileNotFoundError: If must_exist=True and file doesn't exist

    Example:
        >>> path = validate_file_path("video.mp4", extensions=['.mp4', '.avi'])
        >>> print(path)
        C:/videos/video.mp4
    """
    if not path or not isinstance(path, (str, Path)):
        raise ValueError(f"Invalid path type: {type(path)}")

    # Normalize and resolve path
    path = Path(path).resolve()

    # SECURITY: Check if path is within allowed directories
    if enforce_whitelist:
        if allowed_base_dirs is None:
            allowed_base_dirs = ALLOWED_BASE_DIRS

        is_allowed = False
        for base_dir in allowed_base_dirs:
            try:
                # Check if path is relative to allowed base directory
                path.relative_to(base_dir.resolve())
                is_allowed = True
                logger.debug(f"Path within allowed directory: {base_dir}")
                break
            except ValueError:
                # MEDIUM-07 FIX: Add debug logging for rejections
                logger.debug(f"Path '{path}' not within allowed directory '{base_dir}'")
                continue

        if not is_allowed:
            allowed_dirs_str = ", ".join(str(d) for d in allowed_base_dirs)
            raise ValueError(
                f"Security: Path '{path}' is outside allowed directories. "
                f"Allowed: {allowed_dirs_str}"
            )

    # Check existence
    if must_exist and not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Check extension
    if extensions:
        extensions_lower = [ext.lower() for ext in extensions]
        if path.suffix.lower() not in extensions_lower:
            raise ValueError(f"Invalid extension {path.suffix}, allowed: {extensions}")

    logger.debug(f"Path validated: {path}")
    return path


def validate_directory_path(
    path: str | Path,
    create: bool = False,
    allowed_base_dirs: list[Path] | None = None,
    enforce_whitelist: bool = True,
) -> Path:
    """
    Validates and normalizes directory path with security checks.

    Security Features:
    - Path Traversal Protection (CWE-22)
    - Directory Whitelist enforcement

    Args:
        path: Directory path to validate
        create: If True, creates directory if it doesn't exist
        allowed_base_dirs: List of allowed base directories
                          (defaults to ALLOWED_BASE_DIRS)
        enforce_whitelist: If True, enforces directory whitelist

    Returns:
        Normalized Path object

    Raises:
        ValueError: If path is invalid, outside allowed dirs, or not a directory
        FileNotFoundError: If directory doesn't exist and create=False

    Example:
        >>> path = validate_directory_path("output", create=True)
        >>> path.exists()
        True
    """
    if not path or not isinstance(path, (str, Path)):
        raise ValueError(f"Invalid path type: {type(path)}")

    # Normalize and resolve path
    path = Path(path).resolve()

    # SECURITY: Check if path is within allowed directories
    if enforce_whitelist:
        if allowed_base_dirs is None:
            allowed_base_dirs = ALLOWED_BASE_DIRS

        is_allowed = False
        for base_dir in allowed_base_dirs:
            try:
                # Check if path is relative to allowed base directory
                path.relative_to(base_dir.resolve())
                is_allowed = True
                logger.debug(f"Directory within allowed directory: {base_dir}")
                break
            except ValueError:
                continue

        if not is_allowed:
            allowed_dirs_str = ", ".join(str(d) for d in allowed_base_dirs)
            raise ValueError(
                f"Security: Path '{path}' is outside allowed directories. "
                f"Allowed: {allowed_dirs_str}"
            )

    # Create or validate existence
    if create:
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory created: {path}")
    elif not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    elif not path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    logger.debug(f"Directory validated: {path}")
    return path


def ensure_extension(path: Path, extension: str) -> Path:
    """
    Ensures path has correct extension.

    Args:
        path: Path to modify
        extension: Extension to ensure (with or without dot)

    Returns:
        Path with correct extension

    Example:
        >>> path = Path("video")
        >>> result = ensure_extension(path, ".mp4")
        >>> print(result)
        video.mp4
    """
    if not extension.startswith("."):
        extension = f".{extension}"

    if path.suffix.lower() != extension.lower():
        path = path.with_suffix(extension)
        logger.debug(f"Extension ensured: {path}")

    return path


def get_relative_path(path: Path, base: Path) -> Path:
    """
    Get relative path from base.

    Args:
        path: Absolute path
        base: Base directory

    Returns:
        Relative path

    Example:
        >>> path = Path("/project/src/file.py")
        >>> base = Path("/project")
        >>> result = get_relative_path(path, base)
        >>> print(result)
        src/file.py
    """
    try:
        return path.relative_to(base)
    except ValueError:
        # Paths are not relative
        return path


def to_relative_path(file_path: str | Path, base_dir: str | Path = None) -> str:
    """
    Convert file path to relative path (for database storage).

    Ensures portability: If project folder is moved, clips still work.

    Args:
        file_path: Absolute or relative file path
        base_dir: Base directory (defaults to project root)

    Returns:
        Relative path as string (forward slashes for portability)

    Raises:
        ValueError: If file_path is not within base_dir

    Example:
        >>> file_path = "C:/Project/pb_studio/test_clips/clip_fast.mp4"
        >>> rel_path = to_relative_path(file_path)
        >>> print(rel_path)
        test_clips/clip_fast.mp4
    """
    if base_dir is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir).resolve()

    file_path = Path(file_path).resolve()

    try:
        rel_path = file_path.relative_to(base_dir)
        # Return with forward slashes for cross-platform compatibility
        return rel_path.as_posix()
    except ValueError as e:
        # MEDIUM-06 FIX: Raise exception instead of silent failure with absolute path
        logger.error(f"Path {file_path} is not within base directory {base_dir}")
        raise ValueError(
            f"Cannot convert path to relative: '{file_path}' is not within base directory '{base_dir}'. "
            f"Please ensure all media files are within the project directory."
        ) from e


def validate_ffmpeg_path(path: str | Path) -> Path:
    """
    Validate path for safe use with FFmpeg (Command Injection Protection).

    Security Features:
    - Filters dangerous shell metacharacters
    - Ensures absolute path
    - Protects against command injection (CWE-78)

    Args:
        path: File path to validate for FFmpeg usage

    Returns:
        Validated absolute Path object

    Raises:
        ValueError: If path contains dangerous characters or is not absolute

    Example:
        >>> path = validate_ffmpeg_path("C:/videos/video.mp4")
        >>> # Safe to use in FFmpeg command

    Dangerous paths (will raise ValueError):
        >>> validate_ffmpeg_path("video.mp4; rm -rf /")
        ValueError: Path contains dangerous character: ;
    """
    # Normalize path
    path = Path(path).resolve()

    # SECURITY: Check for dangerous shell metacharacters
    # Note: '&' is allowed in filenames - ffmpeg-python properly escapes arguments
    # Only block characters that are always dangerous in shell contexts
    dangerous_chars = [";", "|", "$", "`", "\n", "\r", "<", ">"]
    path_str = str(path)

    for char in dangerous_chars:
        if char in path_str:
            raise ValueError(
                f"Security: Path contains dangerous character '{char}'. "
                f"This could allow command injection."
            )

    # SECURITY: Ensure path is absolute (no relative path tricks)
    if not path.is_absolute():
        raise ValueError(f"Security: Path must be absolute, got: {path}")

    logger.debug(f"FFmpeg path validated: {path}")
    return path


def resolve_relative_path(file_path: str | Path, base_dir: str | Path = None) -> Path:
    """
    Convert relative path to absolute path (for file operations).

    Args:
        file_path: Relative or absolute file path
        base_dir: Base directory for relative paths (defaults to project root)

    Returns:
        Absolute Path object

    Raises:
        ValueError: If resolved path escapes base_dir (path traversal protection)

    Example:
        >>> rel_path = "test_clips/clip_fast.mp4"
        >>> abs_path = resolve_relative_path(rel_path)
        >>> print(abs_path)
        C:/Project/pb_studio/test_clips/clip_fast.mp4
    """
    if base_dir is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir).resolve()

    file_path = Path(file_path)

    # If already absolute, return as-is
    if file_path.is_absolute():
        return file_path.resolve()

    # Resolve relative to base
    abs_path = (base_dir / file_path).resolve()

    # MEDIUM-10 FIX: Validate resolved path is within base_dir (path traversal protection)
    try:
        abs_path.relative_to(base_dir)
    except ValueError:
        logger.error(f"Security: Resolved path '{abs_path}' escapes base directory '{base_dir}'")
        raise ValueError(
            f"Security: Path traversal detected. '{file_path}' resolves to '{abs_path}' "
            f"which is outside base directory '{base_dir}'"
        )

    if not abs_path.exists():
        logger.warning(f"Resolved path does not exist: {abs_path}")

    return abs_path


def get_ffmpeg_path() -> Path:
    """
    Get absolute path to ffmpeg executable.

    Priority:
    1. Bundled (PyInstaller _MEIPASS/bin)
    2. Local bin folder (project/bin)
    3. System PATH

    Returns:
        Path to ffmpeg executable

    Raises:
        FileNotFoundError: If ffmpeg not found
    """
    import os
    import shutil
    import sys

    exe_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"

    # 1. Bundled (PyInstaller)
    if hasattr(sys, "_MEIPASS"):
        bundled_path = Path(sys._MEIPASS) / "bin" / exe_name
        if bundled_path.exists():
            return bundled_path

    # 2. Local bin folder (dev environment)
    # Assumes src/pb_studio/utils/path_utils.py -> project_root/bin
    project_root = Path(__file__).parent.parent.parent.parent
    local_bin = project_root / "bin" / exe_name
    if local_bin.exists():
        return local_bin

    # 3. System PATH
    path_on_system = shutil.which("ffmpeg")
    if path_on_system:
        return Path(path_on_system)

    raise FileNotFoundError(
        "FFmpeg not found. Please install FFmpeg or place it in the 'bin' folder."
    )


def get_ffprobe_path() -> Path:
    """
    Get absolute path to ffprobe executable.

    Priority:
    1. Bundled (PyInstaller _MEIPASS/bin)
    2. Local bin folder (project/bin)
    3. System PATH

    Returns:
        Path to ffprobe executable

    Raises:
        FileNotFoundError: If ffprobe not found
    """
    import os
    import shutil
    import sys

    exe_name = "ffprobe.exe" if sys.platform == "win32" else "ffprobe"

    # 1. Bundled (PyInstaller)
    if hasattr(sys, "_MEIPASS"):
        bundled_path = Path(sys._MEIPASS) / "bin" / exe_name
        if bundled_path.exists():
            return bundled_path

    # 2. Local bin folder (dev environment)
    project_root = Path(__file__).parent.parent.parent.parent
    local_bin = project_root / "bin" / exe_name
    if local_bin.exists():
        return local_bin

    # 3. System PATH
    path_on_system = shutil.which("ffprobe")
    if path_on_system:
        return Path(path_on_system)

    raise FileNotFoundError(
        "FFprobe not found. Please install FFmpeg or place it in the 'bin' folder."
    )
