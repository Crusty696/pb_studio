"""
Subprocess utilities for PB_studio.

Provides Windows-compatible subprocess functions that hide console windows.
"""

import subprocess
import sys
from typing import Any


def get_subprocess_flags() -> int:
    """
    Get creation flags for subprocess on Windows.

    Returns:
        CREATE_NO_WINDOW flag on Windows, 0 on other platforms
    """
    if sys.platform == "win32":
        return subprocess.CREATE_NO_WINDOW
    return 0


def run_hidden(
    args: str | list[str],
    *,
    capture_output: bool = False,
    text: bool = False,
    timeout: float | None = None,
    check: bool = False,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
    shell: bool = False,
    cwd: str | None = None,
    env: dict | None = None,
    **kwargs,
) -> subprocess.CompletedProcess:
    """
    Run subprocess with hidden window on Windows.

    This is a drop-in replacement for subprocess.run() that automatically
    adds CREATE_NO_WINDOW flag on Windows to prevent console windows from
    appearing during parallel processing.

    CRITICAL-08 FIX: Args must be a list when shell=False for security

    Args:
        args: Command to run (list when shell=False, string when shell=True)
        capture_output: Capture stdout and stderr
        text: Decode output as text
        timeout: Timeout in seconds
        check: Raise exception on non-zero exit code
        stdin, stdout, stderr: Standard streams
        shell: Use shell
        cwd: Working directory
        env: Environment variables
        **kwargs: Additional arguments passed to subprocess.run

    Returns:
        CompletedProcess instance

    Example:
        # Instead of:
        subprocess.run(['ffmpeg', '-version'])

        # Use:
        run_hidden(['ffmpeg', '-version'])
    """
    # CRITICAL-08 FIX: Validate args type matches shell parameter
    if not shell and isinstance(args, str):
        raise ValueError(
            "args must be a list when shell=False (for security). "
            "Either pass args as list: ['cmd', 'arg1', 'arg2'] "
            "or set shell=True (only if command is trusted)"
        )

    # Add CREATE_NO_WINDOW flag on Windows
    if sys.platform == "win32":
        kwargs["creationflags"] = kwargs.get("creationflags", 0) | subprocess.CREATE_NO_WINDOW

    return subprocess.run(
        args,
        capture_output=capture_output,
        text=text,
        timeout=timeout,
        check=check,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        shell=shell,
        cwd=cwd,
        env=env,
        **kwargs,
    )


def popen_hidden(
    args: str | list[str],
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
    shell: bool = False,
    cwd: str | None = None,
    env: dict | None = None,
    **kwargs,
) -> subprocess.Popen:
    """
    Create subprocess.Popen with hidden window on Windows.

    This is a drop-in replacement for subprocess.Popen() that automatically
    adds CREATE_NO_WINDOW flag on Windows.

    Args:
        args: Command to run
        stdin, stdout, stderr: Standard streams
        shell: Use shell
        cwd: Working directory
        env: Environment variables
        **kwargs: Additional arguments passed to Popen

    Returns:
        Popen instance

    Important:
        MEDIUM-09 FIX: Caller is responsible for cleanup! Always call:
        - process.wait() or process.communicate() to avoid zombies
        - process.terminate() or process.kill() if needed
        - Use context manager or try/finally to ensure cleanup

    Example:
        >>> proc = popen_hidden(['ffmpeg', '-version'], stdout=subprocess.PIPE)
        >>> try:
        ...     output, _ = proc.communicate(timeout=5)
        ... finally:
        ...     if proc.poll() is None:
        ...         proc.kill()
    """
    # Add CREATE_NO_WINDOW flag on Windows
    if sys.platform == "win32":
        kwargs["creationflags"] = kwargs.get("creationflags", 0) | subprocess.CREATE_NO_WINDOW

    return subprocess.Popen(
        args, stdin=stdin, stdout=stdout, stderr=stderr, shell=shell, cwd=cwd, env=env, **kwargs
    )


# Monkey-patch ffmpeg-python to use hidden windows
def patch_ffmpeg_subprocess():
    """
    Patch ffmpeg-python library to hide console windows on Windows.

    This should be called early in application startup before any
    ffmpeg operations are performed.
    """
    if sys.platform != "win32":
        return

    try:
        import ffmpeg._run

        _original_run = ffmpeg._run.run

        def _patched_run(
            stream_spec,
            cmd="ffmpeg",
            capture_stdout=False,
            capture_stderr=False,
            input=None,
            quiet=False,
            pipe_stdin=False,
            pipe_stdout=False,
            pipe_stderr=False,
            overwrite_output=False,
        ):
            """Patched ffmpeg run that hides console windows on Windows."""
            original_popen = subprocess.Popen

            def patched_popen(*args, **kwargs):
                if "creationflags" not in kwargs:
                    kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
                return original_popen(*args, **kwargs)

            subprocess.Popen = patched_popen
            try:
                return _original_run(
                    stream_spec,
                    cmd,
                    capture_stdout,
                    capture_stderr,
                    input,
                    quiet,
                    pipe_stdin,
                    pipe_stdout,
                    pipe_stderr,
                    overwrite_output,
                )
            finally:
                subprocess.Popen = original_popen

        ffmpeg._run.run = _patched_run

    except ImportError:
        pass  # ffmpeg-python not installed


# HIGH-15 FIX: Auto-patch on import removed to prevent side effects
# Callers should explicitly call patch_ffmpeg_subprocess() if needed
# patch_ffmpeg_subprocess()
