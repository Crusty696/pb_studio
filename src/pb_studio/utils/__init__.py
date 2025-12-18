"""Utility-Module für PB_studio."""

from .logger import get_logger, setup_logging
from .memory_pool import (
    AudioBufferPool,
    FrameBufferPool,
    get_global_audio_pool,
    get_global_frame_pool,
)
from .parallel import (
    ParallelProcessor,
    get_optimal_worker_count,
    parallel_analyze_clips,
    parallel_extract_segments,
    parallel_map,
    parallel_map_unordered,
)
from .subprocess_utils import patch_ffmpeg_subprocess, popen_hidden, run_hidden
from .video_utils import (
    VideoCapture,
    extract_frame_safe,
    get_video_info_safe,
    open_video,
)

__all__ = [
    "get_logger",
    "setup_logging",
    # Subprocess utilities (Windows fix)
    "run_hidden",
    "popen_hidden",
    "patch_ffmpeg_subprocess",
    "FrameBufferPool",
    "AudioBufferPool",
    "get_global_frame_pool",
    "get_global_audio_pool",
    "ParallelProcessor",
    "parallel_map",
    "parallel_map_unordered",
    "parallel_extract_segments",
    "parallel_analyze_clips",
    "get_optimal_worker_count",
    # Video utilities (PERF-02 Fix)
    "VideoCapture",
    "open_video",
    "get_video_info_safe",
    "extract_frame_safe",
]
