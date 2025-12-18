"""
Video Beatgrid - Motion Peak Detection for Video Clips.

Detects "beats" (motion peaks) in video clips similar to audio beat detection.
These motion peaks can be used for intelligent clip synchronization with music.

Features:
- Frame-by-frame motion analysis using Optical Flow
- Peak detection for motion intensity
- Exportable beatgrid format compatible with audio beatgrid
- Database storage support
- Cache for performance

Usage:
    detector = VideoBeatgridDetector()
    beatgrid = detector.detect_video_beats(video_path)
    print(f"Found {len(beatgrid.beat_times)} motion peaks at BPM-equivalent: {beatgrid.estimated_bpm}")

Author: PB_studio Development Team
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from scipy.signal import find_peaks, savgol_filter

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VideoBeatgrid:
    """
    Video Beatgrid containing motion peak times.

    Attributes:
        video_path: Path to the analyzed video
        beat_times: List of motion peak times in seconds
        motion_scores: Motion intensity at each beat (0-1)
        estimated_bpm: Estimated "beats per minute" based on peak frequency
        duration: Video duration in seconds
        fps: Video frames per second
        frame_count: Total number of frames
    """

    video_path: str
    beat_times: list[float] = field(default_factory=list)
    motion_scores: list[float] = field(default_factory=list)
    estimated_bpm: float = 0.0
    duration: float = 0.0
    fps: float = 0.0
    frame_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "video_path": self.video_path,
            "beat_times": self.beat_times,
            "motion_scores": self.motion_scores,
            "estimated_bpm": self.estimated_bpm,
            "duration": self.duration,
            "fps": self.fps,
            "frame_count": self.frame_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoBeatgrid":
        """Create from dictionary."""
        return cls(
            video_path=data.get("video_path", ""),
            beat_times=data.get("beat_times", []),
            motion_scores=data.get("motion_scores", []),
            estimated_bpm=data.get("estimated_bpm", 0.0),
            duration=data.get("duration", 0.0),
            fps=data.get("fps", 0.0),
            frame_count=data.get("frame_count", 0),
        )

    def get_beat_at_time(self, time: float, tolerance: float = 0.1) -> int | None:
        """
        Find beat index nearest to given time.

        Args:
            time: Time in seconds
            tolerance: Maximum distance in seconds

        Returns:
            Beat index or None if no beat within tolerance
        """
        for i, beat_time in enumerate(self.beat_times):
            if abs(beat_time - time) <= tolerance:
                return i
        return None

    def get_beats_in_range(self, start: float, end: float) -> list[float]:
        """Get all beat times within a time range."""
        return [t for t in self.beat_times if start <= t <= end]


class VideoBeatgridDetector:
    """
    Detects motion "beats" in video clips using optical flow analysis.

    Algorithm:
    1. Extract frames from video
    2. Calculate optical flow between consecutive frames
    3. Compute motion magnitude for each frame pair
    4. Smooth motion curve with Savitzky-Golay filter
    5. Detect peaks in motion curve
    6. Filter peaks by prominence and distance
    7. Calculate estimated BPM from peak frequency
    """

    def __init__(
        self,
        min_peak_distance: float = 0.2,  # Minimum seconds between peaks
        peak_prominence: float = 0.1,  # Minimum prominence for peaks
        smoothing_window: int = 7,  # Savitzky-Golay window size
        frame_skip: int = 2,  # Skip frames for performance
        resize_factor: float = 0.5,  # Resize frames for performance
    ):
        """
        Initialize detector.

        Args:
            min_peak_distance: Minimum time between detected beats (seconds)
            peak_prominence: Minimum prominence for peak detection (0-1)
            smoothing_window: Window size for motion curve smoothing
            frame_skip: Analyze every N-th frame (1=all, 2=every other)
            resize_factor: Resize frames to this factor (0.5 = half size)
        """
        self.min_peak_distance = min_peak_distance
        self.peak_prominence = peak_prominence
        self.smoothing_window = smoothing_window
        self.frame_skip = max(1, frame_skip)
        self.resize_factor = max(0.1, min(1.0, resize_factor))

        logger.info(
            f"VideoBeatgridDetector initialized: min_distance={min_peak_distance}s, "
            f"prominence={peak_prominence}, frame_skip={frame_skip}"
        )

    def detect_video_beats(self, video_path: str) -> VideoBeatgrid | None:
        """
        Detect motion beats in a video file.

        Args:
            video_path: Path to video file

        Returns:
            VideoBeatgrid object or None if analysis fails
        """
        path = Path(video_path)
        if not path.exists():
            logger.error(f"Video not found: {video_path}")
            return None

        logger.info(f"Analyzing video beats: {path.name}")

        try:
            # Open video
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                cap.release()  # PERF-02 FIX: Release even on failed open
                logger.error(f"Failed to open video: {video_path}")
                return None

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0

            if frame_count < 10:
                logger.warning(f"Video too short: {frame_count} frames")
                cap.release()
                return None

            # Analyze motion
            motion_curve = self._compute_motion_curve(cap, fps)
            cap.release()

            if len(motion_curve) < 3:
                logger.warning("Motion curve too short for peak detection")
                return None

            # Smooth motion curve
            if len(motion_curve) >= self.smoothing_window:
                motion_curve = savgol_filter(motion_curve, self.smoothing_window, polyorder=2)

            # Detect peaks
            effective_fps = fps / self.frame_skip
            min_distance_frames = int(self.min_peak_distance * effective_fps)
            min_distance_frames = max(1, min_distance_frames)

            peaks, properties = find_peaks(
                motion_curve, distance=min_distance_frames, prominence=self.peak_prominence
            )

            # Convert peak indices to times
            beat_times = []
            motion_scores = []

            for peak_idx in peaks:
                time = (peak_idx * self.frame_skip) / fps
                beat_times.append(float(time))
                motion_scores.append(float(motion_curve[peak_idx]))

            # Calculate estimated BPM
            estimated_bpm = 0.0
            if len(beat_times) >= 2:
                intervals = np.diff(beat_times)
                avg_interval = np.mean(intervals)
                if avg_interval > 0:
                    estimated_bpm = 60.0 / avg_interval

            beatgrid = VideoBeatgrid(
                video_path=str(path),
                beat_times=beat_times,
                motion_scores=motion_scores,
                estimated_bpm=estimated_bpm,
                duration=duration,
                fps=fps,
                frame_count=frame_count,
            )

            logger.info(
                f"Detected {len(beat_times)} motion beats in {path.name}, "
                f"estimated BPM: {estimated_bpm:.1f}"
            )

            return beatgrid

        except Exception as e:
            logger.error(f"Video beat detection failed: {e}", exc_info=True)
            return None

    def _compute_motion_curve(self, cap: cv2.VideoCapture, fps: float) -> np.ndarray:
        """
        Compute motion intensity curve using optical flow.

        Args:
            cap: OpenCV VideoCapture object
            fps: Video FPS for timing

        Returns:
            Array of motion magnitudes
        """
        motion_values = []
        prev_gray = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames for performance
            if frame_idx % self.frame_skip != 0:
                frame_idx += 1
                continue

            # Resize for performance
            if self.resize_factor < 1.0:
                new_size = (
                    int(frame.shape[1] * self.resize_factor),
                    int(frame.shape[0] * self.resize_factor),
                )
                frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                # Calculate optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray,
                    gray,
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0,
                )

                # Compute flow magnitude
                magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                avg_motion = np.mean(magnitude)

                # Normalize to 0-1 range (rough normalization)
                normalized = min(1.0, avg_motion / 10.0)
                motion_values.append(normalized)

            prev_gray = gray
            frame_idx += 1

        return np.array(motion_values)

    def batch_detect(
        self, video_paths: list[str], progress_callback: Callable | None = None
    ) -> dict[str, VideoBeatgrid]:
        """
        Detect beats in multiple videos.

        Args:
            video_paths: List of video file paths
            progress_callback: Optional callback(current, total, path)

        Returns:
            Dict mapping video paths to VideoBeatgrid objects
        """
        results = {}
        total = len(video_paths)

        for i, path in enumerate(video_paths):
            if progress_callback:
                progress_callback(i, total, path)

            beatgrid = self.detect_video_beats(path)
            if beatgrid:
                results[path] = beatgrid

        logger.info(f"Batch detection complete: {len(results)}/{total} videos analyzed")
        return results


def save_beatgrid_to_json(beatgrid: VideoBeatgrid, output_path: str) -> bool:
    """
    Save VideoBeatgrid to JSON file.

    Args:
        beatgrid: VideoBeatgrid to save
        output_path: Output file path

    Returns:
        True if successful
    """
    try:
        with open(output_path, "w") as f:
            json.dump(beatgrid.to_dict(), f, indent=2)
        logger.info(f"Beatgrid saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save beatgrid: {e}")
        return False


def load_beatgrid_from_json(input_path: str) -> VideoBeatgrid | None:
    """
    Load VideoBeatgrid from JSON file.

    Args:
        input_path: Input file path

    Returns:
        VideoBeatgrid or None if loading fails
    """
    try:
        with open(input_path) as f:
            data = json.load(f)
        return VideoBeatgrid.from_dict(data)
    except Exception as e:
        logger.error(f"Failed to load beatgrid: {e}")
        return None
