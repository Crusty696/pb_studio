"""
Keyframe String Generator
-------------------------
Generates mathematical animation strings for external tools (like Deforum, After Effects expressions).
Based on beat data and intensity curves.
"""

import math
from typing import Dict, List, Tuple


class KeyframeGenerator:
    """
    Generates keyframe strings for various parameters (Zoom, Translation, Rotation).
    """

    @staticmethod
    def _format_keyframe_dict(keyframes: Dict[int, float]) -> str:
        """Helper to format dictionary of frames into a Deforum string."""
        # Sort by frame number
        sorted_frames = sorted(keyframes.keys())
        return ", ".join([f"{f}: ({keyframes[f]:.3f})" for f in sorted_frames])

    @staticmethod
    def generate_zoom_curve(beats: list[float], intensity: float = 1.0, fps: int = 30) -> str:
        """
        Generates a Deforum-style math string for zoom.
        Example: "0: (1.0), 30: (1.05), ..."
        """
        if not beats:
            return "0: (1.0)"

        # Use dict to ensure unique frame indices
        kf_dict = {0: 1.0}

        for beat_time in beats:
            frame = int(beat_time * fps)
            # Zoom hit on beat (e.g., 1.05)
            kf_dict[frame] = 1.0 + (0.05 * intensity)
            # Return to normal (1.0) after 5 frames or half-way to next beat
            return_frame = frame + 5
            kf_dict[return_frame] = 1.0

        return KeyframeGenerator._format_keyframe_dict(kf_dict)

    @staticmethod
    def generate_shake_curve(beats: list[float], intensity: float = 1.0, fps: int = 30) -> str:
        """
        Generates a shake effect string (Translation X).
        """
        if not beats:
            return "0: (0.0)"

        kf_dict = {0: 0.0}

        for beat_time in beats:
            frame = int(beat_time * fps)
            # Shake left/right pattern
            kf_dict[frame] = 10.0 * intensity
            kf_dict[frame + 2] = -10.0 * intensity
            kf_dict[frame + 4] = 0.0

        return KeyframeGenerator._format_keyframe_dict(kf_dict)

    @staticmethod
    def generate_raw_math_string() -> str:
        """
        Returns a pure math expression (sin/cos) for continuous motion.
        """
        return "1.0 + 0.1*sin(2*3.14*t/10)"  # Example breathing effect