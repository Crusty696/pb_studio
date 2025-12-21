"""
Keyframe String Generator
-------------------------
Generates mathematical animation strings for external tools (like Deforum, After Effects expressions).
Based on beat data and intensity curves.
"""

import math
from typing import List, Tuple


class KeyframeGenerator:
    """
    Generates keyframe strings for various parameters (Zoom, Translation, Rotation).
    """

    @staticmethod
    def generate_zoom_curve(beats: list[float], intensity: float = 1.0, fps: int = 30) -> str:
        """
        Generates a Deforum-style math string for zoom.
        Example: "0: (1.0), 30: (1.05), ..."
        
        Args:
            beats: List of beat timestamps in seconds.
            intensity: Multiplier for the effect.
            fps: Frames per second.
            
        Returns:
            Formatted string map.
        """
        if not beats:
            return "0: (1.0)"

        keyframes = []
        # Initial state
        keyframes.append("0: (1.0)")
        
        for beat_time in beats:
            frame = int(beat_time * fps)
            # Zoom hit on beat
            keyframes.append(f"{frame}: ({1.0 + (0.05 * intensity)})")
            # Return to normal shortly after
            keyframes.append(f"{frame + 5}: (1.0)")
            
        return ", ".join(keyframes)

    @staticmethod
    def generate_shake_curve(beats: list[float], intensity: float = 1.0, fps: int = 30) -> str:
        """
        Generates a shake effect string (Translation X).
        """
        if not beats:
            return "0: (0)"
            
        keyframes = []
        keyframes.append("0: (0)")
        
        for beat_time in beats:
            frame = int(beat_time * fps)
            # Shake left
            keyframes.append(f"{frame}: ({10 * intensity})")
            # Shake right (next frame)
            keyframes.append(f"{frame + 2}: ({-10 * intensity})")
            # Center
            keyframes.append(f"{frame + 4}: (0)")
            
        return ", ".join(keyframes)
        
    @staticmethod
    def generate_raw_math_string() -> str:
        """
        Returns a pure math expression (sin/cos) for continuous motion.
        """
        return "1.0 + 0.1*sin(2*3.14*t/10)" # Example breathing effect
