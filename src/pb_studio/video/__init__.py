"""
Video-Modul fuer PB_studio

Komponenten:
- VideoManager: FFmpeg-basiertes Video-Management (Metadaten, Thumbnails, Import)
- VideoAnalyzer: PySceneDetect-basierte Scene Detection
"""

from .video_analyzer import VideoAnalyzer
from .video_manager import VideoManager

__all__ = ["VideoManager", "VideoAnalyzer"]
