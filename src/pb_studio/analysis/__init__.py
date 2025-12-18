"""
Video-Analyse Package fuer PB_studio

Dieses Package stellt umfassende Video-Analyse-Funktionen bereit:
- Farb-Analyse (Palette, Temperatur, Helligkeit)
- Bewegungs-Analyse (Motion, Kamera)
- Szenentyp-Erkennung (Portrait, Wide, etc.)
- Stimmungs-Analyse (Mood, Energy)
- Objekt-Erkennung (YOLO)
- Style-Analyse (Visual Aesthetics)
- Aehnlichkeitssuche (FAISS, pHash)
"""

from .availability_manager import AvailabilityManager
from .fingerprint import ContentFingerprint, PerceptualHash
from .status_manager import ANALYSIS_VERSIONS, AnalysisStatusManager, ClipAnalysisStatus
from .video_analyzer import VideoAnalyzer

__all__ = [
    "VideoAnalyzer",
    "ContentFingerprint",
    "PerceptualHash",
    "AnalysisStatusManager",
    "ClipAnalysisStatus",
    "ANALYSIS_VERSIONS",
    "AvailabilityManager",
]
