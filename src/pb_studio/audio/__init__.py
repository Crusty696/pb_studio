"""
Audio-Modul fuer PB_studio

Komponenten:
- AudioAnalyzer: BPM, Beatgrid, Song-Struktur-Analyse (Original)
- AudioAnalyzerOptimized: Optimierte Version mit 60-70% Performance-Gewinn
"""

from .audio_analyzer import AudioAnalyzer

# AudioAnalyzerOptimized has been merged into AudioAnalyzer (optimized version is now default)

__all__ = ["AudioAnalyzer"]  # Now using optimized version as default
