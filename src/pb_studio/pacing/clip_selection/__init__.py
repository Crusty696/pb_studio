"""
Clip Selection Strategies Package

Implements Strategy Pattern for flexible clip selection algorithms.
Reduces cyclomatic complexity of AdvancedPacingEngine from CC=54 to CC<10.
"""

from .base_strategy import ClipSelectionResult, ClipSelectionStrategy
from .diversity_manager import DiversityManager
from .faiss_strategy import FAISSStrategy
from .roundrobin_strategy import RoundRobinStrategy
from .smart_strategy import SmartStrategy

__all__ = [
    "ClipSelectionStrategy",
    "ClipSelectionResult",
    "DiversityManager",
    "FAISSStrategy",
    "SmartStrategy",
    "RoundRobinStrategy",
]
