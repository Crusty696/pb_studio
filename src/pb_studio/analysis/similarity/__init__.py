"""
Similarity Search Module fuer PB_studio

Enthaelt:
- FAISSManager: Vektorbasierte Aehnlichkeitssuche
- SimilaritySearch: Haupt-API fuer Aehnlichkeitssuche
"""

from .faiss_manager import FAISSManager, SearchResult
from .similarity_search import SimilarityResult, SimilaritySearch

__all__ = [
    "FAISSManager",
    "SearchResult",
    "SimilaritySearch",
    "SimilarityResult",
]
