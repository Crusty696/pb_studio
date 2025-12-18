"""
ClipMatcher Protocol für PB_studio

ARCH-07 FIX: Definiert ein einheitliches Interface für alle ClipMatcher-Implementierungen.

Dieses Protocol ermöglicht:
- Austauschbare Backends (FAISS, Qdrant, etc.)
- Bessere Testbarkeit durch Mock-Implementierungen
- Type Safety mit Protocol statt ABC

Usage:
    def process_clips(matcher: ClipMatcherProtocol, clips: list) -> None:
        matcher.build_index(clips)
        result = matcher.find_best_clip(motion_score=0.8, energy=0.7)
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ClipMatcherProtocol(Protocol):
    """
    Protocol für ClipMatcher-Implementierungen.

    Alle ClipMatcher (FAISS, Qdrant, etc.) müssen dieses Interface implementieren.
    Verwendet Protocol statt ABC für strukturelles Subtyping.
    """

    def build_index(self, clips: list[dict[str, Any]]) -> None:
        """
        Baut den Suchindex aus Clip-Daten.

        Args:
            clips: Liste von Clip-Dicts mit Analysis-Daten
                   Format: [{'id': int, 'analysis': {...}, 'file_path': str}, ...]

        Raises:
            ValueError: Wenn keine gültigen Embeddings extrahiert werden konnten
        """
        ...

    def find_best_clip(
        self,
        target_motion_score: float,
        target_energy: float,
        target_motion_type: str | None = None,
        target_moods: list[str] | None = None,
        exclude_clip_ids: set | list | None = None,  # PERF: Accept set for O(1) lookup
        k: int = 5,
        previous_clip_id: int | None = None,  # CONTINUITY: For visual flow
        continuity_weight: float = 0.4,  # Balance between target match and visual continuity
    ) -> tuple[int, str, float] | None:
        """
        Findet den besten passenden Clip.

        Args:
            target_motion_score: Ziel-Motion-Score (0.0 - 1.0)
            target_energy: Ziel-Energy-Level (0.0 - 1.0)
            target_motion_type: Optionaler Motion-Typ ('STATIC', 'SLOW', 'MEDIUM', 'FAST')
            target_moods: Optionale Liste von Ziel-Stimmungen
            exclude_clip_ids: IDs von Clips, die ausgeschlossen werden sollen (set preferred for performance)
            k: Anzahl der zu durchsuchenden Kandidaten
            previous_clip_id: ID des vorherigen Clips für visuelle Kontinuität ("roter Faden")
            continuity_weight: Gewichtung der visuellen Ähnlichkeit (0=keine, 1=maximal)

        Returns:
            Tuple von (clip_id, file_path, distance) oder None wenn kein Clip gefunden
        """
        ...

    def find_similar_clips(
        self, reference_clip_id: int, k: int = 5, exclude_ids: list[int] | None = None
    ) -> list[tuple[int, str, float]]:
        """
        Findet ähnliche Clips zu einem Referenz-Clip.

        Args:
            reference_clip_id: ID des Referenz-Clips
            k: Anzahl der zurückzugebenden Ergebnisse
            exclude_ids: IDs von Clips, die ausgeschlossen werden sollen

        Returns:
            Liste von Tuples (clip_id, file_path, distance), sortiert nach Ähnlichkeit
        """
        ...

    def get_clip_count(self) -> int:
        """
        Gibt die Anzahl der indizierten Clips zurück.

        Returns:
            Anzahl der Clips im Index
        """
        ...

    def clear_index(self) -> None:
        """
        Löscht den aktuellen Index.

        Nach diesem Aufruf muss build_index() erneut aufgerufen werden.
        """
        ...


class ClipMatcherStats(Protocol):
    """
    Optional: Erweiterte Statistik-Methoden für ClipMatcher.

    Nicht alle Implementierungen müssen diese Methoden unterstützen.
    """

    def get_index_stats(self) -> dict[str, Any]:
        """
        Gibt Statistiken über den Index zurück.

        Returns:
            Dict mit Statistiken wie:
            - 'clip_count': Anzahl der Clips
            - 'dimension': Embedding-Dimension
            - 'index_type': Typ des Index
            - 'memory_usage_mb': Geschätzter Speicherverbrauch
        """
        ...

    def get_search_stats(self) -> dict[str, Any]:
        """
        Gibt Such-Statistiken zurück.

        Returns:
            Dict mit Statistiken wie:
            - 'total_searches': Gesamtzahl der Suchen
            - 'avg_search_time_ms': Durchschnittliche Suchzeit
            - 'cache_hit_rate': Cache-Trefferquote
        """
        ...


def validate_clip_matcher(matcher: Any) -> bool:
    """
    Prüft ob ein Objekt das ClipMatcherProtocol implementiert.

    Args:
        matcher: Zu prüfendes Objekt

    Returns:
        True wenn das Objekt das Protocol implementiert
    """
    return isinstance(matcher, ClipMatcherProtocol)
