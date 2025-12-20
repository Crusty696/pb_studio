"""
Parallel Processing Utilities für PB_studio

OPTIMIZATION: Beschleunigt I/O und CPU-bound Operations durch Parallelisierung.

Features:
- Thread Pool für I/O Operations (File Reading, FFmpeg Calls)
- Process Pool für CPU-bound Tasks (Audio Analysis, Video Processing)
- Adaptive Worker Count basierend auf CPU Cores
- Progress Tracking für parallele Tasks

Performance Impact:
- 2-4x schnelleres Segment Extraction (Thread Pool)
- 3-8x schnellere Clip Analysis (Process Pool)
- Bessere CPU/GPU Auslastung
"""

import logging
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any

logger = logging.getLogger(__name__)


def get_optimal_worker_count(task_type: str = "cpu") -> int:
    """
    Ermittelt optimale Anzahl Worker basierend auf Task-Typ.

    Args:
        task_type: "cpu" für CPU-bound, "io" für I/O-bound Tasks

    Returns:
        Optimale Anzahl Worker
    """
    cpu_count = os.cpu_count() or 4

    if task_type == "io":
        # I/O-bound: Mehr Workers (2-3x CPU Cores)
        return min(cpu_count * 2, 32)  # Max 32 für Stability
    elif task_type == "cpu":
        # CPU-bound: 1 Worker pro Core (avoid oversubscription)
        return max(cpu_count - 1, 1)  # Leave 1 core free
    else:
        return cpu_count


class ParallelProcessor:
    """
    Utility Class für parallele Verarbeitung mit Thread/Process Pools.

    Usage:
        processor = ParallelProcessor(task_type="io")
        results = processor.map(my_function, items)
    """

    def __init__(
        self, task_type: str = "io", max_workers: int | None = None, use_processes: bool = False
    ):
        """
        Args:
            task_type: "io" oder "cpu" für auto worker count
            max_workers: Maximale Anzahl Workers (None = auto)
            use_processes: Process Pool statt Thread Pool (für CPU-bound)
        """
        self.task_type = task_type
        self.use_processes = use_processes

        if max_workers is None:
            self.max_workers = get_optimal_worker_count(task_type)
        else:
            self.max_workers = max_workers

        logger.info(
            f"ParallelProcessor initialized: type={task_type}, "
            f"workers={self.max_workers}, processes={use_processes}"
        )

    def map(
        self,
        func: Callable,
        items: Iterable,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[Any]:
        """
        Wendet Funktion parallel auf alle Items an.

        Args:
            func: Function to apply to each item
            items: Iterable of items to process
            progress_callback: Optional callback(completed, total)

        Returns:
            List of results (same order as input)
        """
        items_list = list(items)
        total = len(items_list)
        results = [None] * total

        Executor = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor

        # FIX: Explicit exception handling for better error recovery
        # The context manager ensures executor.shutdown(wait=True) is always called
        try:
            with Executor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_index = {executor.submit(func, item): i for i, item in enumerate(items_list)}

                # Collect results as they complete
                completed = 0
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        results[index] = result
                        completed += 1

                        if progress_callback:
                            progress_callback(completed, total)

                    except Exception as e:
                        logger.error(f"Error processing item {index}: {e}")
                        results[index] = None
        except Exception as e:
            # FIX: Log and propagate executor-level errors
            logger.error(f"ParallelProcessor executor error: {e}", exc_info=True)
            raise

        return results

    def map_unordered(
        self,
        func: Callable,
        items: Iterable,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[Any]:
        """
        Wie map(), aber Ergebnisse in completion order (schneller).

        Args:
            func: Function to apply
            items: Items to process
            progress_callback: Optional callback(completed, total)

        Returns:
            List of results (unordered)
        """
        items_list = list(items)
        total = len(items_list)
        results = []

        Executor = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor

        with Executor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(func, item) for item in items_list]

            completed = 0
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1

                    if progress_callback:
                        progress_callback(completed, total)

                except Exception as e:
                    logger.error(f"Error processing item: {e}")

        return results


def parallel_map(
    func: Callable,
    items: Iterable,
    max_workers: int | None = None,
    use_processes: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[Any]:
    """
    Convenience function für paralleles Mapping.

    Args:
        func: Function to apply to each item
        items: Items to process
        max_workers: Max workers (None = auto)
        use_processes: Use ProcessPool instead of ThreadPool
        progress_callback: Optional callback(completed, total)

    Returns:
        List of results (ordered)
    """
    processor = ParallelProcessor(
        task_type="cpu" if use_processes else "io",
        max_workers=max_workers,
        use_processes=use_processes,
    )
    return processor.map(func, items, progress_callback)


def parallel_map_unordered(
    func: Callable,
    items: Iterable,
    max_workers: int | None = None,
    use_processes: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[Any]:
    """
    Convenience function für paralleles Mapping (ungeordnet, schneller).

    Args:
        func: Function to apply
        items: Items to process
        max_workers: Max workers (None = auto)
        use_processes: Use ProcessPool instead of ThreadPool
        progress_callback: Optional callback(completed, total)

    Returns:
        List of results (unordered)
    """
    processor = ParallelProcessor(
        task_type="cpu" if use_processes else "io",
        max_workers=max_workers,
        use_processes=use_processes,
    )
    return processor.map_unordered(func, items, progress_callback)


# Convenience functions für häufige Use Cases


def parallel_extract_segments(
    extract_func: Callable,
    segment_specs: list,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list:
    """
    Paralleles Segment Extraction (I/O-bound).

    Args:
        extract_func: Function(spec) -> segment_path
        segment_specs: List of segment specifications
        progress_callback: Optional progress callback

    Returns:
        List of extracted segment paths
    """
    return parallel_map(
        extract_func,
        segment_specs,
        max_workers=get_optimal_worker_count("io"),
        use_processes=False,  # I/O-bound → Threads
        progress_callback=progress_callback,
    )


def parallel_analyze_clips(
    analyze_func: Callable,
    clip_paths: list,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list:
    """
    Parallele Clip Analysis (CPU-bound).

    Args:
        analyze_func: Function(clip_path) -> analysis_result
        clip_paths: List of clip paths
        progress_callback: Optional progress callback

    Returns:
        List of analysis results
    """
    return parallel_map(
        analyze_func,
        clip_paths,
        max_workers=get_optimal_worker_count("cpu"),
        use_processes=True,  # CPU-bound → Processes
        progress_callback=progress_callback,
    )


# =============================================================================
# M-02 FIX: Throttled Callback Helper (DRY principle)
# =============================================================================


def create_throttled_callback(
    callback: Callable[[int, str], None],
    scale_min: int = 0,
    scale_max: int = 100,
    min_change: int = 1,
) -> Callable[[float, str], None]:
    """
    Erstellt einen gedrosselten Callback für Progress-Updates.

    M-02 FIX: Verhindert Handle-Leaks durch zu häufige UI-Updates.
    Emittiert nur bei signifikanter Änderung (Standard: 1%).

    HANDLE LEAK PREVENTION:
    - Windows hat ein Limit von ~10.000 USER objects pro Prozess
    - Ohne Throttling: 1000+ Updates/Sekunde → Handle exhaustion
    - Mit Throttling: Max 100 Updates → kein Leak

    Args:
        callback: Original callback (progress_percent: int, message: str)
        scale_min: Minimum output value (default: 0)
        scale_max: Maximum output value (default: 100)
        min_change: Minimum change required to emit (default: 1%)

    Returns:
        Throttled callback function

    Usage:
        # In RenderWorker:
        throttled = create_throttled_callback(
            self.progress_updated.emit,
            scale_min=20,  # Map 0% to 20%
            scale_max=100  # Map 100% to 100%
        )
        throttled(0.5, "Rendering...")  # Emits (60, "Rendering...")
    """
    # Mutable container for closure state
    last_emitted = [scale_min - min_change]  # Start below minimum to emit first update

    def throttled_callback(progress: float, message: str) -> None:
        """Throttled callback that only emits on significant change."""
        # Map progress (0.0-1.0) to scale range
        scaled = int(scale_min + (progress * (scale_max - scale_min)))

        # Only emit if change is significant
        if scaled >= last_emitted[0] + min_change:
            last_emitted[0] = scaled
            callback(scaled, message)

    return throttled_callback


def create_throttled_pacing_callback(
    callback: Callable[[int, str], None], scale_max: int = 20
) -> Callable[[int, int, str], None]:
    """
    Erstellt einen gedrosselten Callback für Pacing-Progress (current/total format).

    M-02 FIX: Spezialisierte Version für PacingProgressCallback Signatur.

    Args:
        callback: Original callback (progress_percent: int, message: str)
        scale_max: Maximum output value (default: 20 for pacing phase)

    Returns:
        Throttled pacing callback function

    Usage:
        throttled = create_throttled_pacing_callback(
            self.progress_updated.emit,
            scale_max=20  # Pacing is 20% of total render
        )
        throttled(50, 100, "Analyzing triggers...")  # Emits (10, "...")
    """
    last_emitted = [0]

    def throttled_callback(current: int, total: int, message: str) -> None:
        """Throttled pacing callback."""
        if total <= 0:
            return

        # Calculate percentage scaled to range
        pacing_percent = int((current / total) * scale_max)

        # Only emit if progress increased by at least 1%
        if pacing_percent > last_emitted[0]:
            last_emitted[0] = pacing_percent
            callback(pacing_percent, message)

    return throttled_callback
