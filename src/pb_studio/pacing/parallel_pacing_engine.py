"""
Parallele Pacing Engine - ProcessPoolExecutor Implementation
FIX-FERTIG für Integration in PB_studio

ACHTUNG: Dieses Modul ist NICHT aktiv, bis es in advanced_pacing_engine.py eingebunden wird!
"""

import logging
import os
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np

# Module-level logger for use in __main__ block
logger = logging.getLogger(__name__)


@dataclass
class ClipSelectionTask:
    """
    Daten für eine einzelne Clip-Auswahl-Aufgabe
    WICHTIG: Alle Daten müssen picklable sein (für multiprocessing)
    """

    task_id: int
    target_energy: float
    target_motion: float
    target_duration: float
    timeline_position: float

    # Zusätzliche Parameter (optional)
    min_energy: float | None = None
    max_energy: float | None = None
    min_motion: float | None = None
    max_motion: float | None = None

    # Constraints
    avoid_clip_ids: list[int] | None = None
    preferred_tags: list[str] | None = None


@dataclass
class ClipSelectionResult:
    """Ergebnis einer Clip-Auswahl"""

    task_id: int
    clip_id: int
    clip_path: str
    score: float
    energy: float
    motion: float
    duration: float

    # Metadata
    selection_time: float  # Wie lange die Auswahl gedauert hat
    worker_id: int  # Welcher Worker das gemacht hat


def select_clip_worker(task_data: dict[str, Any]) -> dict[str, Any]:
    """
    Worker-Funktion für parallele Clip-Auswahl (DUMMY-IMPLEMENTATION)

    WICHTIG: Diese Funktion läuft in einem separaten Prozess!
    Sie darf KEINE Referenzen auf GUI-Objekte, PyQt, oder nicht-picklable Daten haben.

    Args:
        task_data: Dictionary mit allen benötigten Daten:
            - task: ClipSelectionTask
            - clips_features: List[Dict] mit allen Clip-Features
                Jedes Dict muss enthalten:
                - 'id': Clip ID
                - 'path': File path
                - 'energy': Energy score (0-1)
                - 'motion': Motion score (0-1)
                - 'duration': Duration in seconds

    Returns:
        Dictionary mit ClipSelectionResult Daten
    """
    start_time = time.time()
    worker_id = os.getpid()

    # Unpack task data
    task = task_data["task"]
    clips_features = task_data["clips_features"]

    # SIMPLE IMPLEMENTATION: Distance-based selection
    target_vector = np.array([task.target_energy, task.target_motion])

    best_clip_id = 0
    best_score = float("inf")

    for i, clip in enumerate(clips_features):
        # Skip if in avoid list
        if task.avoid_clip_ids and clip["id"] in task.avoid_clip_ids:
            continue

        # Calculate distance
        clip_vector = np.array([clip["energy"], clip["motion"]])
        distance = np.linalg.norm(target_vector - clip_vector)

        if distance < best_score:
            best_score = distance
            best_clip_id = i

    best_clip = clips_features[best_clip_id]
    selection_time = time.time() - start_time

    return {
        "task_id": task.task_id,
        "clip_id": best_clip["id"],
        "clip_path": best_clip["path"],
        "score": best_score,
        "energy": best_clip["energy"],
        "motion": best_clip["motion"],
        "duration": best_clip["duration"],
        "selection_time": selection_time,
        "worker_id": worker_id,
    }


class ParallelPacingEngine:
    """
    Parallele Pacing Engine mit ProcessPoolExecutor

    Diese Klasse kann als Drop-in Replacement für AdvancedPacingEngine verwendet werden.
    """

    def __init__(self, max_workers: int | None = None):
        """
        Args:
            max_workers: Anzahl paralleler Worker (None = auto-detect)
        """
        self.max_workers = max_workers or os.cpu_count()
        logger.info(f"[ParallelPacingEngine] Initialized with {self.max_workers} workers")

        # Stats
        self.total_clips_selected = 0
        self.total_selection_time = 0.0
        self.speedup_factor = 0.0

    def generate_cutlist_parallel(
        self,
        tasks: list[ClipSelectionTask],
        clips_features: list[dict[str, Any]],
        faiss_index_data: bytes | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
        chunksize: int = 1,
    ) -> list[ClipSelectionResult]:
        """
        Generiert Cutlist parallel mit ProcessPoolExecutor

        Args:
            tasks: Liste von ClipSelectionTask
            clips_features: Liste von Clip-Features (muss picklable sein!)
            faiss_index_data: Serialisierte FAISS Index Daten (optional)
            progress_callback: Callback für Progress Updates (current, total, message)
            chunksize: Anzahl Tasks pro Worker-Batch (für Optimierung)

        Returns:
            Liste von ClipSelectionResult (sortiert nach task_id)
        """
        logger.info("[ParallelPacingEngine] Starting parallel cutlist generation")
        logger.info(f"  Tasks: {len(tasks)}")
        logger.info(f"  Workers: {self.max_workers}")
        logger.info(f"  Chunksize: {chunksize}")

        start_time = time.time()
        results = []

        # Prepare task data (muss picklable sein!)
        task_data_list = [
            {"task": task, "clips_features": clips_features, "faiss_index_data": faiss_index_data}
            for task in tasks
        ]

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task_id = {
                executor.submit(select_clip_worker, task_data): task_data["task"].task_id
                for task_data in task_data_list
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_task_id):
                task_id = future_to_task_id[future]

                try:
                    result_dict = future.result()

                    # Convert dict to ClipSelectionResult
                    result = ClipSelectionResult(**result_dict)
                    results.append(result)

                    completed += 1

                    # Progress callback
                    if progress_callback:
                        progress_callback(
                            completed,
                            len(tasks),
                            f"Clip {completed}/{len(tasks)} selected (Worker PID: {result.worker_id})",
                        )

                    # Log every 10 clips
                    if completed % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed
                        eta = (len(tasks) - completed) / rate
                        logger.info(
                            f"  Progress: {completed}/{len(tasks)} clips "
                            f"({rate:.1f} clips/sec, ETA: {eta:.0f}s)"
                        )

                except Exception as e:
                    logger.error(f"  Task {task_id} failed: {e}")
                    # Optional: retry logic hier

        # Sort results by task_id
        results.sort(key=lambda r: r.task_id)

        # Stats
        total_time = time.time() - start_time
        self.total_clips_selected = len(results)
        self.total_selection_time = total_time

        # Calculate average sequential time (from selection_time of each clip)
        avg_clip_time = np.mean([r.selection_time for r in results])
        estimated_sequential_time = avg_clip_time * len(results)
        self.speedup_factor = estimated_sequential_time / total_time

        logger.info("[ParallelPacingEngine] Parallel generation complete!")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Clips selected: {len(results)}")
        logger.info(f"  Rate: {len(results) / total_time:.1f} clips/sec")
        logger.info(f"  Estimated speedup: {self.speedup_factor:.2f}x")

        return results

    def generate_cutlist_with_map(
        self,
        tasks: list[ClipSelectionTask],
        clips_features: list[dict[str, Any]],
        faiss_index_data: bytes | None = None,
        chunksize: int = 10,
    ) -> list[ClipSelectionResult]:
        """
        Alternative Implementierung mit executor.map() statt submit()

        Einfacher aber weniger Kontrolle über Progress.

        Args:
            tasks: Liste von ClipSelectionTask
            clips_features: Liste von Clip-Features
            faiss_index_data: Serialisierte FAISS Index Daten (optional)
            chunksize: Anzahl Tasks pro Worker-Batch

        Returns:
            Liste von ClipSelectionResult
        """
        logger.info("[ParallelPacingEngine] Starting parallel cutlist generation (map mode)")
        logger.info(f"  Tasks: {len(tasks)}")
        logger.info(f"  Workers: {self.max_workers}")
        logger.info(f"  Chunksize: {chunksize}")

        start_time = time.time()

        # Prepare task data
        task_data_list = [
            {"task": task, "clips_features": clips_features, "faiss_index_data": faiss_index_data}
            for task in tasks
        ]

        # Execute with map (returns results in order!)
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            result_dicts = executor.map(select_clip_worker, task_data_list, chunksize=chunksize)

            # Convert to ClipSelectionResult objects
            results = [ClipSelectionResult(**r) for r in result_dicts]

        total_time = time.time() - start_time

        logger.info("[ParallelPacingEngine] Parallel generation complete!")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Rate: {len(results) / total_time:.1f} clips/sec")

        return results


class ProgressTracker:
    """
    Helper class für Thread-safe Progress Tracking
    Kann von mehreren Prozessen verwendet werden
    """

    def __init__(self, total: int):
        self.total = total
        self.current = 0
        self.start_time = time.time()

    def update(self, n: int = 1) -> dict[str, Any]:
        """
        Update progress

        Args:
            n: Anzahl abgeschlossener Tasks

        Returns:
            Dict mit progress info
        """
        self.current += n
        elapsed = time.time() - self.start_time

        if elapsed > 0:
            rate = self.current / elapsed
            eta = (self.total - self.current) / rate if rate > 0 else 0
        else:
            rate = 0
            eta = 0

        # H-03 FIX: Division by Zero Guard
        progress_pct = (self.current / self.total) * 100 if self.total > 0 else 0

        return {
            "current": self.current,
            "total": self.total,
            "progress_pct": progress_pct,
            "elapsed": elapsed,
            "rate": rate,
            "eta": eta,
        }

    def get_message(self) -> str:
        """Returns formatted progress message"""
        info = self.update(0)  # Don't increment, just get info
        return (
            f"{info['current']}/{info['total']} clips "
            f"({info['progress_pct']:.1f}%) - "
            f"{info['rate']:.1f} clips/sec - "
            f"ETA: {info['eta']:.0f}s"
        )


# Utility functions


def estimate_speedup(
    num_clips: int, avg_clip_time_sequential: float, max_workers: int | None = None
) -> dict[str, float]:
    """
    Schätzt Speedup für parallele Verarbeitung

    Args:
        num_clips: Anzahl zu verarbeitender Clips
        avg_clip_time_sequential: Durchschnittliche Zeit pro Clip (sequenziell)
        max_workers: Anzahl Worker (None = auto)

    Returns:
        Dict mit Schätzungen
    """
    if max_workers is None:
        max_workers = os.cpu_count()

    # Amdahl's Law: Speedup = 1 / ((1 - P) + P/N)
    # Annahme: P = 0.95 (95% parallelisierbar)
    P = 0.95
    theoretical_speedup = 1 / ((1 - P) + P / max_workers)

    # Realistischer Speedup (mit Overhead)
    overhead_factor = 0.85  # 15% Overhead für IPC, etc.
    realistic_speedup = theoretical_speedup * overhead_factor

    # Zeitschätzungen
    sequential_time = num_clips * avg_clip_time_sequential
    parallel_time = sequential_time / realistic_speedup
    time_saved = sequential_time - parallel_time

    return {
        "num_clips": num_clips,
        "max_workers": max_workers,
        "theoretical_speedup": theoretical_speedup,
        "realistic_speedup": realistic_speedup,
        "sequential_time_hours": sequential_time / 3600,
        "parallel_time_hours": parallel_time / 3600,
        "time_saved_hours": time_saved / 3600,
        "avg_clip_time_sequential": avg_clip_time_sequential,
        "avg_clip_time_parallel": avg_clip_time_sequential / realistic_speedup,
    }


if __name__ == "__main__":
    # Configure basic logging for standalone execution
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Quick test
    logger.info("ParallelPacingEngine - Ready for integration!")
    logger.info(f"CPU Cores available: {os.cpu_count()}")

    # Estimate for PB_studio
    estimate = estimate_speedup(
        num_clips=553,
        avg_clip_time_sequential=180,  # 3 Minuten (konservativ)
        max_workers=os.cpu_count(),
    )

    logger.info("\nEstimate for PB_studio (553 clips):")
    logger.info(f"  Workers: {estimate['max_workers']}")
    logger.info(f"  Theoretical speedup: {estimate['theoretical_speedup']:.2f}x")
    logger.info(f"  Realistic speedup: {estimate['realistic_speedup']:.2f}x")
    logger.info(f"  Sequential time: {estimate['sequential_time_hours']:.1f} hours")
    logger.info(f"  Parallel time: {estimate['parallel_time_hours']:.1f} hours")
    logger.info(f"  Time saved: {estimate['time_saved_hours']:.1f} hours")
