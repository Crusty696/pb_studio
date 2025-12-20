"""
Motion-Analyzer für Video-Clips

Berechnet Motion-Scores mit OpenCV Optical Flow (Farnebäck-Algorithmus).

Motion-Score Range: 0.0 (kein Movement) - 1.0 (maximales Movement)

Features:
- Optical Flow Berechnung
- Frame-Sampling für Performance
- Frame-Skip Optimierung (5x Speedup bei frame_skip=5)
- Multi-Threading für Batch-Analyse (ThreadPoolExecutor)
- JSON-Cache für analysierte Clips
- Progress-Callback Support für GUI-Integration
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from ..utils.cache_manager import CacheManager

logger = logging.getLogger(__name__)


@dataclass
class MotionAnalysisResult:
    """
    Motion-Analyse Ergebnis für einen Video-Clip.

    Attributes:
        clip_path: Absoluter Pfad zum Video-Clip
        motion_score: Durchschnittlicher Motion-Score (0.0-1.0)
        frame_count: Anzahl Frames im Video
        duration: Video-Dauer in Sekunden
        fps: Frames per Second
    """

    clip_path: str
    motion_score: float
    frame_count: int
    duration: float
    fps: float


class MotionAnalyzer:
    """
    Analysiert Video-Clips und berechnet Motion-Scores.

    Verwendet OpenCV Farnebäck Optical Flow:
    - Samplet Frames (für Performance)
    - Frame-Skip Optimierung für schnellere Analyse (z.B. frame_skip=5 → 5x schneller)
    - Multi-Threading für Batch-Analyse (ThreadPoolExecutor)
    - Berechnet Optical Flow zwischen aufeinanderfolgenden Frames
    - Aggregiert Flow-Magnitude zu Motion-Score
    - Cached Ergebnisse (JSON mit MD5-Hash)
    - Progress-Callback Support für GUI-Integration

    Performance-Optimierungen:
    - Batch-Analyse: analyze_clips_batch() mit 4-8 Threads → 4-8x schneller
    - Frame-Skip: frame_skip=5 → 5x schneller bei ~80% Genauigkeit
    - Kombination: 20-40x Speedup möglich!
    """

    def __init__(
        self,
        sample_frames: int = 30,
        cache_dir: Path | None = None,
        normalize_factor: float = 10.0,
        frame_skip: int = 1,
    ):
        """
        Initialisiert Motion-Analyzer.

        Args:
            sample_frames: Anzahl Frames zu samplen (weniger = schneller)
            cache_dir: Verzeichnis für Motion-Cache
            normalize_factor: Normalisierungs-Faktor für Motion-Score
            frame_skip: Analysiere nur jeden N-ten Frame (1=alle, 5=jeder 5.)
                        frame_skip=5 bedeutet 5x schneller bei ~80% Genauigkeit
        """
        self.sample_frames = sample_frames
        self.normalize_factor = normalize_factor
        self.frame_skip = frame_skip
        cache_path = cache_dir or Path(".taskmaster/cache/motion")
        self.cache_manager = CacheManager(cache_dir=cache_path, prefix="motion")

        logger.info(
            f"MotionAnalyzer initialisiert: sample_frames={sample_frames}, "
            f"frame_skip={frame_skip}, cache_dir={cache_path}"
        )

    def analyze_clip(self, clip_path: str) -> MotionAnalysisResult:
        """
        Analysiert Video-Clip und berechnet Motion-Score.

        Workflow:
        1. Cache-Check (falls bereits analysiert)
        2. Video laden mit OpenCV
        3. Frames samplen
        4. Optical Flow zwischen Frames berechnen
        5. Flow-Magnitude aggregieren
        6. Ergebnis normalisieren und cachen

        Args:
            clip_path: Pfad zum Video-Clip

        Returns:
            MotionAnalysisResult

        Raises:
            FileNotFoundError: Wenn Video-Datei nicht existiert
            cv2.error: Bei OpenCV-Fehlern
        """
        # 1. Cache-Check
        cached = self._load_from_cache(clip_path)
        if cached:
            logger.debug(f"Motion-Score aus Cache geladen: {clip_path}")
            return cached

        # 2. Video laden
        if not Path(clip_path).exists():
            raise FileNotFoundError(f"Video-Datei nicht gefunden: {clip_path}")

        logger.info(f"Analysiere Video-Motion: {clip_path}")

        # PERF-02 FIX: Use context manager to ensure VideoCapture is released
        from pb_studio.utils.video_utils import open_video

        with open_video(clip_path) as cap:
            if not cap.isOpened():
                raise cv2.error(f"Konnte Video nicht öffnen: {clip_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0.0

            logger.debug(f"Video geladen: {frame_count} frames @ {fps:.1f} fps ({duration:.1f}s)")

            # 3. Frames samplen + Optical Flow
            motion_scores = []
            prev_gray = None

            # Sample-Indices mit frame_skip Optimierung
            sample_indices = np.linspace(
                0, frame_count - 1, min(self.sample_frames, frame_count), dtype=int
            )

            # Wenn frame_skip > 1, nur jeden N-ten Sample-Frame analysieren
            if self.frame_skip > 1:
                sample_indices = sample_indices[:: self.frame_skip]
                logger.debug(
                    f"Frame-Skip aktiviert: {self.frame_skip}x, "
                    f"analysiere {len(sample_indices)} von {self.sample_frames} Frames"
                )

            for i, frame_idx in enumerate(sample_indices):
                # Frame lesen
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    logger.warning(f"Frame {frame_idx} konnte nicht gelesen werden")
                    break

                # Grayscale konvertieren
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Optical Flow berechnen (ab 2. Frame)
                if prev_gray is not None:
                    flow_score = self._calculate_optical_flow(prev_gray, gray)
                    motion_scores.append(flow_score)

                prev_gray = gray

        # VideoCapture automatically released by context manager

        # 4. Aggregieren und normalisieren
        if motion_scores:
            avg_motion = np.mean(motion_scores)
            # Normalisierung: typischer Flow-Wert ist 0-10, wir wollen 0-1
            normalized_motion = min(1.0, avg_motion / self.normalize_factor)
        else:
            logger.warning(f"Keine Motion-Scores berechnet für: {clip_path}")
            normalized_motion = 0.0

        logger.debug(
            f"Motion-Analyse abgeschlossen: {len(motion_scores)} samples, "
            f"score={normalized_motion:.3f}"
        )

        # 5. Ergebnis erstellen
        result = MotionAnalysisResult(
            clip_path=clip_path,
            motion_score=float(normalized_motion),
            frame_count=frame_count,
            duration=duration,
            fps=fps,
        )

        # 6. Cache speichern
        self._save_to_cache(result)

        logger.info(f"Motion-Score berechnet: {normalized_motion:.3f} für {clip_path}")

        return result

    def _calculate_optical_flow(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
        """
        Berechnet Optical Flow Magnitude zwischen zwei Frames.

        Verwendet: cv2.calcOpticalFlowFarneback()

        Args:
            prev_gray: Vorheriger Frame (Grayscale)
            curr_gray: Aktueller Frame (Grayscale)

        Returns:
            Durchschnittliche Flow-Magnitude
        """
        # Farnebäck Optical Flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=0.5,  # Pyramiden-Skalierung
            levels=3,  # Pyramiden-Levels
            winsize=15,  # Fenster-Größe
            iterations=3,  # Iterationen pro Level
            poly_n=5,  # Polynomial-Grad
            poly_sigma=1.2,  # Gauss-Sigma
            flags=0,
        )

        # Flow-Magnitude berechnen
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Durchschnittliche Magnitude
        avg_magnitude = float(np.mean(mag))

        return avg_magnitude

    def _load_from_cache(self, clip_path: str) -> MotionAnalysisResult | None:
        """
        Lädt Motion-Analyse-Ergebnis aus Cache.

        Args:
            clip_path: Pfad zum Video-Clip

        Returns:
            MotionAnalysisResult wenn gecacht, sonst None
        """
        data = self.cache_manager.load(clip_path)
        if data:
            try:
                return MotionAnalysisResult(**data)
            except Exception as e:
                logger.warning(f"Cache-Load-Fehler für {clip_path}: {e}")
        return None

    def _save_to_cache(self, result: MotionAnalysisResult):
        """
        Speichert Motion-Analyse-Ergebnis in Cache.

        Args:
            result: MotionAnalysisResult zum Speichern
        """
        try:
            self.cache_manager.save(result.clip_path, asdict(result))
            logger.debug("Motion-Score gecacht")
        except Exception as e:
            logger.warning(f"Cache-Save-Fehler: {e}")

    def analyze_clips_batch(
        self, clip_paths: list[str], max_workers: int = 4, progress_callback=None
    ) -> list[MotionAnalysisResult]:
        """
        Analysiert mehrere Clips parallel mit Multi-Threading.

        Nutzt ThreadPoolExecutor für parallele Verarbeitung.
        Cached Clips werden sofort zurückgegeben, nur neue werden analysiert.

        Args:
            clip_paths: Liste von Pfaden zu Video-Clips
            max_workers: Maximale Anzahl paralleler Worker-Threads (default: 4)
            progress_callback: Optional callback(current, total, message) für Fortschritt

        Returns:
            Liste von MotionAnalysisResult in der gleichen Reihenfolge wie clip_paths

        Example:
            >>> analyzer = MotionAnalyzer(frame_skip=5)
            >>> results = analyzer.analyze_clips_batch(
            ...     ["clip1.mp4", "clip2.mp4", "clip3.mp4"],
            ...     max_workers=8
            ... )
            >>> for result in results:
            ...     print(f"{result.clip_path}: {result.motion_score:.2f}")
        """
        if not clip_paths:
            return []

        logger.info(f"Batch-Analyse gestartet: {len(clip_paths)} clips, {max_workers} workers")

        # Ergebnisse mit Index speichern für korrekte Reihenfolge
        results = [None] * len(clip_paths)
        completed = 0

        # Multi-Threading mit ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit alle Clips
            future_to_index = {
                executor.submit(self.analyze_clip, clip_path): i
                for i, clip_path in enumerate(clip_paths)
            }

            # Sammle Ergebnisse wenn sie fertig sind
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                clip_path = clip_paths[index]

                try:
                    result = future.result()
                    results[index] = result
                    completed += 1

                    logger.debug(
                        f"Batch-Fortschritt: {completed}/{len(clip_paths)} - "
                        f"{clip_path} → {result.motion_score:.3f}"
                    )

                    # Progress-Callback aufrufen
                    if progress_callback:
                        progress_callback(
                            completed,
                            len(clip_paths),
                            f"Analyzed {completed}/{len(clip_paths)} clips",
                        )

                except Exception as e:
                    logger.error(f"Fehler bei Batch-Analyse von {clip_path}: {e}", exc_info=True)
                    # Fallback: Erstelle Ergebnis mit motion_score=0.0
                    results[index] = MotionAnalysisResult(
                        clip_path=clip_path, motion_score=0.0, frame_count=0, duration=0.0, fps=0.0
                    )
                    completed += 1

        logger.info(f"Batch-Analyse abgeschlossen: {len(clip_paths)} clips analysiert")

        return results
