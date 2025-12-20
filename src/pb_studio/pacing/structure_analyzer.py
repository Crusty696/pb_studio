"""
Song-Struktur-Analyse für intelligentes Pacing

Erkennt Songstruktur mit Self-Similarity Matrix:
- Intro/Verse/Chorus/Drop/Outro Detection
- Segment-Boundaries mit Novelty Curve
- Chroma-Features für harmonische Analyse

Author: PB_studio Development Team
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SegmentInfo:
    """
    Song-Segment Information.

    Attributes:
        start_time: Start-Zeit in Sekunden
        end_time: End-Zeit in Sekunden
        segment_type: Art des Segments (intro, verse, chorus, drop, outro)
        energy: Durchschnittliche Energy des Segments (0.0-1.0)
        chroma_mean: Durchschnittliches Chroma-Feature
    """

    start_time: float
    end_time: float
    segment_type: str
    energy: float
    chroma_mean: np.ndarray


@dataclass
class StructureAnalysisResult:
    """
    Ergebnis der Song-Struktur-Analyse.

    Attributes:
        audio_path: Pfad zur Audio-Datei
        duration: Gesamtdauer in Sekunden
        segments: Liste der erkannten Segmente
        boundary_times: Liste der Segment-Grenzen in Sekunden
        similarity_matrix: Self-Similarity Matrix (Optional, für Debugging)
    """

    audio_path: str
    duration: float
    segments: list[SegmentInfo]
    boundary_times: list[float]
    similarity_matrix: np.ndarray | None = None


class StructureAnalyzer:
    """
    Analysiert Song-Struktur mit librosa.

    Verwendet:
    - Chroma-Features für harmonische Analyse
    - Self-Similarity Matrix (Recurrence Matrix)
    - Novelty Curve für Boundary Detection
    - Energy-basierte Segment-Klassifizierung

    PERFORMANCE LIMIT: Vorherige Recurrence-Matrix war O(n²) Speicher.
    Jetzt speichereffiziente Online-Novelty (O(n)).
    """

    def __init__(
        self,
        hop_length: int = 512,
        n_fft: int = 2048,
        n_chroma: int = 12,
        min_segment_duration: float = 4.0,
        novelty_window_seconds: float = 12.0,
        novelty_smooth_frames: int = 9,
        novelty_peak_delta: float = 0.05,
        recurrence_neighbors: int = 20,
        max_debug_matrix_frames: int = 4000,
    ):
        """
        Initialisiert Structure-Analyzer.

        Args:
            hop_length: Hop-Length für STFT (default: 512)
            n_fft: FFT-Größe (default: 2048)
            n_chroma: Anzahl Chroma-Bins (default: 12)
            min_segment_duration: Minimale Segment-Länge in Sekunden (default: 4.0)
            novelty_window_seconds: Fenstergröße für Online-Novelty (sekundenbasiert)
            novelty_smooth_frames: Glättungs-Fenster für Novelty-Kurve (Anzahl Frames)
            novelty_peak_delta: Sensitivität für Peak-Picking (Novelty-Minimum)
            recurrence_neighbors: K-NN für optionale Sparse-Recurrence (Debug)
            max_debug_matrix_frames: Max. Frames für optionale Similarity-Matrix (verhindert O(n^2))
        """
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_chroma = n_chroma
        self.min_segment_duration = min_segment_duration
        self.novelty_window_seconds = novelty_window_seconds
        self.novelty_smooth_frames = max(1, int(novelty_smooth_frames))
        self.novelty_peak_delta = novelty_peak_delta
        self.recurrence_neighbors = recurrence_neighbors
        self.max_debug_matrix_frames = max_debug_matrix_frames

        logger.info(
            f"StructureAnalyzer initialisiert: hop_length={hop_length}, "
            f"min_segment_duration={min_segment_duration}s"
        )

    def analyze_structure(
        self, audio_path: str, return_similarity_matrix: bool = False
    ) -> StructureAnalysisResult:
        """
        Analysiert Song-Struktur.

        Args:
            audio_path: Pfad zur Audio-Datei
            return_similarity_matrix: Ob Similarity-Matrix zurückgegeben werden soll

        Returns:
            StructureAnalysisResult mit erkannten Segmenten

        Raises:
            FileNotFoundError: Wenn Audio-Datei nicht existiert
            ValueError: Wenn Audio-Datei korrupt oder zu kurz
        """
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio-Datei nicht gefunden: {audio_path}")

        logger.info(f"Analysiere Song-Struktur: {Path(audio_path).name}")

        try:
            # Zuerst Dauer prüfen ohne volles Laden
            duration = librosa.get_duration(path=audio_path)

            if duration < self.min_segment_duration:
                raise ValueError(
                    f"Audio zu kurz ({duration:.1f}s), min: {self.min_segment_duration}s"
                )

            # Audio laden (Mono für effiziente Analyse)
            y, sr = librosa.load(audio_path, sr=None, mono=True)

            # Chroma-Features extrahieren
            chroma = self._extract_chroma_features(y, sr)

            # Online-Novelty-Kurve (O(n) Speicher)
            novelty_curve = self._compute_online_novelty(chroma, sr)

            # Segment-Boundaries finden
            boundary_frames = self._detect_boundaries_from_novelty(
                novelty_curve, chroma.shape[1], sr
            )

            # Frames → Zeit konvertieren
            boundary_times = librosa.frames_to_time(
                boundary_frames, sr=sr, hop_length=self.hop_length
            )

            # Segmente klassifizieren
            segments = self._classify_segments(y, sr, boundary_times, chroma)

            similarity_matrix = None
            if return_similarity_matrix and chroma.shape[1] <= self.max_debug_matrix_frames:
                similarity_matrix = self._compute_similarity_matrix(chroma)

            logger.info(f"Struktur-Analyse abgeschlossen: {len(segments)} Segmente erkannt")

            return StructureAnalysisResult(
                audio_path=audio_path,
                duration=duration,
                segments=segments,
                boundary_times=list(boundary_times),
                similarity_matrix=similarity_matrix if return_similarity_matrix else None,
            )

        except Exception as e:
            logger.error(f"Struktur-Analyse-Fehler für {audio_path}: {e}")
            raise

    def _extract_chroma_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extrahiert Chroma-Features.

        Args:
            y: Audio-Signal
            sr: Sample-Rate

        Returns:
            Chroma-Features (n_chroma x n_frames)
        """
        chroma = librosa.feature.chroma_cqt(
            y=y, sr=sr, hop_length=self.hop_length, n_chroma=self.n_chroma
        )

        # Normalisierung (Energie-unabhängig)
        chroma = librosa.util.normalize(chroma, axis=0)

        return chroma

    def _compute_online_novelty(self, chroma: np.ndarray, sr: int) -> np.ndarray:
        """
        Berechnet speichereffiziente Novelty-Kurve (O(n) Speicher).

        Kombination aus spektraler Fluss-Approximation (librosa onset_strength)
        und einfachem Frame-Differenz-Maß, um Strukturwechsel auch bei langen
        Dateien ohne volle Recurrence-Matrix zu erkennen.
        """
        spectral_flux = librosa.onset.onset_strength(
            S=chroma, sr=sr, hop_length=self.hop_length, center=False
        )

        frame_diff = np.sum(np.abs(np.diff(chroma, axis=1)), axis=0)
        frame_diff = np.concatenate([[0.0], frame_diff])

        novelty = frame_diff
        if spectral_flux.size > 0:
            spectral_flux = librosa.util.fix_length(spectral_flux, size=novelty.shape[0])
            novelty = novelty + spectral_flux

        if self.novelty_smooth_frames > 1 and novelty.size > 0:
            kernel = np.ones(self.novelty_smooth_frames, dtype=float) / self.novelty_smooth_frames
            novelty = np.convolve(novelty, kernel, mode="same")

        if novelty.size > 0:
            # Kontextuelle Glättung über ein sekundenbasiertes Fenster (ohne O(n²))
            window_frames = max(
                self.novelty_smooth_frames,
                int(max(1, (self.novelty_window_seconds * sr) / self.hop_length / 6)),
            )
            if window_frames > 1:
                long_kernel = np.ones(window_frames, dtype=float) / window_frames
                novelty = np.convolve(novelty, long_kernel, mode="same")

        if novelty.size > 0:
            novelty = librosa.util.normalize(novelty)

        return novelty

    def _detect_boundaries_from_novelty(
        self, novelty_curve: np.ndarray, n_frames: int, sr: int
    ) -> np.ndarray:
        """
        Erkennt Segment-Boundaries basierend auf einer Novelty-Kurve.

        Args:
            novelty_curve: 1D Novelty-Kurve (Länge ≈ n_frames)
            n_frames: Anzahl Frames (Backup für Grenzen)
            sr: Sample-Rate

        Returns:
            Array von Boundary-Frame-Indizes
        """
        if novelty_curve.size == 0:
            return np.array([0, max(0, n_frames - 1)])

        min_frames = max(1, int(self.min_segment_duration * sr / self.hop_length))

        peaks = librosa.util.peak_pick(
            novelty_curve,
            pre_max=min_frames,
            post_max=min_frames,
            pre_avg=min_frames,
            post_avg=min_frames,
            delta=self.novelty_peak_delta,
            wait=min_frames,
        )

        if peaks.size == 0 and novelty_curve.size > 0:
            # Fallback: staerkster Peak als Boundary
            max_idx = int(np.argmax(novelty_curve))
            if 0 < max_idx < n_frames - 1:
                peaks = np.array([max_idx])

        boundaries = np.concatenate([[0], peaks, [n_frames - 1]])
        boundaries = np.unique(np.clip(boundaries, 0, n_frames - 1))

        return boundaries

    def _compute_similarity_matrix(self, chroma: np.ndarray) -> np.ndarray | None:
        """
        Optionale Similarity-Matrix für Debugging (sparsam, kein O(n^2) Voll-Matrix).
        """
        try:
            k = min(self.recurrence_neighbors, max(1, chroma.shape[1] - 1))
            return librosa.segment.recurrence_matrix(
                chroma, k=k, metric="cosine", mode="affinity", sparse=False
            )
        except Exception as exc:
            logger.debug(f"Similarity-Matrix konnte nicht berechnet werden: {exc}")
            return None

    def _classify_segments(
        self, y: np.ndarray, sr: int, boundary_times: np.ndarray, chroma: np.ndarray
    ) -> list[SegmentInfo]:
        """
        Klassifiziert Segmente nach Typ.

        Heuristik:
        - Intro: Erstes Segment, niedrige Energy
        - Outro: Letztes Segment, niedrige Energy
        - Drop: Hohe Energy + plötzlicher Anstieg
        - Chorus: Hohe Energy + wiederkehrendes Chroma-Pattern
        - Verse: Mittlere Energy

        Args:
            y: Audio-Signal
            sr: Sample-Rate
            boundary_times: Segment-Boundaries in Sekunden
            chroma: Chroma-Features

        Returns:
            Liste von SegmentInfo
        """
        segments = []
        duration = librosa.get_duration(y=y, sr=sr)

        # RMS Energy für gesamten Song
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        rms_median = np.median(rms)
        if rms_median <= 1e-8:
            rms_median = 1e-8

        for i in range(len(boundary_times) - 1):
            start_time = float(boundary_times[i])
            end_time = float(boundary_times[i + 1])

            # Segment-Audio extrahieren
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment_y = y[start_sample:end_sample]

            # Segment-Energy
            segment_rms = np.mean(librosa.feature.rms(y=segment_y)[0])
            normalized_energy = float(segment_rms / rms_median)

            # Segment-Chroma
            start_frame = int(start_time * sr / self.hop_length)
            end_frame = int(end_time * sr / self.hop_length)
            segment_chroma = chroma[:, start_frame:end_frame]
            chroma_mean = np.mean(segment_chroma, axis=1)

            # Segment-Typ klassifizieren
            segment_type = self._infer_segment_type(
                i, len(boundary_times) - 1, normalized_energy, start_time, end_time, duration
            )

            segments.append(
                SegmentInfo(
                    start_time=start_time,
                    end_time=end_time,
                    segment_type=segment_type,
                    energy=min(1.0, normalized_energy),
                    chroma_mean=chroma_mean,
                )
            )

            logger.debug(
                f"  Segment {i}: {start_time:.1f}s-{end_time:.1f}s, "
                f"type={segment_type}, energy={normalized_energy:.2f}"
            )

        return segments

    def _infer_segment_type(
        self,
        segment_idx: int,
        total_segments: int,
        energy: float,
        start_time: float,
        end_time: float,
        total_duration: float,
    ) -> str:
        """
        Inferiert Segment-Typ mit verbesserter Heuristik.

        Args:
            segment_idx: Index des Segments
            total_segments: Gesamtzahl Segmente
            energy: Normalisierte Energy (relativ zu Median)
            start_time: Start-Zeit
            end_time: End-Zeit
            total_duration: Gesamtdauer

        Returns:
            Segment-Typ (intro, verse, chorus, drop, bridge, outro)
        """
        # Relative Position im Song
        relative_position = start_time / total_duration

        # Intro: Erstes Segment ODER frühe Position + niedrige Energy
        if segment_idx == 0:
            return "intro"
        if relative_position < 0.15 and energy < 0.9:
            return "intro"

        # Outro: Letztes Segment ODER späte Position + niedrige Energy
        if segment_idx == total_segments - 2:
            return "outro"
        if relative_position > 0.85 and energy < 0.9:
            return "outro"

        # Bridge: Mittlere Position (40-60%) + moderate Energy
        if 0.4 < relative_position < 0.6 and 0.85 < energy < 1.15:
            return "bridge"

        # Drop: Sehr hohe Energy (>1.4x Median) - typisch für EDM
        if energy > 1.4:
            return "drop"

        # Chorus: Hohe Energy (>1.15x Median)
        if energy > 1.15:
            return "chorus"

        # Build-up: Steigende Energy vor hohem Segment
        # (Erkennung würde Energy-Gradient erfordern - TODO)

        # Verse: Moderate bis niedrige Energy
        if energy < 1.05:
            return "verse"

        # Default: Verse (sicherste Annahme)
        return "verse"

    def get_segment_at_time(
        self, result: StructureAnalysisResult, time: float
    ) -> SegmentInfo | None:
        """
        Findet Segment zu gegebener Zeit.

        Args:
            result: Struktur-Analyse-Ergebnis
            time: Zeit in Sekunden

        Returns:
            SegmentInfo wenn gefunden, sonst None
        """
        for segment in result.segments:
            if segment.start_time <= time < segment.end_time:
                return segment

        return None

    def _analyze_structure_simplified(
        self, audio_path: str, duration: float
    ) -> StructureAnalysisResult:
        """
        Vereinfachte Struktur-Analyse fuer lange Dateien.

        Statt Recurrence-Matrix (O(n²) Speicher) wird Energy-basierte
        Segmentierung verwendet. Schnell und speichereffizient.

        Args:
            audio_path: Pfad zur Audio-Datei
            duration: Dauer in Sekunden

        Returns:
            StructureAnalysisResult mit groben Segmenten
        """
        logger.info(f"Vereinfachte Struktur-Analyse fuer {duration/60:.1f} Min Audio")

        # Downsampled Audio laden (11025 Hz fuer Performance)
        y, sr = librosa.load(audio_path, sr=11025, mono=True)

        # RMS Energy mit grossem Hop fuer schnelle Berechnung
        hop = 4096  # Weniger Frames = schneller
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        rms_median = np.median(rms)

        # Segment-Laenge: alle 2 Minuten ein neues Segment
        segment_duration = 120.0  # 2 Minuten
        num_segments = int(np.ceil(duration / segment_duration))

        segments = []
        boundary_times = [0.0]

        for i in range(num_segments):
            start_time = i * segment_duration
            end_time = min((i + 1) * segment_duration, duration)
            boundary_times.append(end_time)

            # Energy fuer dieses Segment
            start_frame = int(start_time * sr / hop)
            end_frame = min(int(end_time * sr / hop), len(rms))

            if end_frame > start_frame:
                segment_rms = np.mean(rms[start_frame:end_frame])
                normalized_energy = float(segment_rms / rms_median) if rms_median > 0 else 1.0
            else:
                normalized_energy = 1.0

            # Segment-Typ basierend auf Position und Energy
            segment_type = self._infer_segment_type(
                i, num_segments, normalized_energy, start_time, end_time, duration
            )

            # Dummy Chroma (12 Nullen)
            chroma_mean = np.zeros(12)

            segments.append(
                SegmentInfo(
                    start_time=start_time,
                    end_time=end_time,
                    segment_type=segment_type,
                    energy=min(1.0, normalized_energy),
                    chroma_mean=chroma_mean,
                )
            )

            logger.debug(
                f"  Segment {i}: {start_time:.1f}s-{end_time:.1f}s, "
                f"type={segment_type}, energy={normalized_energy:.2f}"
            )

        logger.info(f"Vereinfachte Struktur-Analyse abgeschlossen: {len(segments)} Segmente")

        return StructureAnalysisResult(
            audio_path=audio_path,
            duration=duration,
            segments=segments,
            boundary_times=boundary_times,
            similarity_matrix=None,
        )
