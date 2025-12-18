"""Onset und Drum Detection.

Spezialisiertes Modul für Onset-Erkennung und präzise Drum-Hit-Detection.
"""

import logging
from typing import TypeAlias, TypedDict

import librosa
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Type aliases
AudioSamples: TypeAlias = NDArray[np.floating]


class DrumHitsResult(TypedDict):
    """Präzise Drum-Hit-Erkennung mit Frequenzband-Filterung."""

    kick_times: list[float]
    snare_times: list[float]
    hihat_times: list[float]
    kick_strengths: list[float]
    snare_strengths: list[float]
    hihat_strengths: list[float]


class OnsetDetector:
    """Onset und Drum-Hit Detection."""

    # Chunk-basierte Verarbeitung für lange Audio-Dateien
    DRUM_CHUNK_SECONDS = 300.0  # 5 Minuten
    DRUM_OVERLAP_SECONDS = 10.0  # Überlappung

    def __init__(self, sr: int = 11025) -> None:
        """
        Initialize Onset Detector.

        Args:
            sr: Sample rate for audio processing
        """
        self.sr = sr

    def detect_onsets(
        self, y: AudioSamples, sr: int, onset_envelope: NDArray | None = None
    ) -> np.ndarray:
        """
        Detect onsets in audio.

        Args:
            y: Audio samples
            sr: Sample rate
            onset_envelope: Pre-computed onset envelope (optional)

        Returns:
            Array of onset times in seconds
        """
        try:
            if onset_envelope is None:
                onset_envelope = librosa.onset.onset_strength(
                    y=y, sr=sr, aggregate=np.median, hop_length=512
                )

            onset_frames = librosa.onset.onset_detect(
                onset_envelope=onset_envelope, sr=sr, hop_length=512, backtrack=True
            )

            onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
            return onset_times

        except Exception as e:
            logger.error(f"Onset detection failed: {e}", exc_info=True)
            return np.array([])

    def detect_drum_hits(self, audio_path: str, sr: int = 32000) -> DrumHitsResult:
        """
        Präzise Drum-Detection mit frequenzbandbasierten Onsets.

        Chunk-basiert (O(n) Speicher) für auch lange Audio-Dateien (60+ Minuten).

        Args:
            audio_path: Path to audio file
            sr: Sample rate for drum detection (32000 recommended)

        Returns:
            DrumHitsResult with detected drum hits
        """
        try:
            duration = librosa.get_duration(path=audio_path)
            logger.info(f"Starting drum detection for {duration:.1f}s audio (chunked processing)")

            chunk_len = float(self.DRUM_CHUNK_SECONDS)
            overlap = float(self.DRUM_OVERLAP_SECONDS)
            stride = max(0.1, chunk_len - overlap)

            kick_times: list[float] = []
            snare_times: list[float] = []
            hihat_times: list[float] = []
            kick_strengths: list[float] = []
            snare_strengths: list[float] = []
            hihat_strengths: list[float] = []

            offset = 0.0
            while offset < duration + 1e-6:
                read_dur = min(chunk_len, duration - offset)
                if read_dur <= 0:
                    break

                y_chunk, sr_chunk = librosa.load(
                    audio_path, sr=sr, mono=True, offset=offset, duration=read_dur + overlap
                )

                chunk_result = self._analyze_drum_chunk(y_chunk, sr_chunk, offset)

                kick_times, kick_strengths = self._merge_hits(
                    kick_times,
                    kick_strengths,
                    chunk_result["kick_times"],
                    chunk_result["kick_strengths"],
                )
                snare_times, snare_strengths = self._merge_hits(
                    snare_times,
                    snare_strengths,
                    chunk_result["snare_times"],
                    chunk_result["snare_strengths"],
                )
                hihat_times, hihat_strengths = self._merge_hits(
                    hihat_times,
                    hihat_strengths,
                    chunk_result["hihat_times"],
                    chunk_result["hihat_strengths"],
                )

                offset += stride

            result: DrumHitsResult = {
                "kick_times": kick_times,
                "snare_times": snare_times,
                "hihat_times": hihat_times,
                "kick_strengths": kick_strengths,
                "snare_strengths": snare_strengths,
                "hihat_strengths": hihat_strengths,
            }

            logger.info(
                f"Drum detection complete: "
                f"{len(kick_times)} kicks, {len(snare_times)} snares, {len(hihat_times)} hihats"
            )

            return result

        except Exception as e:
            logger.error(f"Drum detection failed: {e}", exc_info=True)
            return {
                "kick_times": [],
                "snare_times": [],
                "hihat_times": [],
                "kick_strengths": [],
                "snare_strengths": [],
                "hihat_strengths": [],
            }

    def classify_onsets(self, onsets: np.ndarray, y: AudioSamples, sr: int) -> list[str]:
        """
        Classify onsets by type (kick, snare, hihat, other).

        Args:
            onsets: Array of onset times
            y: Audio samples
            sr: Sample rate

        Returns:
            List of onset classifications
        """
        try:
            classifications = []

            for onset_time in onsets:
                frame_idx = librosa.time_to_frames(onset_time, sr=sr, hop_length=512)
                sample_idx = frame_idx * 512

                # Extract short window around onset
                window_samples = int(0.05 * sr)  # 50ms window
                start_idx = max(0, sample_idx - window_samples // 2)
                end_idx = min(len(y), sample_idx + window_samples // 2)

                if end_idx <= start_idx:
                    classifications.append("unknown")
                    continue

                window = y[start_idx:end_idx]

                # Simple frequency-based classification
                spec = np.abs(librosa.stft(window))
                freqs = librosa.fft_frequencies(sr=sr)

                # Energy in different bands
                low_energy = np.sum(spec[(freqs >= 20) & (freqs <= 150)])
                mid_energy = np.sum(spec[(freqs >= 150) & (freqs <= 4000)])
                high_energy = np.sum(spec[(freqs >= 6000) & (freqs <= 16000)])

                # Classify based on dominant energy
                total_energy = low_energy + mid_energy + high_energy
                if total_energy == 0:
                    classifications.append("unknown")
                    continue

                if low_energy / total_energy > 0.5:
                    classifications.append("kick")
                elif high_energy / total_energy > 0.4:
                    classifications.append("hihat")
                elif mid_energy / total_energy > 0.5:
                    classifications.append("snare")
                else:
                    classifications.append("other")

            return classifications

        except Exception as e:
            logger.error(f"Onset classification failed: {e}", exc_info=True)
            return ["unknown"] * len(onsets)

    def _analyze_drum_chunk(
        self, y_full: np.ndarray, sr_full: int, offset_seconds: float
    ) -> DrumHitsResult:
        """Analysiert einen Audio-Chunk und gibt zeitversetzte Treffer zurück."""
        logger.debug(
            f"Processing chunk: offset={offset_seconds:.2f}s, duration={len(y_full)/sr_full:.2f}s"
        )

        # KICK (20-150 Hz)
        kick_onset_env = librosa.onset.onset_strength(
            y=y_full, sr=sr_full, hop_length=512, aggregate=np.median, fmin=20, fmax=150, n_mels=32
        )
        kick_frames = librosa.onset.onset_detect(
            onset_envelope=kick_onset_env,
            sr=sr_full,
            hop_length=512,
            backtrack=False,
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=5,
            delta=0.15,
            wait=4,
        )
        kick_times = (
            librosa.frames_to_time(kick_frames, sr=sr_full, hop_length=512) + offset_seconds
        )
        kick_strengths = [
            float(kick_onset_env[fr]) for fr in kick_frames if fr < len(kick_onset_env)
        ]
        if kick_strengths:
            max_kick = max(kick_strengths) if max(kick_strengths) > 0 else 1.0
            kick_strengths = [s / max_kick for s in kick_strengths]

        # SNARE (150-400 Hz + Transient)
        snare_onset_env = librosa.onset.onset_strength(
            y=y_full, sr=sr_full, hop_length=512, aggregate=np.median, fmin=150, fmax=400, n_mels=32
        )
        y_perc = librosa.effects.percussive(y_full, margin=3)
        transient_env = librosa.onset.onset_strength(
            y=y_perc,
            sr=sr_full,
            hop_length=512,
            aggregate=np.median,
            fmin=1000,
            fmax=4000,
            n_mels=32,
        )
        combined_snare = snare_onset_env * 0.6 + transient_env * 0.4
        snare_frames = librosa.onset.onset_detect(
            onset_envelope=combined_snare,
            sr=sr_full,
            hop_length=512,
            backtrack=False,
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=5,
            delta=0.12,
            wait=6,
        )
        snare_times = (
            librosa.frames_to_time(snare_frames, sr=sr_full, hop_length=512) + offset_seconds
        )
        snare_strengths = [
            float(combined_snare[fr]) for fr in snare_frames if fr < len(combined_snare)
        ]
        if snare_strengths:
            max_snare = max(snare_strengths) if max(snare_strengths) > 0 else 1.0
            snare_strengths = [s / max_snare for s in snare_strengths]

        # HIHAT (6000-16000 Hz)
        hihat_onset_env = librosa.onset.onset_strength(
            y=y_full,
            sr=sr_full,
            hop_length=256,
            aggregate=np.median,
            fmin=6000,
            fmax=16000,
            n_mels=64,
        )
        hihat_frames = librosa.onset.onset_detect(
            onset_envelope=hihat_onset_env,
            sr=sr_full,
            hop_length=256,
            backtrack=False,
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=5,
            delta=0.08,
            wait=2,
        )
        hihat_times = (
            librosa.frames_to_time(hihat_frames, sr=sr_full, hop_length=256) + offset_seconds
        )
        hihat_strengths = [
            float(hihat_onset_env[fr]) for fr in hihat_frames if fr < len(hihat_onset_env)
        ]
        if hihat_strengths:
            max_hihat = max(hihat_strengths) if max(hihat_strengths) > 0 else 1.0
            hihat_strengths = [s / max_hihat for s in hihat_strengths]

        return {
            "kick_times": kick_times.tolist(),
            "snare_times": snare_times.tolist(),
            "hihat_times": hihat_times.tolist(),
            "kick_strengths": kick_strengths,
            "snare_strengths": snare_strengths,
            "hihat_strengths": hihat_strengths,
        }

    def _merge_hits(
        self,
        times: list[float],
        strengths: list[float],
        new_times: list[float],
        new_strengths: list[float],
        tolerance: float = 0.05,
    ) -> tuple[list[float], list[float]]:
        """Fügt neue Treffer hinzu und merged Duplikate innerhalb der Toleranz."""
        for t, s in zip(new_times, new_strengths):
            merged = False
            for idx, existing in enumerate(times):
                if abs(existing - t) <= tolerance:
                    strengths[idx] = max(strengths[idx], s)
                    merged = True
                    break
            if not merged:
                times.append(t)
                strengths.append(s)

        # Sortiere nach Zeit
        if times:
            sorted_pairs = sorted(zip(times, strengths), key=lambda x: x[0])
            times, strengths = [list(x) for x in zip(*sorted_pairs)]

        return times, strengths
