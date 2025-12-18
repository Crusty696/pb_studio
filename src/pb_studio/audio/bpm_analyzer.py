"""BPM und Beatgrid Analyse.

Spezialisiertes Modul für BPM-Detection und Beat-Tracking.
"""

import logging
from typing import TypeAlias

import librosa
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Type aliases
AudioSamples: TypeAlias = NDArray[np.floating]


class BPMAnalyzer:
    """BPM-Detection und Beat-Tracking."""

    def __init__(self, sr: int = 11025) -> None:
        """
        Initialize BPM Analyzer.

        Args:
            sr: Sample rate for audio processing (11025 recommended for beat tracking)
        """
        self.sr = sr

    def detect_bpm(
        self,
        y: AudioSamples,
        sr: int,
        onset_envelope: NDArray | None = None,
        expected_bpm: float = 120.0,
    ) -> tuple[float, NDArray]:
        """
        Detect BPM using onset envelope.

        Args:
            y: Audio samples
            sr: Sample rate
            onset_envelope: Pre-computed onset envelope (optional)
            expected_bpm: Expected BPM for faster convergence

        Returns:
            Tuple of (BPM, beat_frames)
        """
        try:
            # Compute onset envelope if not provided
            if onset_envelope is None:
                onset_envelope = self._compute_onset_envelope(y, sr)

            # BPM detection using onset envelope
            tempo, beat_frames = librosa.beat.beat_track(
                onset_envelope=onset_envelope,
                sr=sr,
                start_bpm=expected_bpm,
                hop_length=512,
                sparse=True,
            )

            return float(tempo), beat_frames

        except Exception as e:
            logger.error(f"BPM detection failed: {e}", exc_info=True)
            return 120.0, np.array([])

    def create_beatgrid(
        self, y: AudioSamples, sr: int, beat_frames: NDArray | None = None
    ) -> list[float]:
        """
        Create beatgrid from beat frames.

        Args:
            y: Audio samples
            sr: Sample rate
            beat_frames: Pre-computed beat frames (optional)

        Returns:
            List of beat times in seconds
        """
        try:
            if beat_frames is None or len(beat_frames) == 0:
                _, beat_frames = self.detect_bpm(y, sr)

            beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)
            return beat_times.tolist()

        except Exception as e:
            logger.error(f"Beatgrid creation failed: {e}", exc_info=True)
            return []

    def refine_bpm(self, y: AudioSamples, sr: int, initial_bpm: float) -> float:
        """
        Refine BPM using autocorrelation.

        Args:
            y: Audio samples
            sr: Sample rate
            initial_bpm: Initial BPM estimate

        Returns:
            Refined BPM
        """
        try:
            # Compute onset envelope
            onset_env = self._compute_onset_envelope(y, sr)

            # Use autocorrelation for refinement
            tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=512)

            # Find dominant tempo from tempogram
            ac_global = librosa.autocorrelate(onset_env, max_size=tempogram.shape[0])
            ac_global = librosa.util.normalize(ac_global)

            # Find peaks
            tempo_frequencies = librosa.tempo_frequencies(tempogram.shape[0], sr=sr, hop_length=512)
            peak_idx = np.argmax(ac_global)

            if peak_idx < len(tempo_frequencies):
                refined_bpm = tempo_frequencies[peak_idx]
            else:
                refined_bpm = initial_bpm

            return float(refined_bpm)

        except Exception as e:
            logger.error(f"BPM refinement failed: {e}", exc_info=True)
            return initial_bpm

    def _compute_onset_envelope(self, y: AudioSamples, sr: int) -> NDArray:
        """
        Compute onset envelope for beat tracking.

        Args:
            y: Audio samples
            sr: Sample rate

        Returns:
            Onset envelope array
        """
        # Optimized parameters (from librosa issue #119)
        nyquist = sr / 2.0
        fmax = min(8000, nyquist - 200)  # 200 Hz margin for safety

        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, aggregate=np.median, fmax=fmax, n_mels=128, hop_length=512
        )

        return onset_env
