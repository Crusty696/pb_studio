"""Spectral Features Analyse.

Spezialisiertes Modul für spektrale Merkmalsextraktion.
"""

import logging
from typing import TypeAlias

import librosa
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Type aliases
AudioSamples: TypeAlias = NDArray[np.floating]


class SpectralAnalyzer:
    """Spectral Features Analysis."""

    def __init__(self, sr: int = 11025) -> None:
        """
        Initialize Spectral Analyzer.

        Args:
            sr: Sample rate for audio processing
        """
        self.sr = sr

    def extract_spectral_features(self, y: AudioSamples, sr: int) -> dict:
        """
        Extract spectral features for audio-visual mapping.

        Computes features useful for matching audio characteristics to visual content:
        - Spectral centroid (brightness)
        - Spectral bandwidth (richness)
        - Spectral rolloff (high-frequency content)
        - MFCC (timbral characteristics)
        - Chroma features (harmonic content)
        - RMS energy (loudness)

        Args:
            y: Audio samples
            sr: Sample rate

        Returns:
            Dictionary of spectral features
        """
        try:
            hop_length = 512
            frame_length = 2048

            # Spectral centroid (brightness) - higher = brighter
            spectral_centroid = librosa.feature.spectral_centroid(
                y=y, sr=sr, hop_length=hop_length
            )[0]

            # Spectral bandwidth (richness) - higher = richer
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=y, sr=sr, hop_length=hop_length
            )[0]

            # Spectral rolloff (high-frequency content)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[
                0
            ]

            # MFCC (timbral fingerprint) - 13 coefficients
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)

            # Chroma features (harmonic/pitch content)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)

            # RMS energy (loudness)
            rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

            # Zero-crossing rate (noisiness/percussiveness)
            zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)[0]

            # Compute frame times
            frame_times = librosa.frames_to_time(
                np.arange(len(spectral_centroid)), sr=sr, hop_length=hop_length
            )

            # Aggregate statistics for overall audio character
            result = {
                # Frame-level features (for temporal mapping)
                "frame_times": frame_times.tolist(),
                "spectral_centroid": spectral_centroid.tolist(),
                "spectral_bandwidth": spectral_bandwidth.tolist(),
                "spectral_rolloff": spectral_rolloff.tolist(),
                "rms_energy": rms.tolist(),
                "zero_crossing_rate": zcr.tolist(),
                # Summary statistics
                "mean_centroid": float(np.mean(spectral_centroid)),
                "mean_bandwidth": float(np.mean(spectral_bandwidth)),
                "mean_rolloff": float(np.mean(spectral_rolloff)),
                "mean_energy": float(np.mean(rms)),
                "energy_variance": float(np.var(rms)),
                # MFCC summary (first 4 coefficients are most important)
                "mfcc_means": [float(np.mean(mfcc[i])) for i in range(min(4, len(mfcc)))],
                "mfcc_stds": [float(np.std(mfcc[i])) for i in range(min(4, len(mfcc)))],
                # Chroma summary (dominant pitch classes)
                "chroma_means": [float(np.mean(chroma[i])) for i in range(12)],
                # Derived characteristics
                "brightness": float(np.mean(spectral_centroid) / (sr / 2)),  # Normalized 0-1
                "richness": float(np.mean(spectral_bandwidth) / (sr / 2)),  # Normalized 0-1
                "percussiveness": float(np.mean(zcr)),  # Higher = more percussive
            }

            logger.info(
                f"Spectral features extracted: brightness={result['brightness']:.3f}, "
                f"richness={result['richness']:.3f}, percussiveness={result['percussiveness']:.3f}"
            )
            return result

        except Exception as e:
            logger.error(f"Spectral feature extraction failed: {e}", exc_info=True)
            return {}

    def compute_spectral_centroid(self, y: AudioSamples, sr: int) -> np.ndarray:
        """
        Compute spectral centroid (brightness).

        Args:
            y: Audio samples
            sr: Sample rate

        Returns:
            Spectral centroid array
        """
        try:
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
            return centroid
        except Exception as e:
            logger.error(f"Spectral centroid computation failed: {e}", exc_info=True)
            return np.array([])

    def compute_spectral_rolloff(self, y: AudioSamples, sr: int) -> np.ndarray:
        """
        Compute spectral rolloff (high-frequency content).

        Args:
            y: Audio samples
            sr: Sample rate

        Returns:
            Spectral rolloff array
        """
        try:
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=512)[0]
            return rolloff
        except Exception as e:
            logger.error(f"Spectral rolloff computation failed: {e}", exc_info=True)
            return np.array([])

    def compute_chroma_features(self, y: AudioSamples, sr: int) -> np.ndarray:
        """
        Compute chroma features (harmonic content).

        Args:
            y: Audio samples
            sr: Sample rate

        Returns:
            Chroma feature array (12 pitch classes)
        """
        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
            return chroma
        except Exception as e:
            logger.error(f"Chroma feature computation failed: {e}", exc_info=True)
            return np.zeros((12, 1))

    def compute_mfcc(self, y: AudioSamples, sr: int, n_mfcc: int = 13) -> np.ndarray:
        """
        Compute MFCC (timbral fingerprint).

        Args:
            y: Audio samples
            sr: Sample rate
            n_mfcc: Number of MFCC coefficients

        Returns:
            MFCC array
        """
        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=512)
            return mfcc
        except Exception as e:
            logger.error(f"MFCC computation failed: {e}", exc_info=True)
            return np.zeros((n_mfcc, 1))
