"""
Waveform Analyzer für PB_studio

Analysiert Audio-Dateien und generiert Waveform-Daten für Visualisierung.

Author: PB_studio Development Team
"""

import logging
from pathlib import Path

import librosa
import numpy as np

logger = logging.getLogger(__name__)


class WaveformAnalyzer:
    """
    Analyzes audio files for waveform visualization.

    Features:
        - Load audio with librosa
        - Downsample for display performance
        - Calculate peak min/max for rendering
        - Support various audio formats

    Example:
        >>> analyzer = WaveformAnalyzer()
        >>> samples, sr = analyzer.load_audio("audio.wav")
        >>> peaks_min, peaks_max = analyzer.downsample_for_display(samples, 1920)
    """

    def __init__(self, sample_rate: int = 22050):
        """
        Initialize Waveform Analyzer.

        Args:
            sample_rate: Target sample rate for loading (default: 22050 Hz)
        """
        self.sample_rate = sample_rate
        logger.info(f"WaveformAnalyzer initialized: sample_rate={sample_rate} Hz")

    def load_audio(self, audio_path: str | Path, sr: int | None = None) -> tuple[np.ndarray, int]:
        """
        Load audio file.

        Args:
            audio_path: Path to audio file
            sr: Sample rate (uses instance default if None)

        Returns:
            Tuple of (samples, sample_rate)

        Raises:
            FileNotFoundError: If audio file doesn't exist

        Example:
            >>> analyzer = WaveformAnalyzer()
            >>> samples, sr = analyzer.load_audio("music.wav")
            >>> print(f"Duration: {len(samples) / sr:.2f}s")
        """
        sr = sr or self.sample_rate
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Loading audio: {audio_path.name}")
        samples, actual_sr = librosa.load(str(audio_path), sr=sr, mono=True)
        logger.info(f"Loaded {len(samples)} samples at {actual_sr} Hz")

        return samples, actual_sr

    def downsample_for_display(
        self, samples: np.ndarray, target_width: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Downsample audio for display.

        Calculates peak min/max for each pixel width to preserve
        waveform detail while reducing data for fast rendering.

        Args:
            samples: Audio samples
            target_width: Target width in pixels

        Returns:
            Tuple of (peak_min, peak_max) arrays

        Example:
            >>> samples = np.random.randn(100000)
            >>> peaks_min, peaks_max = analyzer.downsample_for_display(samples, 1920)
            >>> print(peaks_min.shape)
            (1920,)

        Note:
            When len(samples) <= target_width, returns two copies of the sample array.
            The caller should handle this case by rendering the raw samples directly.

            BUGFIX #1: Return separate copies (not same array reference) to prevent
            accidental mutations affecting both min/max. This defensive programming
            approach prevents subtle bugs if caller modifies the returned arrays.
        """
        if len(samples) <= target_width:
            # BUGFIX #1: Return separate copies for defensive programming
            # Prevents accidental mutations from affecting both min and max arrays
            return samples.copy(), samples.copy()

        samples_per_pixel = len(samples) // target_width
        peak_min = np.zeros(target_width)
        peak_max = np.zeros(target_width)

        for i in range(target_width):
            start_idx = i * samples_per_pixel
            end_idx = start_idx + samples_per_pixel
            chunk = samples[start_idx:end_idx]

            if len(chunk) > 0:
                peak_min[i] = np.min(chunk)
                peak_max[i] = np.max(chunk)

        logger.debug(f"Downsampled {len(samples)} samples to {target_width} peaks")
        return peak_min, peak_max

    def get_duration(self, samples: np.ndarray, sr: int) -> float:
        """
        Get audio duration in seconds.

        Args:
            samples: Audio samples
            sr: Sample rate

        Returns:
            Duration in seconds
        """
        return len(samples) / sr

    def extract_beats(self, samples: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract beat positions from audio.

        Args:
            samples: Audio samples
            sr: Sample rate

        Returns:
            Array of beat times in seconds

        Example:
            >>> samples, sr = analyzer.load_audio("music.wav")
            >>> beats = analyzer.extract_beats(samples, sr)
            >>> print(f"Found {len(beats)} beats")
        """
        try:
            tempo, beat_frames = librosa.beat.beat_track(y=samples, sr=sr)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            logger.info(f"Extracted {len(beat_times)} beats, tempo={tempo:.1f} BPM")
            return beat_times
        except Exception as e:
            logger.warning(f"Beat extraction failed: {e}")
            return np.array([])

    def get_rms_energy(
        self, samples: np.ndarray, frame_length: int = 2048, hop_length: int = 512
    ) -> np.ndarray:
        """
        Calculate RMS energy over time.

        Args:
            samples: Audio samples
            frame_length: Frame length for RMS calculation
            hop_length: Hop length between frames

        Returns:
            RMS energy values
        """
        rms = librosa.feature.rms(y=samples, frame_length=frame_length, hop_length=hop_length)[0]
        return rms
