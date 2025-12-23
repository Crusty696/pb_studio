"""
PB_studio Energy Curve Control for Pacing (Task 28)

Analyzes audio energy to drive video pacing decisions. Maps audio intensity
to cut frequency, density, and timing for energy-driven editing workflows.

Features:
- RMS energy extraction with customizable window sizes
- Spectral centroid analysis for brightness/intensity
- Energy curve smoothing and normalization
- Energy-to-pacing mapping (energy → cut density)
- Integration with PacingEngine for automatic cut generation
- Energy zone detection (low/medium/high energy sections)
- Peak detection for cut trigger points

Usage:
    from pb_studio.pacing import EnergyAnalyzer, EnergyBasedPacingEngine

    # Analyze audio energy
    analyzer = EnergyAnalyzer()
    energy_curve = analyzer.analyze_energy(
        audio_path="track.mp3",
        window_size=0.5,  # 500ms windows
        smoothing=True
    )

    # Create energy-based pacing
    engine = EnergyBasedPacingEngine(
        energy_curve=energy_curve,
        bpm=140.0,
        energy_mode="adaptive"  # high/medium/low → fast/medium/slow cuts
    )

    # Generate cuts from energy
    cuts = engine.generate_energy_driven_cuts(
        clip_pool=["clip_01", "clip_02", "clip_03"],
        min_cut_duration=0.5,
        max_cut_duration=4.0
    )
"""

# Standard library
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

# Third-party
import librosa
import numpy as np
from numpy.typing import NDArray
from scipy import signal

from .pacing_engine import PacingEngine

# Local
from .pacing_models import AudioTrackReference, CutListEntry, PacingBlueprint

logger = logging.getLogger(__name__)

# Type aliases
FloatArray = NDArray[np.floating]
EnergyMode = Literal["adaptive", "proportional", "threshold"]
EnergyZone = Literal["low", "medium", "high"]


class SmoothingMethod(Enum):
    """Energy curve smoothing methods."""

    SAVGOL = "savitzky_golay"  # Savitzky-Golay filter (preserves peaks)
    GAUSSIAN = "gaussian"  # Gaussian smoothing (very smooth)
    MEDIAN = "median"  # Median filter (removes outliers)
    NONE = "none"  # No smoothing


@dataclass
class EnergyCurveData:
    """
    Time-series energy data extracted from audio.

    Attributes:
        times: Timeline positions in seconds
        energy_values: Normalized energy values (0.0-1.0)
        raw_energy: Unnormalized energy values
        window_size: Analysis window size in seconds
        sample_rate: Audio sample rate
        duration: Total audio duration
        smoothing_method: Smoothing method applied
        spectral_centroids: Optional spectral centroid values (brightness)
    """

    times: list[float]
    energy_values: list[float]
    raw_energy: list[float]
    window_size: float
    sample_rate: int
    duration: float
    smoothing_method: str = "none"
    spectral_centroids: list[float] | None = None

    def __post_init__(self):
        """Validate energy curve data."""
        if len(self.times) != len(self.energy_values):
            raise ValueError(
                f"times ({len(self.times)}) and energy_values ({len(self.energy_values)}) "
                "must have same length"
            )

        if self.spectral_centroids and len(self.spectral_centroids) != len(self.times):
            raise ValueError("spectral_centroids must have same length as times/energy_values")

    @property
    def num_frames(self) -> int:
        """Number of energy frames."""
        return len(self.times)

    @property
    def mean_energy(self) -> float:
        """Mean energy value."""
        return float(np.mean(self.energy_values))

    @property
    def max_energy(self) -> float:
        """Maximum energy value."""
        return float(np.max(self.energy_values))

    @property
    def min_energy(self) -> float:
        """Minimum energy value."""
        return float(np.min(self.energy_values))

    def get_energy_at_time(self, time: float) -> float:
        """
        Get interpolated energy value at specific time.

        Args:
            time: Time in seconds

        Returns:
            Energy value (0.0-1.0)
        """
        if time < 0 or time > self.duration:
            raise ValueError(f"Time {time}s out of range (0-{self.duration}s)")

        # Find nearest frame
        idx = np.searchsorted(self.times, time)

        # Boundary cases
        if idx == 0:
            return self.energy_values[0]
        if idx >= len(self.energy_values):
            return self.energy_values[-1]

        # Linear interpolation
        t0, t1 = self.times[idx - 1], self.times[idx]
        e0, e1 = self.energy_values[idx - 1], self.energy_values[idx]

        # FIX: Prevent Division by Zero when duplicate times exist
        time_diff = t1 - t0
        if time_diff < 1e-9:
            return e0  # Return first value for duplicates
        weight = (time - t0) / time_diff
        return e0 + weight * (e1 - e0)

    def get_energy_zone(
        self, time: float, thresholds: tuple[float, float] = (0.33, 0.66)
    ) -> EnergyZone:
        """
        Determine energy zone at time (low/medium/high).

        Args:
            time: Time in seconds
            thresholds: (low_threshold, high_threshold) for zone boundaries

        Returns:
            Energy zone classification
        """
        energy = self.get_energy_at_time(time)
        low_threshold, high_threshold = thresholds

        if energy < low_threshold:
            return "low"
        elif energy < high_threshold:
            return "medium"
        else:
            return "high"

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"EnergyCurveData(frames={self.num_frames}, "
            f"duration={self.duration:.2f}s, window={self.window_size:.3f}s, "
            f"mean_energy={self.mean_energy:.3f}, smoothing={self.smoothing_method})"
        )


class EnergyAnalyzer:
    """
    Analyzes audio to extract energy curves for pacing control.

    Provides RMS energy, spectral analysis, and energy curve smoothing
    for integration with pacing engines.
    """

    def __init__(
        self,
        sr: int = 22050,
        window_size: float = 0.5,
        hop_size: float = 0.1,
        smoothing: SmoothingMethod = SmoothingMethod.SAVGOL,
    ):
        """
        Initialize energy analyzer.

        Args:
            sr: Sample rate for audio loading
            window_size: Analysis window size in seconds (default: 0.5s)
            hop_size: Hop size between windows in seconds (default: 0.1s)
            smoothing: Default smoothing method
        """
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if hop_size <= 0:
            raise ValueError(f"hop_size must be positive, got {hop_size}")
        if hop_size > window_size:
            raise ValueError(f"hop_size ({hop_size}) cannot exceed window_size ({window_size})")

        self.sr = sr
        self.default_window_size = window_size
        self.default_hop_size = hop_size
        self.default_smoothing = smoothing

        logger.info(
            f"EnergyAnalyzer initialized: sr={sr}, "
            f"window={window_size}s, hop={hop_size}s, smoothing={smoothing.value}"
        )

    def analyze_energy(
        self,
        audio_path: str | Path,
        window_size: float | None = None,
        hop_size: float | None = None,
        smoothing: SmoothingMethod | None = None,
        include_spectral: bool = False,
        normalize: bool = True,
    ) -> EnergyCurveData | None:
        """
        Extract energy curve from audio file.

        Args:
            audio_path: Path to audio file
            window_size: Analysis window size in seconds (default: instance default)
            hop_size: Hop size in seconds (default: instance default)
            smoothing: Smoothing method (default: instance default)
            include_spectral: Include spectral centroid analysis
            normalize: Normalize energy to 0.0-1.0 range

        Returns:
            Energy curve data or None on error
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            logger.error(f"Audio file not found: {audio_path}")
            return None

        # Use defaults if not specified
        window_size = window_size or self.default_window_size
        hop_size = hop_size or self.default_hop_size
        smoothing = smoothing or self.default_smoothing

        try:
            logger.info(f"Energy analysis starting: {audio_path.name}")

            # Load audio
            y, sr = librosa.load(str(audio_path), sr=self.sr, mono=True, dtype=np.float32)

            if len(y) == 0:
                logger.error(f"Audio file is empty: {audio_path}")
                return None

            # Calculate frame parameters
            frame_length = int(window_size * sr)
            hop_length = int(hop_size * sr)

            # Extract RMS energy
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

            # Calculate time points
            duration = librosa.get_duration(y=y, sr=sr)
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

            # Store raw energy
            raw_energy = rms.copy()

            # Apply smoothing
            if smoothing != SmoothingMethod.NONE:
                rms = self._smooth_energy(rms, smoothing)

            # Normalize to 0-1 range
            if normalize:
                energy_max = np.max(rms)
                if energy_max > 0:
                    energy_normalized = rms / energy_max
                else:
                    energy_normalized = rms
            else:
                energy_normalized = rms

            # Optional spectral analysis
            spectral_centroids = None
            if include_spectral:
                centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]

                # Normalize centroids to 0-1
                centroid_max = np.max(centroids)
                if centroid_max > 0:
                    spectral_centroids = (centroids / centroid_max).tolist()
                else:
                    spectral_centroids = centroids.tolist()

            # Create energy curve data
            curve = EnergyCurveData(
                times=times.tolist(),
                energy_values=energy_normalized.tolist(),
                raw_energy=raw_energy.tolist(),
                window_size=window_size,
                sample_rate=sr,
                duration=duration,
                smoothing_method=smoothing.value,
                spectral_centroids=spectral_centroids,
            )

            logger.info(
                f"Energy analysis complete: {curve.num_frames} frames, "
                f"mean_energy={curve.mean_energy:.3f}, duration={duration:.2f}s"
            )

            return curve

        except Exception as e:
            logger.error(f"Energy analysis failed: {e}", exc_info=True)
            return None

    def _smooth_energy(self, energy: FloatArray, method: SmoothingMethod) -> FloatArray:
        """
        Apply smoothing to energy curve.

        Args:
            energy: Raw energy values
            method: Smoothing method

        Returns:
            Smoothed energy values
        """
        if method == SmoothingMethod.NONE:
            return energy

        elif method == SmoothingMethod.SAVGOL:
            # Savitzky-Golay filter (preserves peaks)
            window_length = min(51, len(energy) if len(energy) % 2 == 1 else len(energy) - 1)
            if window_length < 5:
                return energy
            return signal.savgol_filter(energy, window_length, 3)

        elif method == SmoothingMethod.GAUSSIAN:
            # Gaussian smoothing
            sigma = 3.0
            return signal.gaussian_filter1d(energy, sigma)

        elif method == SmoothingMethod.MEDIAN:
            # Median filter (removes outliers)
            kernel_size = min(11, len(energy))
            return signal.medfilt(energy, kernel_size)

        else:
            logger.warning(f"Unknown smoothing method: {method}, returning raw energy")
            return energy

    def detect_peaks(
        self, curve: EnergyCurveData, prominence: float = 0.1, min_distance: float = 0.5
    ) -> list[float]:
        """
        Detect energy peaks for potential cut points.

        Args:
            curve: Energy curve data
            prominence: Minimum peak prominence (0.0-1.0)
            min_distance: Minimum time between peaks in seconds

        Returns:
            List of peak times in seconds
        """
        if prominence < 0 or prominence > 1:
            raise ValueError(f"prominence must be 0-1, got {prominence}")

        # Convert min_distance to frame count
        min_distance_frames = int(min_distance / curve.window_size)

        # Find peaks
        peaks, _ = signal.find_peaks(
            curve.energy_values, prominence=prominence, distance=max(1, min_distance_frames)
        )

        # Convert to times
        peak_times = [curve.times[idx] for idx in peaks]

        logger.debug(f"Detected {len(peak_times)} energy peaks (prominence={prominence})")

        return peak_times

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"EnergyAnalyzer(sr={self.sr}, window={self.default_window_size}s, "
            f"hop={self.default_hop_size}s, smoothing={self.default_smoothing.value})"
        )


class EnergyBasedPacingEngine:
    """
    Pacing engine driven by audio energy curves.

    Maps energy levels to cut density and timing for automatic
    energy-driven video editing workflows.
    """

    def __init__(
        self,
        energy_curve: EnergyCurveData,
        bpm: float = 120.0,
        beatgrid_offset: float = 0.0,
        time_signature: int = 4,
        energy_mode: EnergyMode = "adaptive",
    ):
        """
        Initialize energy-based pacing engine.

        Args:
            energy_curve: Analyzed energy curve data
            bpm: BPM for beat synchronization
            beatgrid_offset: Beatgrid offset in seconds
            time_signature: Time signature (beats per bar)
            energy_mode: Energy-to-pacing mapping mode
        """
        self.energy_curve = energy_curve
        self.energy_mode = energy_mode

        # Create underlying PacingEngine for beatgrid support
        self.pacing_engine = PacingEngine(
            bpm=bpm,
            beatgrid_offset=beatgrid_offset,
            time_signature=time_signature,
            total_duration=energy_curve.duration,
        )

        logger.info(
            f"EnergyBasedPacingEngine initialized: mode={energy_mode}, "
            f"BPM={bpm}, duration={energy_curve.duration:.2f}s"
        )

    def generate_energy_driven_cuts(
        self,
        clip_pool: list[str],
        min_cut_duration: float = 0.5,
        max_cut_duration: float = 4.0,
        snap_to_beat: bool = True,
        energy_thresholds: tuple[float, float] = (0.33, 0.66),
    ) -> list[CutListEntry]:
        """
        Generate cuts based on energy curve analysis.

        Energy zones determine cut duration:
        - High energy (>0.66): Short cuts (min_cut_duration)
        - Medium energy (0.33-0.66): Medium cuts ((min+max)/2)
        - Low energy (<0.33): Long cuts (max_cut_duration)

        Args:
            clip_pool: List of available clip IDs
            min_cut_duration: Minimum cut duration (high energy)
            max_cut_duration: Maximum cut duration (low energy)
            snap_to_beat: Snap cut times to beatgrid
            energy_thresholds: (low, high) thresholds for energy zones

        Returns:
            List of validated CutListEntry objects

        Raises:
            ValueError: If clip_pool is empty or duration constraints invalid
        """
        if not clip_pool:
            raise ValueError("clip_pool cannot be empty")

        if min_cut_duration <= 0:
            raise ValueError(f"min_cut_duration must be positive, got {min_cut_duration}")

        if max_cut_duration < min_cut_duration:
            raise ValueError(
                f"max_cut_duration ({max_cut_duration}) must be >= "
                f"min_cut_duration ({min_cut_duration})"
            )

        logger.info(f"Generating energy-driven cuts: mode={self.energy_mode}")

        cuts: list[CutListEntry] = []
        current_time = 0.0
        clip_index = 0
        max_iterations = 10000  # Sicherheitslimit gegen Endlosschleifen
        iteration_count = 0

        while current_time < self.energy_curve.duration and iteration_count < max_iterations:
            iteration_count += 1
            # Get energy zone at current time
            zone = self.energy_curve.get_energy_zone(current_time, energy_thresholds)

            # Map energy zone to cut duration
            if zone == "high":
                duration = min_cut_duration
            elif zone == "medium":
                duration = (min_cut_duration + max_cut_duration) / 2
            else:  # low
                duration = max_cut_duration

            # Apply beat snapping if enabled
            if snap_to_beat:
                current_time = self.pacing_engine.snap_to_beat(current_time, mode="beat")
                end_time = self.pacing_engine.snap_to_beat(current_time + duration, mode="beat")
            else:
                end_time = current_time + duration

            # Clamp to total duration
            if end_time > self.energy_curve.duration:
                end_time = self.energy_curve.duration

            # FIX: Garantierter Fortschritt - mindestens min_cut_duration voranschreiten
            if end_time <= current_time:
                logger.warning(
                    f"Kein Fortschritt bei Zeit {current_time:.3f}s erkannt, "
                    f"erzwinge Fortschritt um min_cut_duration"
                )
                end_time = current_time + min_cut_duration
                if end_time > self.energy_curve.duration:
                    break  # Am Ende angekommen

            # Skip if too short
            if end_time - current_time < 0.1:
                logger.debug(f"Cut zu kurz ({end_time - current_time:.3f}s), überspringe")
                current_time = end_time  # Trotzdem voranschreiten!
                continue

            # Select clip from pool (cycle through)
            clip_id = clip_pool[clip_index % len(clip_pool)]
            clip_index += 1

            # Create cut
            cut = CutListEntry(clip_id=clip_id, start_time=current_time, end_time=end_time)
            cuts.append(cut)

            logger.debug(
                f"Cut {len(cuts)}: {cut.clip_id} @ {cut.start_time:.2f}s-{cut.end_time:.2f}s "
                f"(zone={zone}, duration={cut.duration:.2f}s)"
            )

            # Advance time
            current_time = end_time

        # Warnung bei Erreichen des Sicherheitslimits
        if iteration_count >= max_iterations:
            logger.warning(
                f"Sicherheitslimit von {max_iterations} Iterationen erreicht - "
                f"Schleife abgebrochen bei Zeit {current_time:.3f}s"
            )

        logger.info(f"Generated {len(cuts)} energy-driven cuts")

        return cuts

    def generate_blueprint(
        self,
        name: str,
        clip_pool: list[str],
        description: str | None = None,
        audio_track_id: str | None = None,
        min_cut_duration: float = 0.5,
        max_cut_duration: float = 4.0,
        snap_to_beat: bool = True,
    ) -> PacingBlueprint:
        """
        Generate complete pacing blueprint from energy curve.

        Args:
            name: Blueprint name
            clip_pool: List of available clip IDs
            description: Optional blueprint description
            audio_track_id: Optional audio track identifier
            min_cut_duration: Minimum cut duration
            max_cut_duration: Maximum cut duration
            snap_to_beat: Snap cuts to beatgrid

        Returns:
            Validated PacingBlueprint

        Raises:
            ValueError: If no cuts generated
        """
        # Generate cuts
        cuts = self.generate_energy_driven_cuts(
            clip_pool=clip_pool,
            min_cut_duration=min_cut_duration,
            max_cut_duration=max_cut_duration,
            snap_to_beat=snap_to_beat,
        )

        if not cuts:
            raise ValueError("No cuts generated from energy curve")

        # Create audio track reference
        audio_track = None
        if audio_track_id:
            audio_track = AudioTrackReference(
                track_id=audio_track_id,
                bpm=self.pacing_engine.beatgrid.bpm,
                beatgrid_offset=self.pacing_engine.beatgrid.beatgrid_offset,
            )

        # Build description with energy info
        if description is None:
            description = (
                f"Energy-driven edit ({self.energy_mode} mode): "
                f"{len(cuts)} cuts, mean_energy={self.energy_curve.mean_energy:.3f}"
            )

        # Create blueprint
        blueprint = PacingBlueprint(
            name=name,
            cuts=cuts,
            total_duration=self.energy_curve.duration,
            audio_track=audio_track,
            description=description,
        )

        logger.info(f"Energy-driven blueprint created: {len(cuts)} cuts")

        return blueprint

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"EnergyBasedPacingEngine(mode={self.energy_mode}, "
            f"BPM={self.pacing_engine.beatgrid.bpm:.1f}, "
            f"duration={self.energy_curve.duration:.2f}s)"
        )
