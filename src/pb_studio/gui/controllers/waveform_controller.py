"""
Waveform Controller for PB_studio

PERF-05 FIX: Moves waveform loading to background thread to prevent UI blocking.

Uses QThread to run I/O and CPU-intensive waveform operations (audio loading,
downsampling, beat extraction) in a separate thread, emitting signals when complete.
"""

import logging
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, QThread
from PyQt6.QtCore import pyqtSignal as Signal

logger = logging.getLogger(__name__)


class WaveformLoadWorker(QThread):
    """
    Worker thread for waveform loading.

    PERF-05 FIX: Prevents UI blocking during waveform loading by running
    heavy I/O and computations (audio loading, downsampling, beat extraction)
    in background.

    Signals:
        started: Emitted when loading begins
        progress: Emitted with progress percentage and status message
        finished: Emitted with waveform data dict when complete
        error: Emitted with error message if loading fails
    """

    # Signals for thread communication
    started = Signal()
    progress = Signal(int, str)  # (percentage, status_message)
    finished = Signal(dict)  # Results dictionary
    error = Signal(str)  # Error message

    def __init__(self, audio_path: str, target_width: int = 1920, parent: QObject | None = None):
        """
        Initialize the waveform load worker.

        Args:
            audio_path: Path to the audio file to load
            target_width: Target width for downsampling (default: 1920)
            parent: Parent QObject (optional)
        """
        super().__init__(parent)
        self.audio_path = audio_path
        self.target_width = target_width
        self._is_cancelled = False

    def cancel(self):
        """Request cancellation of the loading."""
        self._is_cancelled = True

    def run(self):
        """
        Execute waveform loading in background thread.

        Loads audio, downsamples for display, and extracts beats.
        Emits progress updates and results via signals.
        """
        try:
            self.started.emit()
            self.progress.emit(0, "Loading audio file...")

            # Import here to avoid circular imports
            from ...audio.waveform_analyzer import WaveformAnalyzer
            from ...audio.waveform_cache import WaveformCache

            if self._is_cancelled:
                return

            audio_path = Path(self.audio_path)
            analyzer = WaveformAnalyzer()
            cache = WaveformCache()

            # Step 1: Check cache (0-10%)
            self.progress.emit(5, "Checking cache...")
            cached = cache.get(audio_path, 22050, self.target_width)

            if cached:
                peaks_min, peaks_max, sr = cached
                self.progress.emit(60, "Cache hit - loading beats...")

                # Still need to load for duration/beats
                samples, sr = analyzer.load_audio(audio_path)
                duration = analyzer.get_duration(samples, sr)
            else:
                # Step 2: Load audio file (10-40%)
                self.progress.emit(10, "Loading audio...")
                if self._is_cancelled:
                    return

                samples, sr = analyzer.load_audio(audio_path)
                duration = analyzer.get_duration(samples, sr)

                if self._is_cancelled:
                    return

                self.progress.emit(40, "Audio loaded")

                # Step 3: Downsample for display (40-60%)
                self.progress.emit(45, "Generating waveform...")
                if self._is_cancelled:
                    return

                peaks_min, peaks_max = analyzer.downsample_for_display(
                    samples, target_width=self.target_width
                )

                if self._is_cancelled:
                    return

                self.progress.emit(60, "Waveform generated")

                # Save to cache
                cache.put(audio_path, 22050, self.target_width, peaks_min, peaks_max, sr)

            # Step 4: Extract beats (60-90%)
            self.progress.emit(65, "Extracting beats...")
            if self._is_cancelled:
                return

            beats = analyzer.extract_beats(samples, sr)

            if self._is_cancelled:
                return

            self.progress.emit(90, "Beats extracted")

            # Step 5: Compile results (90-100%)
            self.progress.emit(95, "Finalizing...")

            results: dict[str, Any] = {
                "success": True,
                "audio_path": self.audio_path,
                "peaks_min": peaks_min,
                "peaks_max": peaks_max,
                "sample_rate": sr,
                "duration": duration,
                "beat_times": beats.tolist() if len(beats) > 0 else [],
            }

            logger.info(
                f"Waveform loaded: {audio_path.name}, "
                f"duration={duration:.1f}s, {len(results['beat_times'])} beats"
            )

            self.progress.emit(100, "Complete")
            self.finished.emit(results)

        except Exception as e:
            error_msg = f"Waveform loading failed: {e}"
            logger.error(error_msg, exc_info=True)
            self.error.emit(error_msg)


class WaveformController(QObject):
    """
    Controller for managing waveform loading operations.

    PERF-05 FIX: Provides high-level API for non-blocking waveform loading.
    Manages worker threads and provides convenience signals for UI integration.

    Usage:
        controller = WaveformController(parent_widget)
        controller.load_complete.connect(on_waveform_loaded)
        controller.load(audio_path)
    """

    # High-level signals
    load_started = Signal(str)  # audio_path
    load_progress = Signal(int, str)  # (percentage, status)
    load_complete = Signal(dict)  # results with peaks, duration, beats
    load_error = Signal(str)  # error message

    def __init__(self, parent: QObject | None = None):
        """
        Initialize the waveform controller.

        Args:
            parent: Parent QObject (optional)
        """
        super().__init__(parent)
        self._current_worker: WaveformLoadWorker | None = None

    def load(self, audio_path: str, target_width: int = 1920) -> None:
        """
        Start waveform loading in background thread.

        If loading is already running, it will be cancelled first.

        Args:
            audio_path: Path to audio file to load
            target_width: Target display width for downsampling
        """
        # Cancel any existing load
        self.cancel()

        # Create and start new worker
        self._current_worker = WaveformLoadWorker(audio_path, target_width, self)

        # Connect signals
        self._current_worker.started.connect(lambda: self.load_started.emit(audio_path))
        self._current_worker.progress.connect(self.load_progress.emit)
        self._current_worker.finished.connect(self._on_worker_finished)
        self._current_worker.error.connect(self._on_worker_error)

        # Start the worker
        logger.info(f"Starting background waveform load: {audio_path}")
        self._current_worker.start()

    def cancel(self) -> None:
        """Cancel current loading if running."""
        if self._current_worker and self._current_worker.isRunning():
            logger.info("Cancelling current waveform load")
            self._current_worker.cancel()
            self._current_worker.quit()
            self._current_worker.wait(1000)  # Wait up to 1 second
            self._current_worker = None

    def is_loading(self) -> bool:
        """Check if loading is currently running."""
        return self._current_worker is not None and self._current_worker.isRunning()

    def _on_worker_finished(self, results: dict) -> None:
        """Handle worker completion."""
        self.load_complete.emit(results)
        self._current_worker = None

    def _on_worker_error(self, error_msg: str) -> None:
        """Handle worker error."""
        self.load_error.emit(error_msg)
        self._current_worker = None
