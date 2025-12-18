"""
Audio Analysis Controller for PB_studio

PERF-04 FIX: Moves audio analysis to background thread to prevent UI blocking.

Uses QThread to run CPU-intensive audio analysis (BPM detection, beatgrid extraction)
in a separate thread, emitting signals when complete for UI updates.
"""

import logging
from typing import Any, Optional

from PyQt6.QtCore import QObject, QThread
from PyQt6.QtCore import pyqtSignal as Signal

logger = logging.getLogger(__name__)


class AudioAnalysisWorker(QThread):
    """
    Worker thread for audio analysis.

    PERF-04 FIX: Prevents UI blocking during audio analysis by running
    heavy computations (BPM detection, beatgrid, energy curve) in background.

    Signals:
        started: Emitted when analysis begins
        progress: Emitted with progress percentage (0-100)
        finished: Emitted with analysis results dict when complete
        error: Emitted with error message if analysis fails
    """

    # Signals for thread communication
    started = Signal()
    progress = Signal(int, str)  # (percentage, status_message)
    finished = Signal(dict)  # Results dictionary
    error = Signal(str)  # Error message

    def __init__(
        self,
        audio_path: str,
        use_stems: bool = False,
        auto_create_stems: bool = True,
        parent: QObject | None = None,
    ):
        """
        Initialize the audio analysis worker.

        Args:
            audio_path: Path to the audio file to analyze
            use_stems: Whether to use stem-based trigger analysis (more accurate for DJ mixes)
            auto_create_stems: Whether to automatically create stems during import (NEW)
            parent: Parent QObject (optional)
        """
        super().__init__(parent)
        self.audio_path = audio_path
        self.use_stems = use_stems
        self.auto_create_stems = auto_create_stems
        self._is_cancelled = False

    def cancel(self):
        """Request cancellation of the analysis."""
        self._is_cancelled = True

    def run(self):
        """
        Execute audio analysis in background thread.

        Performs BPM detection, beatgrid extraction, and energy curve calculation.
        Emits progress updates and results via signals.
        """
        try:
            self.started.emit()
            self.progress.emit(0, "Starting audio analysis...")

            # Import here to avoid circular imports and keep imports in worker thread
            from ...audio.audio_analyzer import AudioAnalyzer

            if self._is_cancelled:
                return

            # Create analyzer
            analyzer = AudioAnalyzer()

            # Step 1: BPM Analysis (0-40%)
            self.progress.emit(10, "Detecting BPM...")
            if self._is_cancelled:
                return

            bpm_result = analyzer.analyze_bpm(self.audio_path)

            if self._is_cancelled:
                return

            self.progress.emit(40, "BPM detection complete")

            # Step 2: Beatgrid Analysis (40-80%)
            self.progress.emit(50, "Extracting beatgrid...")
            if self._is_cancelled:
                return

            beatgrid_result = analyzer.analyze_beatgrid(self.audio_path)

            if self._is_cancelled:
                return

            self.progress.emit(80, "Beatgrid extraction complete")

            # Step 3: Pre-warm Trigger Analysis Cache (80-95%)
            # PERFORMANCE FIX: Run trigger analysis during import so it's cached for cut list generation
            # NOTE: Stem-Analyse wird SEPARAT nach der Audio-Analyse im StemSeparationDialog gemacht!
            self.progress.emit(85, "Pre-warming trigger cache...")

            try:
                from ...pacing.trigger_system import TriggerSystem

                # Create trigger system with caching - OHNE Stems (die kommen später separat)
                trigger_system = TriggerSystem(use_cache=True, use_stems=False)

                # Run basic trigger analysis (Fullmix-basiert)
                trigger_analysis = trigger_system.analyze_triggers(
                    self.audio_path, expected_bpm=bpm_result.get("bpm") if bpm_result else None
                )

                logger.info(
                    f"Trigger-Cache pre-warmed: {len(trigger_analysis.beat_times)} beats, "
                    f"{len(trigger_analysis.onset_times)} onsets, "
                    f"{len(trigger_analysis.kick_times)} kicks, "
                    f"{len(trigger_analysis.snare_times)} snares, "
                    f"{len(trigger_analysis.hihat_times)} hihats, "
                    f"{len(trigger_analysis.energy_times)} energy peaks"
                )
            except Exception as e:
                # Non-critical - just log warning, analysis will run later if needed
                logger.warning(f"Trigger cache pre-warming failed (non-critical): {e}")

            if self._is_cancelled:
                return

            # Step 4: Auto-Create Stems (80-95%) - NEW!
            stems_info = None
            if self.auto_create_stems:
                try:
                    self.progress.emit(85, "Creating stems automatically...")

                    from ...audio.auto_stem_processor import get_auto_stem_processor

                    processor = get_auto_stem_processor(quality_mode="auto")

                    # Check if stems exist, create if not
                    all_exist, existing_stems = processor.stems_exist(self.audio_path)

                    if all_exist:
                        logger.info(f"Stems already exist for {self.audio_path}")
                        stems_info = {
                            "created": False,
                            "stems": existing_stems,
                            "model_info": processor.get_model_info(),
                        }
                    else:
                        logger.info(f"Creating stems for {self.audio_path}")

                        # Progress callback für Stem-Creation
                        def stem_progress(percent, message):
                            # Map 0-100% stem progress to 85-95% total progress
                            total_percent = 85 + (percent * 0.1)  # 85% + 10% range
                            self.progress.emit(int(total_percent), f"Stems: {message}")

                        stems = processor.create_stems(
                            self.audio_path, progress_callback=stem_progress
                        )

                        stems_info = {
                            "created": True,
                            "stems": stems,
                            "model_info": processor.get_model_info(),
                        }

                        logger.info(f"Stems created successfully: {list(stems.keys())}")

                except Exception as e:
                    # Non-critical - stems can be created later manually
                    logger.warning(f"Auto-stem creation failed (non-critical): {e}")
                    stems_info = {"created": False, "error": str(e), "stems": {}}

            if self._is_cancelled:
                return

            # Step 5: Compile Results (95-100%)
            self.progress.emit(95, "Compiling results...")

            results: dict[str, Any] = {
                "success": False,
                "audio_path": self.audio_path,
                "bpm": None,
                "beat_times": [],
                "energy": None,
                "stems": stems_info,  # NEW: Include stem information
            }

            if bpm_result and beatgrid_result:
                results["success"] = True
                results["bpm"] = bpm_result.get("bpm", 120.0)
                results["beat_times"] = beatgrid_result.get("beat_times", [])
                results["energy"] = beatgrid_result.get("energy")

                logger.info(
                    f"Audio analysis complete: BPM={results['bpm']:.1f}, "
                    f"{len(results['beat_times'])} beats detected"
                )
            else:
                logger.warning("Audio analysis returned incomplete results")

            self.progress.emit(100, "Analysis complete")
            self.finished.emit(results)

        except Exception as e:
            error_msg = f"Audio analysis failed: {e}"
            logger.error(error_msg, exc_info=True)
            self.error.emit(error_msg)


class AudioAnalysisController(QObject):
    """
    Controller for managing audio analysis operations.

    PERF-04 FIX: Provides high-level API for non-blocking audio analysis.
    Manages worker threads and provides convenience signals for UI integration.

    Usage:
        controller = AudioAnalysisController(parent_widget)
        controller.analysis_complete.connect(on_analysis_done)
        controller.analyze(audio_path)
    """

    # High-level signals
    analysis_started = Signal(str)  # audio_path
    analysis_progress = Signal(int, str)  # (percentage, status)
    analysis_complete = Signal(dict)  # results
    analysis_error = Signal(str)  # error message

    def __init__(self, parent: QObject | None = None):
        """
        Initialize the audio analysis controller.

        Args:
            parent: Parent QObject (optional)
        """
        super().__init__(parent)
        self._current_worker: AudioAnalysisWorker | None = None

    def analyze(
        self, audio_path: str, use_stems: bool = False, auto_create_stems: bool = True
    ) -> None:
        """
        Start audio analysis in background thread.

        If analysis is already running, it will be cancelled first.

        Args:
            audio_path: Path to audio file to analyze
            use_stems: Whether to use stem-based trigger analysis (more accurate for DJ mixes)
            auto_create_stems: Whether to automatically create stems during import (NEW)
        """
        # BUG FIX 5: Cancel with proper synchronization to prevent race condition
        self.cancel()

        # Create and start new worker
        # BUG FIX: Now passes use_stems parameter from GUI settings
        # NEW: Auto-create stems during import
        self._current_worker = AudioAnalysisWorker(
            audio_path, use_stems=use_stems, auto_create_stems=auto_create_stems, parent=self
        )

        # Connect signals
        self._current_worker.started.connect(lambda: self.analysis_started.emit(audio_path))
        self._current_worker.progress.connect(self.analysis_progress.emit)
        self._current_worker.finished.connect(self._on_worker_finished)
        self._current_worker.error.connect(self._on_worker_error)

        # BUG FIX 5: Add cleanup connection to prevent worker leak
        self._current_worker.finished.connect(self._current_worker.deleteLater)
        self._current_worker.error.connect(self._current_worker.deleteLater)

        # Start the worker
        logger.info(f"Starting background audio analysis: {audio_path}")
        self._current_worker.start()

    def cancel(self) -> None:
        """Cancel current analysis if running."""
        # BUG FIX 5: Improved cancellation with proper cleanup and synchronization
        if self._current_worker and self._current_worker.isRunning():
            logger.info("Cancelling current audio analysis")

            # Request cancellation
            self._current_worker.cancel()

            # Disconnect signals to prevent late callbacks
            try:
                self._current_worker.finished.disconnect(self._on_worker_finished)
                self._current_worker.error.disconnect(self._on_worker_error)
            except TypeError:
                # Signals may already be disconnected
                pass

            # Request thread to quit
            self._current_worker.quit()

            # Wait with timeout
            if not self._current_worker.wait(2000):  # 2 second timeout
                logger.warning("Audio analysis worker did not finish in time, forcing termination")
                self._current_worker.terminate()
                self._current_worker.wait()

            # Schedule for deletion
            self._current_worker.deleteLater()
            self._current_worker = None

    def is_analyzing(self) -> bool:
        """Check if analysis is currently running."""
        return self._current_worker is not None and self._current_worker.isRunning()

    def _on_worker_finished(self, results: dict) -> None:
        """Handle worker completion."""
        self.analysis_complete.emit(results)
        self._current_worker = None

    def _on_worker_error(self, error_msg: str) -> None:
        """Handle worker error."""
        self.analysis_error.emit(error_msg)
        self._current_worker = None
