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
    
    PERF-04 FIX: Prevents UI blocking during audio analysis.
    NOW DECOUPLED: Does NOT perform stem separation anymore.
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
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self.audio_path = audio_path
        self.use_stems = use_stems
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            self.started.emit()
            self.progress.emit(0, "Starting basic audio analysis...")

            from ...audio.audio_analyzer import AudioAnalyzer

            if self._is_cancelled:
                return

            analyzer = AudioAnalyzer()

            # Step 1: BPM Analysis (0-30%)
            self.progress.emit(10, "Detecting BPM...")
            bpm_result = analyzer.analyze_bpm(self.audio_path)
            if self._is_cancelled: return

            self.progress.emit(30, "BPM detection complete")

            # Step 2: Beatgrid Analysis (30-60%)
            self.progress.emit(40, "Extracting beatgrid...")
            beatgrid_result = analyzer.analyze_beatgrid(self.audio_path)
            if self._is_cancelled: return

            self.progress.emit(60, "Beatgrid extraction complete")

            # Step 3: Trigger/Structure Analysis (60-100%)
            self.progress.emit(70, " Analyzing song structure...")
            
            # Run structure analysis
            structure_result = analyzer.analyze_structure(self.audio_path)
            
            # Pre-warm Trigger Cache
            self.progress.emit(85, "Pre-warming trigger cache...")
            try:
                from ...pacing.trigger_system import TriggerSystem
                trigger_system = TriggerSystem(use_cache=True, use_stems=False)
                trigger_system.analyze_triggers(
                    self.audio_path, expected_bpm=bpm_result.get("bpm") if bpm_result else None
                )
            except Exception as e:
                logger.warning(f"Trigger cache pre-warming failed (non-critical): {e}")

            if self._is_cancelled: return

            # Compile Results
            self.progress.emit(95, "Compiling results...")
            
            results = {
                "success": False,
                "audio_path": self.audio_path,
                "bpm": None,
                "beat_times": [],
                "duration": 0,
                "structure": structure_result,
                # Stem info is now handled separately, but we leave a placeholder implies "not checked yet"
                "stems": None 
            }

            if bpm_result and beatgrid_result:
                results["success"] = True
                results["bpm"] = bpm_result.get("bpm", 120.0)
                results["beat_times"] = beatgrid_result.get("beat_times", [])
                results["duration"] = bpm_result.get("duration", 0)
            
            self.progress.emit(100, "Basic analysis complete")
            self.finished.emit(results)

        except Exception as e:
            logger.error(f"Audio analysis failed: {e}", exc_info=True)
            self.error.emit(str(e))


class StemSeparationWorker(QThread):
    """
    Dedicated worker for Stem Separation.
    Runs AFTER basic analysis to provide clear progress feedback.
    """
    started = Signal()
    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, audio_path: str, parent: QObject | None = None):
        super().__init__(parent)
        self.audio_path = audio_path
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        self.started.emit()
        self.progress.emit(0, "Initializing stem separator...")
        
        try:
            from ...audio.auto_stem_processor import get_auto_stem_processor
            
            # Check for cancellation
            if self._is_cancelled: return

            processor = get_auto_stem_processor(quality_mode="auto")
            
            # Check if exists first
            all_exist, existing_stems = processor.stems_exist(self.audio_path)
            
            if all_exist:
                self.progress.emit(100, "Stems already exist")
                self.finished.emit({
                    "created": False,
                    "stems": existing_stems,
                    "model_info": processor.get_model_info()
                })
                return

            # Define progress callback
            def on_progress(percent, msg):
                if not self._is_cancelled:
                    self.progress.emit(int(percent), msg)

            self.progress.emit(10, "Starting separation (this may take a while)...")
            
            stems = processor.create_stems(
                self.audio_path, 
                progress_callback=on_progress
            )

            if self._is_cancelled: return

            self.progress.emit(100, "Stem separation complete")
            self.finished.emit({
                "created": True,
                "stems": stems,
                "model_info": processor.get_model_info()
            })

        except Exception as e:
            logger.error(f"Stem separation failed: {e}", exc_info=True)
            self.error.emit(f"Stem separation failed: {e}")


class AudioAnalysisController(QObject):
    """
    Controller for managing audio analysis AND stem separation.
    Chains the two operations sequentially for better UX.
    """

    # Analysis Signals
    analysis_started = Signal(str)
    analysis_progress = Signal(int, str)
    analysis_complete = Signal(dict)
    analysis_error = Signal(str)

    # Stem Signals
    stem_started = Signal(str)
    stem_progress = Signal(int, str)
    stem_complete = Signal(dict)
    stem_error = Signal(str)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._analysis_worker: AudioAnalysisWorker | None = None
        self._stem_worker: StemSeparationWorker | None = None
        # State to track if we should auto-start stems
        self._pending_stem_creation: dict[str, bool] = {} 

    def analyze(
        self, audio_path: str, use_stems: bool = False, auto_create_stems: bool = True
    ) -> None:
        """
        Start full analysis workflow.
        1. Basic Analysis
        2. If auto_create_stems is True -> Start Stem Separation on completion
        """
        self.cancel_all()

        # Store intent
        self._pending_stem_creation[audio_path] = auto_create_stems

        # Start Analysis Worker
        self._analysis_worker = AudioAnalysisWorker(
            audio_path, use_stems=use_stems, parent=self
        )
        
        self._analysis_worker.started.connect(lambda: self.analysis_started.emit(audio_path))
        self._analysis_worker.progress.connect(self.analysis_progress.emit)
        self._analysis_worker.finished.connect(self._on_analysis_finished)
        self._analysis_worker.error.connect(self._on_analysis_error)
        
        # Auto-cleanup
        self._analysis_worker.finished.connect(self._analysis_worker.deleteLater)
        self._analysis_worker.error.connect(self._analysis_worker.deleteLater)

        self._analysis_worker.start()

    def start_stem_separation(self, audio_path: str) -> None:
        """Manually start just the stem separation."""
        if self._stem_worker and self._stem_worker.isRunning():
            self._stem_worker.cancel()
            self._stem_worker.wait()

        self._stem_worker = StemSeparationWorker(audio_path, parent=self)
        
        self._stem_worker.started.connect(lambda: self.stem_started.emit(audio_path))
        self._stem_worker.progress.connect(self.stem_progress.emit)
        self._stem_worker.finished.connect(self._on_stem_finished)
        self._stem_worker.error.connect(self.stem_error.emit)
        
        self._stem_worker.finished.connect(self._stem_worker.deleteLater)
        self._stem_worker.error.connect(self._stem_worker.deleteLater)

        self._stem_worker.start()

    def cancel_all(self):
        """Cancel both workers."""
        if self._analysis_worker and self._analysis_worker.isRunning():
            self._analysis_worker.cancel()
            self._analysis_worker.wait()
        
        if self._stem_worker and self._stem_worker.isRunning():
            self._stem_worker.cancel()
            self._stem_worker.wait()

    def _on_analysis_finished(self, results: dict):
        """Handle analysis completion and optionally trigger stems."""
        self.analysis_complete.emit(results)
        self._analysis_worker = None

        audio_path = results.get("audio_path")
        if audio_path and self._pending_stem_creation.get(audio_path, False):
            # Chain the next task!
            logger.info(f"Auto-starting stem separation for {audio_path}")
            self.start_stem_separation(audio_path)
            # Clear pending flag
            self._pending_stem_creation.pop(audio_path, None)

    def _on_analysis_error(self, error: str):
        self.analysis_error.emit(error)
        self._analysis_worker = None

    def _on_stem_finished(self, results: dict):
        self.stem_complete.emit(results)
        self._stem_worker = None
