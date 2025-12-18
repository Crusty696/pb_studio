"""
Thumbnail Controller for PB_studio

PERF-10 FIX: Moves thumbnail generation to background thread to prevent UI blocking.

Uses QThread to run I/O-intensive thumbnail generation (frame extraction, image processing)
in a separate thread, emitting signals when complete.
"""

import logging
from pathlib import Path

from PyQt6.QtCore import QObject, QThread
from PyQt6.QtCore import pyqtSignal as Signal

logger = logging.getLogger(__name__)


class ThumbnailWorker(QThread):
    """
    Worker thread for thumbnail generation.

    PERF-10 FIX: Prevents UI blocking during thumbnail generation by running
    frame extraction and image processing in background.

    Signals:
        started: Emitted when generation begins
        progress: Emitted with (current, total, video_name)
        single_complete: Emitted when a single thumbnail is done (path, thumb_path)
        finished: Emitted with results dict when all complete
        error: Emitted with error message if generation fails
    """

    # Signals for thread communication
    started = Signal()
    progress = Signal(int, int, str)  # (current, total, video_name)
    single_complete = Signal(str, str)  # (video_path, thumbnail_path)
    finished = Signal(dict)  # Results dictionary
    error = Signal(str)  # Error message

    def __init__(
        self,
        video_paths: list[str],
        thumbnail_dir: str = "thumbnails",
        thumbnail_size: tuple = (160, 90),
        parent: QObject | None = None,
    ):
        """
        Initialize the thumbnail worker.

        Args:
            video_paths: List of video file paths to process
            thumbnail_dir: Directory to store thumbnails
            thumbnail_size: Target size (width, height)
            parent: Parent QObject (optional)
        """
        super().__init__(parent)
        self.video_paths = video_paths
        self.thumbnail_dir = thumbnail_dir
        self.thumbnail_size = thumbnail_size
        self._is_cancelled = False

    def cancel(self):
        """Request cancellation of the generation."""
        self._is_cancelled = True

    def run(self):
        """
        Execute thumbnail generation in background thread.

        Generates thumbnails for all video files, emitting progress updates.
        """
        try:
            self.started.emit()

            # Import here to avoid circular imports
            from ...video.thumbnail_generator import ThumbnailGenerator

            if self._is_cancelled:
                return

            # Create generator
            generator = ThumbnailGenerator(
                cache_dir=self.thumbnail_dir, thumbnail_size=self.thumbnail_size
            )

            results = {
                "success": True,
                "total": len(self.video_paths),
                "generated": 0,
                "cached": 0,
                "failed": 0,
                "thumbnails": {},
            }

            total = len(self.video_paths)

            for i, video_path in enumerate(self.video_paths):
                if self._is_cancelled:
                    results["success"] = False
                    break

                video_path = Path(video_path)
                video_name = video_path.name

                # Emit progress
                self.progress.emit(i + 1, total, video_name)

                # Check if already cached
                cached_path = generator.get_thumbnail_path(video_path)
                if cached_path:
                    results["cached"] += 1
                    results["thumbnails"][str(video_path)] = str(cached_path)
                    self.single_complete.emit(str(video_path), str(cached_path))
                    continue

                # Generate thumbnail
                thumb_path = generator.generate(video_path)

                if thumb_path:
                    results["generated"] += 1
                    results["thumbnails"][str(video_path)] = str(thumb_path)
                    self.single_complete.emit(str(video_path), str(thumb_path))
                else:
                    results["failed"] += 1
                    results["thumbnails"][str(video_path)] = None

            logger.info(
                f"Thumbnail generation complete: "
                f"{results['generated']} generated, "
                f"{results['cached']} cached, "
                f"{results['failed']} failed"
            )

            self.finished.emit(results)

        except Exception as e:
            error_msg = f"Thumbnail generation failed: {e}"
            logger.error(error_msg, exc_info=True)
            self.error.emit(error_msg)


class ThumbnailController(QObject):
    """
    Controller for managing thumbnail generation operations.

    PERF-10 FIX: Provides high-level API for non-blocking thumbnail generation.
    Manages worker threads and provides convenience signals for UI integration.

    Usage:
        controller = ThumbnailController(parent_widget)
        controller.generation_complete.connect(on_thumbnails_done)
        controller.generate(video_paths)
    """

    # High-level signals
    generation_started = Signal()
    generation_progress = Signal(int, int, str)  # (current, total, video_name)
    thumbnail_ready = Signal(str, str)  # (video_path, thumbnail_path)
    generation_complete = Signal(dict)  # results
    generation_error = Signal(str)  # error message

    def __init__(
        self,
        thumbnail_dir: str = "thumbnails",
        thumbnail_size: tuple = (160, 90),
        parent: QObject | None = None,
    ):
        """
        Initialize the thumbnail controller.

        Args:
            thumbnail_dir: Directory to store thumbnails
            thumbnail_size: Target size (width, height)
            parent: Parent QObject (optional)
        """
        super().__init__(parent)
        self.thumbnail_dir = thumbnail_dir
        self.thumbnail_size = thumbnail_size
        self._current_worker: ThumbnailWorker | None = None

    def generate(self, video_paths: list[str]) -> None:
        """
        Start thumbnail generation in background thread.

        If generation is already running, it will be cancelled first.

        Args:
            video_paths: List of video file paths to process
        """
        if not video_paths:
            logger.warning("No video paths provided for thumbnail generation")
            return

        # Cancel any existing generation
        self.cancel()

        # Create and start new worker
        self._current_worker = ThumbnailWorker(
            video_paths, self.thumbnail_dir, self.thumbnail_size, self
        )

        # Connect signals
        self._current_worker.started.connect(self.generation_started.emit)
        self._current_worker.progress.connect(self.generation_progress.emit)
        self._current_worker.single_complete.connect(self.thumbnail_ready.emit)
        self._current_worker.finished.connect(self._on_worker_finished)
        self._current_worker.error.connect(self._on_worker_error)

        # Start the worker
        logger.info(f"Starting background thumbnail generation: {len(video_paths)} videos")
        self._current_worker.start()

    def generate_single(self, video_path: str) -> None:
        """
        Generate single thumbnail in background.

        Convenience method for generating one thumbnail.

        Args:
            video_path: Path to video file
        """
        self.generate([video_path])

    def cancel(self) -> None:
        """Cancel current generation if running."""
        if self._current_worker and self._current_worker.isRunning():
            logger.info("Cancelling current thumbnail generation")
            self._current_worker.cancel()
            self._current_worker.quit()
            self._current_worker.wait(1000)  # Wait up to 1 second
            self._current_worker = None

    def is_generating(self) -> bool:
        """Check if generation is currently running."""
        return self._current_worker is not None and self._current_worker.isRunning()

    def _on_worker_finished(self, results: dict) -> None:
        """Handle worker completion."""
        self.generation_complete.emit(results)
        self._current_worker = None

    def _on_worker_error(self, error_msg: str) -> None:
        """Handle worker error."""
        self.generation_error.emit(error_msg)
        self._current_worker = None
