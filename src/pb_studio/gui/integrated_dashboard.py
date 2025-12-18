"""
Integrated Dashboard für PB_studio

Kombiniert Preview, Waveform und Parameter-Kontrolle in einem Dashboard.

Author: PB_studio Development Team
"""

import logging
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .controllers.waveform_controller import WaveformController
from .preview_widget import PreviewWidget
from .waveform_widget import WaveformWidget

logger = logging.getLogger(__name__)


class IntegratedDashboard(QWidget):
    """
    Integrated dashboard combining preview, waveform, and controls.

    Features:
        - Live video preview with playback controls
        - Audio waveform visualization with sync
        - Synced playhead between video and audio
        - Zoom and navigation controls

    Signals:
        position_changed: Emitted when playback position changes (float: seconds)
    """

    position_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        """Initialize Integrated Dashboard."""
        super().__init__(parent)

        # Components
        self.preview_widget: PreviewWidget | None = None
        self.waveform_widget: WaveformWidget | None = None

        # PERF-05 FIX: Use controller for async waveform loading
        self.waveform_controller = WaveformController(self)

        # State
        self.current_video_path: Path | None = None
        self.current_audio_path: Path | None = None

        self._setup_ui()
        self._connect_signals()

        logger.info("IntegratedDashboard initialized")

    def _setup_ui(self) -> None:
        """Setup user interface."""
        layout = QVBoxLayout(self)

        # Main splitter (vertical)
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Top: Video Preview
        preview_group = QGroupBox("Video Preview")
        preview_layout = QVBoxLayout()
        self.preview_widget = PreviewWidget()
        preview_layout.addWidget(self.preview_widget)
        preview_group.setLayout(preview_layout)
        splitter.addWidget(preview_group)

        # Bottom: Waveform
        waveform_group = QGroupBox("Audio Waveform")
        waveform_layout = QVBoxLayout()
        self.waveform_widget = WaveformWidget()
        self.waveform_widget.setMinimumHeight(150)
        waveform_layout.addWidget(self.waveform_widget)
        waveform_group.setLayout(waveform_layout)
        splitter.addWidget(waveform_group)

        # Set initial splitter sizes (60% preview, 40% waveform)
        splitter.setSizes([600, 400])

        layout.addWidget(splitter)

        # Control bar
        control_layout = QHBoxLayout()

        self.load_video_btn = QPushButton("Load Video")
        self.load_video_btn.clicked.connect(self._on_load_video)
        control_layout.addWidget(self.load_video_btn)

        self.load_audio_btn = QPushButton("Load Audio")
        self.load_audio_btn.clicked.connect(self._on_load_audio)
        control_layout.addWidget(self.load_audio_btn)

        control_layout.addStretch()

        self.status_label = QLabel("Ready")
        control_layout.addWidget(self.status_label)

        layout.addLayout(control_layout)

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        # Sync preview position to waveform
        if self.preview_widget:
            self.preview_widget.position_changed.connect(self._on_preview_position_changed)

        # Sync waveform position to preview
        if self.waveform_widget:
            self.waveform_widget.playhead_moved.connect(self._on_waveform_position_changed)

        # PERF-05 FIX: Connect waveform controller signals
        self.waveform_controller.load_started.connect(self._on_waveform_load_started)
        self.waveform_controller.load_progress.connect(self._on_waveform_load_progress)
        self.waveform_controller.load_complete.connect(self._on_waveform_load_complete)
        self.waveform_controller.load_error.connect(self._on_waveform_load_error)

    def load_video(self, video_path: str | Path) -> bool:
        """
        Load video file into preview.

        Args:
            video_path: Path to video file

        Returns:
            True if loaded successfully
        """
        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            return False

        success = self.preview_widget.load_video(video_path)
        if success:
            self.current_video_path = video_path
            self.status_label.setText(f"Video loaded: {video_path.name}")
            logger.info(f"Loaded video: {video_path}")
        return success

    def load_audio(self, audio_path: str | Path) -> None:
        """
        Load audio file and display waveform (async).

        PERF-05 FIX: Now loads waveform asynchronously to prevent UI blocking.

        Args:
            audio_path: Path to audio file
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            logger.error(f"Audio not found: {audio_path}")
            self.status_label.setText("Audio file not found")
            return

        # Store path for later reference
        self.current_audio_path = audio_path

        # Disable button during load
        self.load_audio_btn.setEnabled(False)

        # Start async loading via controller
        self.waveform_controller.load(str(audio_path))

    def _on_waveform_load_started(self, audio_path: str) -> None:
        """Handle waveform load started."""
        self.status_label.setText("Loading waveform...")
        logger.debug(f"Waveform loading started: {audio_path}")

    def _on_waveform_load_progress(self, percentage: int, status: str) -> None:
        """Handle waveform load progress update."""
        self.status_label.setText(f"Loading: {status} ({percentage}%)")

    def _on_waveform_load_complete(self, results: dict) -> None:
        """Handle waveform load completion."""
        # Re-enable button
        self.load_audio_btn.setEnabled(True)

        if not results.get("success", False):
            self.status_label.setText("Failed to load waveform")
            return

        # Set waveform data
        import numpy as np

        peaks_min = np.array(results["peaks_min"], dtype=np.float32)
        peaks_max = np.array(results["peaks_max"], dtype=np.float32)

        self.waveform_widget.set_waveform_data(
            peaks_min, peaks_max, results["sample_rate"], results["duration"]
        )

        # Set beat markers
        beat_times = results.get("beat_times", [])
        if beat_times:
            self.waveform_widget.set_beat_markers(beat_times)
            logger.info(f"Set {len(beat_times)} beat markers")

        # Update status
        audio_name = Path(results["audio_path"]).name
        duration = results["duration"]
        self.status_label.setText(f"Audio loaded: {audio_name} ({duration:.1f}s)")
        logger.info(f"Waveform loaded: {audio_name}, duration={duration:.1f}s")

    def _on_waveform_load_error(self, error_msg: str) -> None:
        """Handle waveform load error."""
        # Re-enable button
        self.load_audio_btn.setEnabled(True)

        self.status_label.setText(f"Error: {error_msg[:50]}...")
        logger.error(f"Waveform load error: {error_msg}")

    def _on_preview_position_changed(self, position: float) -> None:
        """Handle preview position change."""
        # Update waveform playhead
        self.waveform_widget.set_playhead_position(position)
        self.position_changed.emit(position)

    def _on_waveform_position_changed(self, position: float) -> None:
        """Handle waveform position change."""
        # Update preview position
        if self.preview_widget and self.preview_widget.cap:
            self.preview_widget.seek_to_time(position)

    def _on_load_video(self) -> None:
        """Handle load video button click."""
        from PyQt6.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )

        if file_path:
            self.load_video(file_path)

    def _on_load_audio(self) -> None:
        """Handle load audio button click."""
        from PyQt6.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", "Audio Files (*.wav *.mp3 *.flac *.ogg);;All Files (*)"
        )

        if file_path:
            self.load_audio(file_path)
