"""
Preview Widget für PB_studio

Zeigt Live-Vorschau von Video-Clips.

Author: PB_studio Development Team
"""

import logging
import threading
from pathlib import Path

import cv2
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class PreviewWidget(QWidget):
    """
    Widget for video preview.

    Features:
        - Display video frames
        - Playback controls
        - Seek with slider
        - Auto-update on parameter changes

    Signals:
        position_changed: Emitted when playback position changes (float: seconds)
    """

    position_changed = pyqtSignal(float)  # position in seconds

    def __init__(self, parent=None):
        """Initialize Preview Widget."""
        super().__init__(parent)

        self.cap: cv2.VideoCapture | None = None
        self.fps: float = 30.0
        self.frame_count: int = 0
        self.current_frame: int = 0
        self.is_playing: bool = False

        # FIX #3: Thread-Safety für OpenCV VideoCapture
        # OpenCV ist NICHT thread-safe - gleichzeitiger Zugriff aus Timer und seek_to_time()
        # kann zu Race Conditions führen (Crash, korrupte Frames)
        self._cap_lock = threading.Lock()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._next_frame)

        self._setup_ui()
        logger.debug("PreviewWidget initialized")

    def _setup_ui(self) -> None:
        """Setup user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setStyleSheet("background-color: black; border: 1px solid #555;")
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.video_label)

        # Seek slider
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.sliderMoved.connect(self._on_seek)
        layout.addWidget(self.seek_slider)

        # Controls
        controls_layout = QHBoxLayout()

        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self._toggle_playback)
        self.play_btn.setMaximumWidth(100)
        controls_layout.addWidget(self.play_btn)

        self.time_label = QLabel("0:00 / 0:00")
        controls_layout.addWidget(self.time_label)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

    def load_video(self, video_path: str | Path) -> bool:
        """
        Load video file (thread-safe).

        Args:
            video_path: Path to video file

        Returns:
            True if loaded successfully, False otherwise
        """
        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            return False

        # FIX #3: Thread-Safety - Lock für VideoCapture-Operationen
        with self._cap_lock:
            if self.cap:
                self.cap.release()

            self.cap = cv2.VideoCapture(str(video_path))

            if not self.cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return False

            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.seek_slider.setMaximum(self.frame_count - 1)
        self._display_frame(0)
        self._update_time_label()

        logger.info(
            f"Loaded video: {video_path.name}, {self.frame_count} frames @ {self.fps:.1f} fps"
        )
        return True

    def _display_frame(self, frame_number: int) -> None:
        """Display specific frame (thread-safe)."""
        if not self.cap:
            return

        # FIX #3: Thread-Safety - Lock für alle VideoCapture-Operationen
        with self._cap_lock:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()

            if not ret:
                return

            # Convert BGR to RGB (innerhalb Lock, da frame-Daten noch verwendet werden)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # QImage-Erstellung und UI-Update außerhalb des Locks (nicht VideoCapture-abhängig)
        # Convert to QImage
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Scale to fit widget
        pixmap = QPixmap.fromImage(qt_image)
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.video_label.setPixmap(scaled)
        self.current_frame = frame_number
        self.seek_slider.setValue(frame_number)

        # Emit position
        if self.fps > 0:
            position = frame_number / self.fps
            self.position_changed.emit(position)

    def _toggle_playback(self) -> None:
        """Toggle play/pause."""
        if self.is_playing:
            self.timer.stop()
            self.play_btn.setText("▶ Play")
            self.is_playing = False
            logger.debug("Playback paused")
        else:
            if self.fps > 0:
                interval = int(1000 / self.fps)
                self.timer.start(interval)
                self.play_btn.setText("⏸ Pause")
                self.is_playing = True
                logger.debug("Playback started")

    def _next_frame(self) -> None:
        """Display next frame during playback."""
        next_frame = self.current_frame + 1
        if next_frame >= self.frame_count:
            next_frame = 0

        self._display_frame(next_frame)
        self._update_time_label()

    def _on_seek(self, frame_number: int) -> None:
        """Handle seek slider."""
        self._display_frame(frame_number)
        self._update_time_label()

    def _update_time_label(self) -> None:
        """Update time display."""
        if self.fps == 0:
            return

        current_time = self.current_frame / self.fps
        total_time = self.frame_count / self.fps

        current_str = self._format_time(current_time)
        total_str = self._format_time(total_time)

        self.time_label.setText(f"{current_str} / {total_str}")

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as MM:SS."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"

    def seek_to_time(self, seconds: float) -> None:
        """
        Seek to specific time.

        Args:
            seconds: Time in seconds

        HIGH-14 FIX: Added null check for self.cap before seeking.
        """
        if not self.cap:
            logger.warning("Cannot seek: video not loaded")
            return

        if self.fps > 0:
            frame = int(seconds * self.fps)
            frame = max(0, min(frame, self.frame_count - 1))
            self._display_frame(frame)
            self._update_time_label()

    def closeEvent(self, event):
        """Cleanup on close (thread-safe)."""
        if self.timer.isActive():
            self.timer.stop()

        # FIX #3: Thread-Safety - Lock für VideoCapture-Freigabe
        with self._cap_lock:
            if self.cap:
                self.cap.release()
                self.cap = None

        super().closeEvent(event)
