"""
Timeline Widget für PB_studio

Interaktive Timeline mit Waveform-Visualisierung, Schnitt-Markern und Playhead.
Unterstützt Zoom, Scroll und zeigt Audio-Beats an.

Features:
- Waveform-Visualisierung
- Zeit-Marker und Grid
- Schnitt-Marker (Cuts)
- Playhead mit aktueller Position
- Zoom und Scroll
- Beat-Grid Overlay

Author: PB_studio Development Team
"""

from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPen
from PyQt6.QtWidgets import (
    QGraphicsLineItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..audio.audio_analyzer import AudioAnalyzer
from ..utils.logger import get_logger

logger = get_logger(__name__)


class AudioLoaderWorker(QObject):
    """Worker for loading audio files in background thread."""

    finished = pyqtSignal(dict)  # Emits audio data when done
    error = pyqtSignal(str)  # Emits error message
    progress = pyqtSignal(str)  # Emits progress status

    def __init__(self, audio_path: Path):
        super().__init__()
        self.audio_path = audio_path

    def run(self):
        """Load audio file and analyze beatgrid."""
        try:
            import librosa

            self.progress.emit("Lade Audio-Datei...")

            # Load audio with librosa
            y, sr = librosa.load(str(self.audio_path), sr=22050, mono=True)
            duration = librosa.get_duration(y=y, sr=sr)

            logger.info(f"Audio loaded: duration={duration:.2f}s, sr={sr}")
            self.progress.emit("Analysiere Beats...")

            # Analyze beatgrid
            analyzer = AudioAnalyzer()
            beatgrid_result = analyzer.analyze_beatgrid(self.audio_path)

            result = {
                "audio_path": self.audio_path,
                "waveform_data": y,
                "sample_rate": sr,
                "duration": duration,
                "beatgrid": beatgrid_result,
            }

            self.finished.emit(result)

        except Exception as e:
            logger.error(f"Audio loading failed: {e}", exc_info=True)
            self.error.emit(str(e))


class WaveformItem(QGraphicsRectItem):
    """Custom graphics item for rendering audio waveform."""

    def __init__(self, waveform_data: np.ndarray, width: float, height: float):
        """
        Initialize waveform item.

        Args:
            waveform_data: Audio amplitude data (downsampled)
            width: Width of the waveform display
            height: Height of the waveform display
        """
        super().__init__(0, 0, width, height)
        self.waveform_data = waveform_data
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable, False)

        # QUICK-WIN #3: Pre-compute waveform bars once (50-70% faster rendering)
        self.cached_bars = self._precompute_bars(width, height) if waveform_data is not None else []

    def _precompute_bars(self, width: float, height: float) -> list[tuple[int, int, int, int]]:
        """
        Pre-compute waveform bar positions and heights.

        Returns:
            List of tuples: (x, y, width, height) for each bar
        """
        if self.waveform_data is None or len(self.waveform_data) == 0:
            return []

        bars = []
        center_y = height / 2
        samples_per_pixel = max(1, len(self.waveform_data) // int(width))

        for x in range(int(width)):
            start_idx = x * samples_per_pixel
            end_idx = min(start_idx + samples_per_pixel, len(self.waveform_data))

            if start_idx >= len(self.waveform_data):
                break

            # Get max amplitude in this segment
            segment = self.waveform_data[start_idx:end_idx]
            if len(segment) > 0:
                amplitude = np.max(np.abs(segment))
                bar_height = amplitude * height * 0.8  # Scale to 80% of height

                # Store bar rectangle (x, y, width, height)
                bars.append((int(x), int(center_y - bar_height / 2), 1, int(bar_height)))

        return bars

    def paint(self, painter, option, widget):
        """Custom paint method for waveform (optimized with pre-computed bars)."""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw background
        painter.fillRect(self.rect(), QBrush(QColor(35, 35, 35)))

        if not self.cached_bars:
            return

        # Create gradient for waveform
        gradient = QLinearGradient(0, 0, 0, self.rect().height())
        gradient.setColorAt(0.0, QColor(0, 255, 0, 180))  # Green top
        gradient.setColorAt(0.5, QColor(0, 200, 0, 200))  # Green middle
        gradient.setColorAt(1.0, QColor(0, 255, 0, 180))  # Green bottom

        painter.setPen(QPen(Qt.PenStyle.NoPen))
        painter.setBrush(QBrush(gradient))

        # QUICK-WIN #3: Draw pre-computed waveform bars (50-70% faster)
        for x, y, w, h in self.cached_bars:
            painter.drawRect(x, y, w, h)


class TimelineWidget(QWidget):
    """
    Interactive timeline widget with waveform visualization.

    Signals:
        position_changed: Emitted when playhead position changes (time in seconds)
        zoom_changed: Emitted when zoom level changes (zoom factor)
        cut_clicked: Emitted when a cut marker is clicked (cut_id)
        audio_loaded: Emitted when audio file is fully loaded (with duration)
        audio_load_error: Emitted when audio loading fails (with error message)
    """

    position_changed = pyqtSignal(float)  # time in seconds
    zoom_changed = pyqtSignal(float)  # zoom factor
    cut_clicked = pyqtSignal(int)  # cut_id
    preview_marker_placed = pyqtSignal(float)  # preview start position (seconds)
    audio_loaded = pyqtSignal(float)  # duration in seconds
    audio_load_error = pyqtSignal(str)  # error message

    def __init__(self, parent=None):
        super().__init__(parent)
        logger.info("Initializing TimelineWidget")

        # Audio loader worker thread
        self._audio_thread: QThread | None = None
        self._audio_worker: AudioLoaderWorker | None = None

        # Timeline state
        self.audio_path: Path | None = None
        self.waveform_data: np.ndarray | None = None
        self.sample_rate: int | None = None
        self.duration: float = 0.0

        # Playback state
        self.current_position: float = 0.0  # seconds
        self.is_playing: bool = False

        # Beat grid
        self.beat_times: list[float] = []  # beat positions in seconds
        self.bpm: float | None = None
        self.beat_grid_items: list[QGraphicsLineItem] = []  # beat grid lines

        # Cut markers
        self.cuts: list[tuple[int, float, float]] = []  # (id, start_time, end_time)

        # Preview marker (for F6 Preview)
        self.preview_marker_position: float | None = None  # seconds
        self.preview_marker_line: QGraphicsLineItem | None = None

        # View state
        self.zoom_level: float = 1.0  # 1.0 = fit to width
        self.scroll_position: float = 0.0  # seconds

        # Graphics components
        self.scene: QGraphicsScene | None = None
        self.view: QGraphicsView | None = None
        self.waveform_item: WaveformItem | None = None
        self.playhead_line: QGraphicsLineItem | None = None

        # UI setup
        self._init_ui()

        logger.info("TimelineWidget initialization complete")

    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout()

        # Graphics view for timeline
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setMinimumHeight(150)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Set dark background
        self.view.setBackgroundBrush(QBrush(QColor(25, 25, 25)))

        layout.addWidget(self.view)

        # Control panel
        controls_layout = QHBoxLayout()

        # Zoom controls
        zoom_label = QLabel("Zoom:")
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setMinimum(10)  # 10% (0.1x)
        self.zoom_slider.setMaximum(1000)  # 1000% (10x)
        self.zoom_slider.setValue(100)  # 100% (1x)
        self.zoom_slider.setMaximumWidth(200)
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)

        self.zoom_value_label = QLabel("100%")
        self.zoom_value_label.setMinimumWidth(50)

        # Fit button
        fit_button = QPushButton("Fit")
        fit_button.setMaximumWidth(60)
        fit_button.clicked.connect(self._fit_to_view)

        # Time display
        self.time_label = QLabel("00:00.000 / 00:00.000")
        self.time_label.setMinimumWidth(150)

        # BPM display
        self.bpm_label = QLabel("BPM: --")
        self.bpm_label.setMinimumWidth(80)
        self.bpm_label.setStyleSheet("QLabel { color: #FFD700; font-weight: bold; }")

        controls_layout.addWidget(zoom_label)
        controls_layout.addWidget(self.zoom_slider)
        controls_layout.addWidget(self.zoom_value_label)
        controls_layout.addWidget(fit_button)
        controls_layout.addStretch()
        controls_layout.addWidget(self.bpm_label)
        controls_layout.addWidget(self.time_label)

        layout.addLayout(controls_layout)
        self.setLayout(layout)

        # Timer for playback position updates
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._update_playhead)

        logger.debug("Timeline UI initialized")

    def load_audio(self, audio_path: str | Path):
        """
        Load audio file asynchronously in background thread.

        Args:
            audio_path: Path to audio file
        """
        audio_path = Path(audio_path)

        # Validate file exists
        if not audio_path.exists():
            error_msg = f"Audio file not found: {audio_path}"
            logger.error(error_msg)
            self.audio_load_error.emit(error_msg)
            return

        # Validate audio file format
        valid_extensions = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
        if audio_path.suffix.lower() not in valid_extensions:
            error_msg = f"Unsupported audio format: {audio_path.suffix}"
            logger.error(error_msg)
            self.audio_load_error.emit(error_msg)
            return

        logger.info(f"Loading audio asynchronously: {audio_path.name}")

        # Stop any existing worker
        self._stop_audio_worker()

        # Create worker and thread
        self._audio_thread = QThread()
        self._audio_worker = AudioLoaderWorker(audio_path)
        self._audio_worker.moveToThread(self._audio_thread)

        # Connect signals
        self._audio_thread.started.connect(self._audio_worker.run)
        self._audio_worker.finished.connect(self._on_audio_loaded)
        self._audio_worker.error.connect(self._on_audio_error)
        self._audio_worker.progress.connect(self._on_audio_progress)
        self._audio_worker.finished.connect(self._audio_thread.quit)
        self._audio_worker.error.connect(self._audio_thread.quit)

        # Zentrales Aufräumen, damit gelöschte QThreads nicht erneut verwendet werden
        self._audio_thread.finished.connect(self._cleanup_audio_thread)
        self._audio_worker.finished.connect(self._cleanup_audio_thread)
        self._audio_worker.error.connect(self._cleanup_audio_thread)

        # Start loading
        self._audio_thread.start()

    def _stop_audio_worker(self):
        """Stop any running audio worker thread."""
        try:
            if self._audio_thread and self._audio_thread.isRunning():
                self._audio_thread.quit()
                self._audio_thread.wait(3000)  # Wait max 3 seconds
                logger.debug("Previous audio worker stopped")
        except RuntimeError:
            logger.debug("Audio worker thread already deleted during stop")
        finally:
            self._cleanup_audio_thread()

    def _cleanup_audio_thread(self):
        """Safely delete worker/thread and reset references."""
        if self._audio_worker:
            try:
                self._audio_worker.deleteLater()
            except RuntimeError:
                pass
            self._audio_worker = None

        if self._audio_thread:
            try:
                self._audio_thread.deleteLater()
            except RuntimeError:
                pass
            self._audio_thread = None

    def _on_audio_loaded(self, result: dict):
        """Handle successful audio loading."""
        self.audio_path = result["audio_path"]
        self.waveform_data = result["waveform_data"]
        self.sample_rate = result["sample_rate"]
        self.duration = result["duration"]

        logger.info(f"Audio loaded: duration={self.duration:.2f}s")

        # Set beat grid if available
        beatgrid = result.get("beatgrid")
        if beatgrid:
            self.set_beat_grid(beat_times=beatgrid["beat_times"], bpm=beatgrid["bpm"])
            logger.info(
                f"Beatgrid: {len(beatgrid['beat_times'])} beats at {beatgrid['bpm']:.2f} BPM"
            )
        else:
            logger.warning("No beatgrid available")
            if hasattr(self, "bpm_label"):
                self.bpm_label.setText("BPM: --")

        # Render waveform
        self._render_timeline()

        # Emit signal
        self.audio_loaded.emit(self.duration)
        self._cleanup_audio_thread()

    def _on_audio_error(self, error_msg: str):
        """Handle audio loading error."""
        logger.error(f"Audio loading failed: {error_msg}")
        self.audio_load_error.emit(error_msg)
        self._cleanup_audio_thread()

    def _on_audio_progress(self, status: str):
        """Handle audio loading progress update."""
        logger.info(f"Audio loading: {status}")

    def set_beat_grid(self, beat_times: list[float], bpm: float):
        """
        Set beat grid overlay.

        Args:
            beat_times: List of beat positions in seconds
            bpm: Beats per minute
        """
        self.beat_times = beat_times
        self.bpm = bpm

        # Update BPM display
        if hasattr(self, "bpm_label"):
            self.bpm_label.setText(f"BPM: {bpm:.1f}")

        logger.info(f"Beat grid set: {len(beat_times)} beats, BPM={bpm:.2f}")
        self._render_beat_grid()

    def set_cuts(self, cuts: list[tuple[int, float, float]]):
        """
        Set cut markers on timeline.

        Args:
            cuts: List of (cut_id, start_time, end_time) tuples
        """
        self.cuts = cuts
        logger.info(f"Cut markers set: {len(cuts)} cuts")
        self._render_cut_markers()

    def display_cut_list(self, cut_list: list):
        """
        Display CutListEntry objects as video clips on the timeline.

        Shows each clip as a colored rectangle with the clip name/ID.

        Args:
            cut_list: List of CutListEntry objects (from pacing engine)
        """
        logger.info(f"Displaying cut list with {len(cut_list)} clips on timeline")

        if not cut_list or not self.waveform_item:
            logger.warning("Cannot display cut list: no cuts or waveform not rendered")
            return

        if self.duration <= 0:
            logger.warning("Cannot display cut list: duration is zero")
            return

        # Clear old cut list visualizations if any exist
        # (We'll store them in a new attribute)
        if hasattr(self, "cut_list_items"):
            for item in self.cut_list_items:
                try:
                    if item.scene():
                        self.scene.removeItem(item)
                except RuntimeError:
                    pass
        self.cut_list_items = []

        # Get timeline dimensions
        width = self.waveform_item.rect().width()
        height = self.waveform_item.rect().height()
        pixels_per_second = width / self.duration

        # Color palette for different clips (cycling colors)
        colors = [
            QColor(100, 149, 237),  # Cornflower Blue
            QColor(255, 127, 80),  # Coral
            QColor(144, 238, 144),  # Light Green
            QColor(221, 160, 221),  # Plum
            QColor(255, 215, 0),  # Gold
            QColor(70, 130, 180),  # Steel Blue
            QColor(255, 182, 193),  # Light Pink
            QColor(152, 251, 152),  # Pale Green
        ]

        # Track unique clip_ids for color assignment
        clip_id_to_color = {}
        color_index = 0

        # Render each cut as a video clip rectangle
        for i, cut in enumerate(cut_list):
            start_x = cut.start_time * pixels_per_second
            end_x = cut.end_time * pixels_per_second
            cut_width = end_x - start_x

            # Assign color based on clip_id
            clip_id = cut.clip_id
            if clip_id not in clip_id_to_color:
                clip_id_to_color[clip_id] = colors[color_index % len(colors)]
                color_index += 1

            base_color = clip_id_to_color[clip_id]

            # Draw clip rectangle (positioned in upper half of waveform)
            clip_y = 5  # Small offset from top
            clip_height = height * 0.4  # 40% of waveform height

            clip_rect = QGraphicsRectItem(start_x, clip_y, cut_width, clip_height)
            clip_rect.setBrush(
                QBrush(QColor(base_color.red(), base_color.green(), base_color.blue(), 180))
            )
            clip_rect.setPen(QPen(base_color, 2))
            clip_rect.setZValue(3)  # Above waveform and beat grid
            self.scene.addItem(clip_rect)
            self.cut_list_items.append(clip_rect)

            # Extract clip name from file path
            clip_name = Path(clip_id).stem if clip_id else f"Clip{i}"

            # Draw clip name label (if clip is wide enough)
            if cut_width > 50:  # Only show label if clip is wide enough
                label = QGraphicsTextItem(clip_name)
                label.setDefaultTextColor(QColor(255, 255, 255))

                # Scale text to fit clip
                label.setScale(0.6)

                # Position label in center of clip
                label_width = label.boundingRect().width() * 0.6
                label_x = start_x + (cut_width - label_width) / 2
                label_y = clip_y + (clip_height - label.boundingRect().height() * 0.6) / 2

                label.setPos(label_x, label_y)
                label.setZValue(4)  # Above clip rectangle
                self.scene.addItem(label)
                self.cut_list_items.append(label)

        logger.info(f"Displayed {len(cut_list)} video clips on timeline")

    def set_position(self, position: float):
        """
        Set current playback position.

        Args:
            position: Time in seconds
        """
        self.current_position = max(0.0, min(position, self.duration))
        self._update_playhead()
        self._update_time_display()

    def start_playback(self):
        """Start playback timer."""
        self.is_playing = True
        self.playback_timer.start(16)  # ~60 FPS
        logger.debug("Playback started")

    def stop_playback(self):
        """Stop playback timer."""
        self.is_playing = False
        self.playback_timer.stop()
        logger.debug("Playback stopped")

    def _render_timeline(self):
        """Render the complete timeline with waveform."""
        if self.waveform_data is None:
            return

        # Clear scene
        self.scene.clear()

        # Calculate dimensions
        view_width = self.view.viewport().width()
        view_height = 120
        timeline_width = view_width * self.zoom_level

        logger.debug(f"Rendering timeline: width={timeline_width}, zoom={self.zoom_level}")

        # Create waveform item
        self.waveform_item = WaveformItem(self.waveform_data, timeline_width, view_height)
        self.scene.addItem(self.waveform_item)

        # Draw time markers
        self._render_time_markers(timeline_width, view_height)

        # Draw beat grid if available
        if self.beat_times:
            self._render_beat_grid()

        # Draw cut markers if available
        if self.cuts:
            self._render_cut_markers()

        # Draw playhead
        self._render_playhead(view_height)

        # Update scene rect
        self.scene.setSceneRect(0, 0, timeline_width, view_height + 40)

    def _render_time_markers(self, width: float, height: float):
        """Render time markers on timeline."""
        if self.duration <= 0:
            return

        # Determine marker interval based on zoom
        visible_duration = self.duration / self.zoom_level

        if visible_duration > 60:
            interval = 10.0  # 10 second markers
        elif visible_duration > 30:
            interval = 5.0  # 5 second markers
        elif visible_duration > 10:
            interval = 1.0  # 1 second markers
        else:
            interval = 0.5  # 0.5 second markers

        pixels_per_second = width / self.duration

        time = 0.0
        while time <= self.duration:
            x = time * pixels_per_second

            # Draw marker line
            line = QGraphicsLineItem(x, height, x, height + 10)
            line.setPen(QPen(QColor(100, 100, 100), 1))
            self.scene.addItem(line)

            # Draw time text
            minutes = int(time // 60)
            seconds = time % 60
            time_text = QGraphicsTextItem(f"{minutes:02d}:{seconds:05.2f}")
            time_text.setDefaultTextColor(QColor(150, 150, 150))
            time_text.setPos(x - 25, height + 10)
            time_text.setScale(0.8)
            self.scene.addItem(time_text)

            time += interval

    def _render_beat_grid(self):
        """Render beat grid overlay with intelligent decimation to prevent memory issues."""
        # Clear old beat grid items
        for item in self.beat_grid_items:
            try:
                if item.scene():
                    self.scene.removeItem(item)
            except RuntimeError:
                # Item already deleted by Qt, skip it
                pass
        self.beat_grid_items.clear()

        if not self.beat_times or not self.waveform_item:
            return

        # Prevent division by zero
        if self.duration <= 0:
            logger.warning("Cannot render beat grid: duration is zero or negative")
            return

        width = self.waveform_item.rect().width()
        height = self.waveform_item.rect().height()
        pixels_per_second = width / self.duration

        # Calculate decimation factor to avoid rendering too many lines
        total_beats = len(self.beat_times)
        max_lines = 1000  # Maximum number of lines to render

        # Calculate average pixels per beat
        if total_beats > 1:
            avg_time_per_beat = self.duration / total_beats
            avg_pixels_per_beat = avg_time_per_beat * pixels_per_second
        else:
            avg_pixels_per_beat = width

        # Decimation strategy:
        # - If avg_pixels_per_beat >= 3: render all beats (they're spaced enough)
        # - If avg_pixels_per_beat < 3: decimate to avoid cluttering
        # - Always respect max_lines limit

        if avg_pixels_per_beat >= 3 and total_beats <= max_lines:
            # Render all beats - they're well spaced
            decimation = 1
        else:
            # Calculate decimation factor
            # We want at least 3 pixels between lines
            min_pixels_per_line = 3
            decimation = max(1, int(avg_time_per_beat * pixels_per_second / min_pixels_per_line))

            # Also respect max_lines limit
            if total_beats / decimation > max_lines:
                decimation = max(1, int(total_beats / max_lines))

        rendered_count = 0
        for i, beat_time in enumerate(self.beat_times):
            # Skip beats based on decimation
            if i % decimation != 0:
                continue

            # Skip beats outside the timeline duration
            if beat_time < 0 or beat_time > self.duration:
                continue

            x = beat_time * pixels_per_second

            # Draw beat line
            line = QGraphicsLineItem(x, 0, x, height)
            line.setPen(QPen(QColor(255, 255, 0, 80), 1, Qt.PenStyle.DashLine))
            line.setZValue(1)  # Above waveform
            self.scene.addItem(line)
            self.beat_grid_items.append(line)
            rendered_count += 1

        if decimation > 1:
            logger.debug(
                f"Rendered {rendered_count} of {total_beats} beat grid lines (decimation: every {decimation}th beat)"
            )
        else:
            logger.debug(f"Rendered {rendered_count} beat grid lines")

    def _render_cut_markers(self):
        """Render cut markers on timeline."""
        if not self.cuts or not self.waveform_item:
            return

        width = self.waveform_item.rect().width()
        height = self.waveform_item.rect().height()
        pixels_per_second = width / self.duration

        for cut_id, start_time, end_time in self.cuts:
            start_x = start_time * pixels_per_second
            end_x = end_time * pixels_per_second
            cut_width = end_x - start_x

            # Draw cut region
            cut_rect = QGraphicsRectItem(start_x, 0, cut_width, height)
            cut_rect.setBrush(QBrush(QColor(42, 130, 218, 60)))
            cut_rect.setPen(QPen(QColor(42, 130, 218, 200), 2))
            cut_rect.setZValue(2)  # Above beat grid
            self.scene.addItem(cut_rect)

            # Draw cut ID label
            label = QGraphicsTextItem(f"C{cut_id}")
            label.setDefaultTextColor(QColor(255, 255, 255))
            label.setPos(start_x + 5, 5)
            label.setZValue(3)
            self.scene.addItem(label)

    def _render_playhead(self, height: float):
        """Render playhead line."""
        if self.waveform_item is None:
            return

        width = self.waveform_item.rect().width()
        pixels_per_second = width / self.duration if self.duration > 0 else 0
        x = self.current_position * pixels_per_second

        # Draw playhead line
        self.playhead_line = QGraphicsLineItem(x, 0, x, height)
        self.playhead_line.setPen(QPen(QColor(255, 0, 0), 2))
        self.playhead_line.setZValue(10)  # Always on top
        self.scene.addItem(self.playhead_line)

    def _update_playhead(self):
        """Update playhead position during playback."""
        if not self.is_playing:
            return

        # Increment position (simulated - in real app would sync with audio player)
        self.current_position += 0.016  # ~60 FPS

        if self.current_position >= self.duration:
            self.current_position = self.duration
            self.stop_playback()

        # Update playhead line position
        if self.playhead_line and self.waveform_item:
            width = self.waveform_item.rect().width()
            pixels_per_second = width / self.duration if self.duration > 0 else 0
            x = self.current_position * pixels_per_second

            line = self.playhead_line.line()
            self.playhead_line.setLine(x, line.y1(), x, line.y2())

            # Auto-scroll to keep playhead visible
            self.view.ensureVisible(self.playhead_line, 100, 0)

        self._update_time_display()
        self.position_changed.emit(self.current_position)

    def _update_time_display(self):
        """Update time display label."""
        current_minutes = int(self.current_position // 60)
        current_seconds = self.current_position % 60
        total_minutes = int(self.duration // 60)
        total_seconds = self.duration % 60

        time_str = (
            f"{current_minutes:02d}:{current_seconds:06.3f} / "
            f"{total_minutes:02d}:{total_seconds:06.3f}"
        )
        self.time_label.setText(time_str)

    def _on_zoom_changed(self, value: int):
        """Handle zoom slider change."""
        self.zoom_level = value / 100.0  # Convert to zoom factor
        self.zoom_value_label.setText(f"{value}%")

        if self.waveform_data is not None:
            self._render_timeline()

        self.zoom_changed.emit(self.zoom_level)
        logger.debug(f"Zoom changed: {self.zoom_level:.2f}x")

    def _fit_to_view(self):
        """Reset zoom to fit entire audio in view."""
        self.zoom_slider.setValue(100)
        logger.debug("Zoom reset to fit")

    def clear(self):
        """Clear timeline and reset state."""
        self.audio_path = None
        self.waveform_data = None
        self.sample_rate = None
        self.duration = 0.0
        self.current_position = 0.0
        self.beat_times = []
        self.bpm = None
        self.cuts = []

        self.scene.clear()
        self.time_label.setText("00:00.000 / 00:00.000")

        logger.info("Timeline cleared")
