"""
Clip Library Widget für PB_studio

Zeigt verfügbare Videoclips in einer durchsuchbaren Bibliothek.
Unterstützt Thumbnail-Ansicht, Metadaten-Anzeige und Clip-Import.

Features:
- Grid/List-Ansicht der Clips
- Thumbnail-Vorschau
- Clip-Metadaten (Name, Dauer, Auflösung, FPS)
- Such- und Filterfunktion mit Analyse-Kategorien
- Clip-Import aus Dateisystem
- Sortierung nach verschiedenen Kriterien
- Drag-and-Drop Import von Video-Dateien (Task 35)
- Erweiterte Filterung nach Motion, Mood, Style, Scene

Author: PB_studio Development Team
"""

import json
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDropEvent, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from sqlalchemy.orm import joinedload

from ..core.constants import MAX_INITIAL_RENDER
from ..database.connection import get_db_session
from ..database.db_worker import DatabaseWorker
from ..database.models import VideoClip
from ..utils.logger import get_logger
from ..utils.path_utils import resolve_relative_path, to_relative_path
from ..video.thumbnail_generator import ThumbnailGenerator
from ..video.video_analyzer import VideoAnalyzer
from ..utils.clip_data_loader import ClipDataLoader
from .clip_filter_widget import ClipFilterWidget, apply_filters_to_clips

# UI Layout Konstanten
CLIPS_PER_ROW = 3  # Spalten im Grid-Layout
PROGRESS_UPDATE_INTERVAL = 50  # UI Update alle N Clips

# Unterstützte Video-Formate für Drag-and-Drop
SUPPORTED_VIDEO_FORMATS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv"}

logger = get_logger(__name__)


class ClipThumbnailWidget(QFrame):
    """Widget for displaying a single clip with thumbnail and metadata."""

    clip_selected = pyqtSignal(int)  # clip_id
    clip_delete_requested = pyqtSignal(int)  # clip_id

    # Class-level thumbnail generator (shared across all instances)
    _thumbnail_generator = None

    @classmethod
    def get_thumbnail_generator(cls) -> ThumbnailGenerator:
        """Get or create shared thumbnail generator."""
        if cls._thumbnail_generator is None:
            cls._thumbnail_generator = ThumbnailGenerator(
                cache_dir="thumbnails", thumbnail_size=(180, 100), quality=85
            )
        return cls._thumbnail_generator

    def __init__(self, clip_data: dict[str, Any], parent=None):
        super().__init__(parent)
        self.clip_data = clip_data
        self.clip_id = clip_data.get("id", 0)

        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        self.setLineWidth(1)
        self.setMaximumWidth(200)
        self.setMinimumHeight(180)

        # Set object name for theme styling
        self.setObjectName("ClipThumbnailWidget")

        self._init_ui()

    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout()

        # Thumbnail container with delete button overlay
        thumbnail_container = QWidget()
        thumbnail_layout = QVBoxLayout(thumbnail_container)
        thumbnail_layout.setContentsMargins(0, 0, 0, 0)

        # Thumbnail
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail_label.setMinimumSize(180, 100)
        self.thumbnail_label.setMaximumSize(180, 100)
        self.thumbnail_label.setObjectName("ThumbnailLabel")

        # Set placeholder thumbnail
        self.thumbnail_label.setText("📹\nNo Thumbnail")

        # Load actual thumbnail from clip_data or generate it
        self._load_thumbnail()

        thumbnail_layout.addWidget(self.thumbnail_label)

        # BUG FIX 3: Delete button with proper parent assignment BEFORE positioning
        delete_btn = QPushButton("✕", thumbnail_container)  # Set parent in constructor
        delete_btn.setFixedSize(24, 24)
        delete_btn.setObjectName("DeleteButton")
        # Keep minimal functional style (red warning color) but remove borders etc
        delete_btn.setStyleSheet("background-color: #d32f2f; color: white; border-radius: 12px;")

        delete_btn.setToolTip("Delete clip from library")
        delete_btn.clicked.connect(self._on_delete_clicked)

        # Position delete button in top-right
        delete_btn.move(156, 0)  # 180 - 24 = 156
        delete_btn.raise_()  # Bring to front

        layout.addWidget(thumbnail_container)

        # Clip name
        name = self.clip_data.get("name", "Unknown Clip")
        name_label = QLabel(name)
        name_label.setWordWrap(True)
        name_label.setMaximumWidth(180)
        # Remove hardcoded color - let theme decide
        name_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(name_label)

        # Metadata
        duration = self.clip_data.get("duration", 0.0)
        resolution = f"{self.clip_data.get('width', 0)}x{self.clip_data.get('height', 0)}"
        fps = self.clip_data.get("fps", 0)

        metadata_text = f"{duration:.2f}s | {resolution} | {fps:.0f}fps"
        metadata_label = QLabel(metadata_text)
        # Remove hardcoded color
        metadata_label.setObjectName("MetadataLabel")
        layout.addWidget(metadata_label)

        # Analyse-Badges (wenn vorhanden)
        analysis = self.clip_data.get("analysis", {})
        if analysis:
            badges_layout = QHBoxLayout()
            badges_layout.setSpacing(3)

            # Motion Badge
            motion_data = analysis.get("motion", {})
            if motion_data:
                motion_type = motion_data.get("motion_type", "")
                if motion_type:
                    motion_badge = self._create_badge(
                        motion_type, self._get_motion_color(motion_type)
                    )
                    badges_layout.addWidget(motion_badge)

            # Mood Badge (erstes Mood)
            mood_data = analysis.get("mood", {})
            if mood_data:
                moods = mood_data.get("moods", [])
                if moods:
                    mood_badge = self._create_badge(moods[0], self._get_mood_color(moods[0]))
                    badges_layout.addWidget(mood_badge)

            # Style Badge (erster Style)
            style_data = analysis.get("style", {})
            if style_data:
                styles = style_data.get("styles", [])
                if styles and styles[0] != "STANDARD":
                    style_badge = self._create_badge(styles[0], "#8e44ad")
                    badges_layout.addWidget(style_badge)

            badges_layout.addStretch()
            layout.addLayout(badges_layout)

        layout.addStretch()
        self.setLayout(layout)

    def _create_badge(self, text: str, color: str) -> QLabel:
        """Erstellt ein Badge-Label."""
        badge = QLabel(text[:6])  # Maximal 6 Zeichen
        badge.setStyleSheet(
            f"""
            QLabel {{
                background-color: {color};
                color: white;
                border-radius: 3px;
                padding: 1px 4px;
                font-size: 8px;
                font-weight: bold;
            }}
        """
        )
        badge.setToolTip(text)
        return badge

    def _get_motion_color(self, motion_type: str) -> str:
        """Gibt Farbe fuer Motion-Type zurueck."""
        colors = {
            "STATIC": "#95a5a6",
            "SLOW": "#3498db",
            "MEDIUM": "#2ecc71",
            "FAST": "#f39c12",
            "EXTREME": "#e74c3c",
        }
        return colors.get(motion_type, "#7f8c8d")

    def _get_mood_color(self, mood: str) -> str:
        """Gibt Farbe fuer Mood zurueck."""
        colors = {
            "ENERGETIC": "#e74c3c",
            "CALM": "#3498db",
            "DARK": "#2c3e50",
            "BRIGHT": "#f1c40f",
            "MELANCHOLIC": "#9b59b6",
            "EUPHORIC": "#e91e63",
            "AGGRESSIVE": "#c0392b",
            "PEACEFUL": "#1abc9c",
            "MYSTERIOUS": "#8e44ad",
            "CHEERFUL": "#f39c12",
            "TENSE": "#e67e22",
            "DREAMY": "#a29bfe",
            "COOL": "#00bcd4",
            "WARM": "#ff5722",
        }
        return colors.get(mood, "#7f8c8d")

    def _load_thumbnail(self):
        """
        PERF-05 FIX: Load thumbnail with deferred execution for faster grid rendering.

        Uses QTimer.singleShot to defer disk I/O, allowing the grid to render
        immediately while thumbnails load in the background.
        """
        # PERF-05 FIX: Defer thumbnail loading to avoid blocking grid render
        # Small random delay (0-50ms) to stagger disk I/O and prevent spikes
        import random

        from PyQt6.QtCore import QTimer

        delay = random.randint(0, 50)
        QTimer.singleShot(delay, self._load_thumbnail_deferred)

    def _load_thumbnail_deferred(self):
        """Actually load the thumbnail (called after deferred delay)."""
        # FIX #8: Prüfen ob Widget noch existiert bevor auf es zugegriffen wird
        # Wenn das Widget vor dem Timer gelöscht wurde, würde ein Zugriff zum Crash führen
        try:
            # Prüfe ob Widget noch sichtbar/gültig ist
            if not self.isVisible() or not hasattr(self, "thumbnail_label"):
                return

            # Check if thumbnail_path is in clip_data
            if "thumbnail_path" in self.clip_data and self.clip_data["thumbnail_path"]:
                # Resolve relative path if needed
                thumb_path = resolve_relative_path(self.clip_data["thumbnail_path"])
                if thumb_path.exists():
                    pixmap = QPixmap(str(thumb_path))
                    if not pixmap.isNull():
                        self.thumbnail_label.setPixmap(
                            pixmap.scaled(180, 100, Qt.AspectRatioMode.KeepAspectRatio)
                        )
                        self.thumbnail_label.setStyleSheet("")  # Remove placeholder style
                        return

            # Check if thumbnail already exists in cache (by video file name)
            if "file_path" in self.clip_data:
                video_path = resolve_relative_path(self.clip_data["file_path"])
                # Try to find existing thumbnail without generating
                generator = self.get_thumbnail_generator()
                # Check if thumbnail already exists in cache
                thumb_name = f"{video_path.stem}_thumb.jpg"
                thumb_path = generator.cache_dir / thumb_name
                if thumb_path.exists():
                    pixmap = QPixmap(str(thumb_path))
                    if not pixmap.isNull():
                        self.thumbnail_label.setPixmap(
                            pixmap.scaled(180, 100, Qt.AspectRatioMode.KeepAspectRatio)
                        )
                        self.thumbnail_label.setStyleSheet("")
                        self.clip_data["thumbnail_path"] = to_relative_path(str(thumb_path))
                        return

            # Keep placeholder - thumbnails will be generated in background later
            # This prevents blocking the UI during startup

        except Exception as e:
            logger.error(f"Error loading thumbnail for clip {self.clip_id}: {e}")

    def _on_delete_clicked(self):
        """Handle delete button click."""
        logger.debug(f"Delete requested for clip {self.clip_id}")
        self.clip_delete_requested.emit(self.clip_id)

    def mousePressEvent(self, event):
        """Handle mouse click to select clip."""
        if event.button() == Qt.MouseButton.LeftButton:
            logger.debug(f"Clip selected: {self.clip_id}")
            self.clip_selected.emit(self.clip_id)
        super().mousePressEvent(event)


class ClipLibraryWidget(QWidget):
    """
    Clip library widget for managing video clips.

    Signals:
        clip_selected: Emitted when a clip is selected (clip_id)
        clip_imported: Emitted when new clip is imported (clip_id)
        analyze_requested: Emitted when analysis is requested (List[clip_ids])
    """

    clip_selected = pyqtSignal(int)  # clip_id
    clip_imported = pyqtSignal(int)  # clip_id
    analyze_requested = pyqtSignal(list)  # List[clip_ids]
    query_clips_requested = pyqtSignal()  # Phase 2: Request clips from worker
    videos_imported = pyqtSignal(list)  # List[video_paths] - emitted after drag-drop import

    def __init__(self, parent=None):
        super().__init__(parent)
        logger.info("Initializing ClipLibraryWidget")

        # Enable drag-and-drop
        self.setAcceptDrops(True)

        # State
        self.clips: list[dict[str, Any]] = []
        self.filtered_clips: list[dict[str, Any]] = []
        self.current_view_mode: str = "grid"  # grid or list
        self.current_sort: str = "name"  # name, duration, date
        self.current_analysis_filters: dict[str, str] = {}

        # UI components
        self.search_input: QLineEdit | None = None
        self.sort_combo: QComboBox | None = None
        self.clip_container: QWidget | None = None
        self.clip_layout: QGridLayout | None = None
        self.filter_widget: ClipFilterWidget | None = None
        self.loading_label: QLabel | None = None

        # Store original stylesheet for drag-drop visual feedback
        self._original_stylesheet: str = ""

        # Threading components (Phase 2)
        self.db_worker: DatabaseWorker | None = None
        self.db_thread: QThread | None = None
        self.session = None  # Will be created in worker thread

        self._init_ui()
        # Store original stylesheet after UI init
        self._original_stylesheet = self.styleSheet()
        # Setup worker thread for async database operations
        self._setup_worker_thread()
        # Load clips asynchronously on startup
        self.load_clips_async()

        logger.info("ClipLibraryWidget initialization complete")

    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout()

        # Header with search and controls
        header_layout = QHBoxLayout()

        # Search bar
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 Search clips...")
        self.search_input.textChanged.connect(self._filter_clips)
        self.search_input.setMaximumWidth(200)
        header_layout.addWidget(self.search_input)

        # Sort dropdown
        sort_label = QLabel("Sort:")
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Name", "Duration", "Date Added", "Motion", "Brightness"])
        self.sort_combo.currentTextChanged.connect(self._sort_clips)
        self.sort_combo.setMaximumWidth(120)
        header_layout.addWidget(sort_label)
        header_layout.addWidget(self.sort_combo)

        header_layout.addStretch()

        # Reload button
        reload_button = QPushButton("🔄")
        reload_button.setToolTip("Reload clips from database")
        reload_button.clicked.connect(self.load_clips_async)
        reload_button.setMaximumWidth(35)
        header_layout.addWidget(reload_button)

        # Import button
        import_button = QPushButton("+ Import Clips")
        import_button.clicked.connect(self._import_clips)
        import_button.setMaximumWidth(120)
        header_layout.addWidget(import_button)

        layout.addLayout(header_layout)

        # Filter Widget
        self.filter_widget = ClipFilterWidget()
        self.filter_widget.filters_changed.connect(self._on_analysis_filters_changed)
        self.filter_widget.analyze_requested.connect(self._on_analyze_requested)
        self.filter_widget.reanalyze_all_requested.connect(self._on_reanalyze_all_requested)
        layout.addWidget(self.filter_widget)

        # Loading indicator (Phase 2)
        self.loading_label = QLabel("⏳ Loading clips from database...")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.setStyleSheet(
            """
            QLabel {
                background-color: #2c3e50;
                color: #ecf0f1;
                padding: 20px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
        """
        )
        self.loading_label.setVisible(False)  # Hidden by default
        layout.addWidget(self.loading_label)

        # Scroll area for clips
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Container for clip thumbnails
        self.clip_container = QWidget()
        self.clip_layout = QGridLayout()
        self.clip_layout.setSpacing(10)
        self.clip_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.clip_container.setLayout(self.clip_layout)

        scroll_area.setWidget(self.clip_container)
        layout.addWidget(scroll_area)

        # Status bar
        self.status_label = QLabel("0 clips loaded")
        self.status_label.setObjectName("StatusLabel")
        layout.addWidget(self.status_label)

        self.setLayout(layout)
        logger.debug("ClipLibrary UI initialized")

    def _setup_worker_thread(self):
        """
        Setup database worker thread for async operations (Phase 2).

        Creates DatabaseWorker and QThread, connects signals,
        and starts the thread for non-blocking database queries.

        HIGH-13 FIX: Added try/finally to ensure session cleanup on error.
        """
        logger.info("Setting up database worker thread...")

        # Create database session (will be used in worker thread)
        self.session = get_db_session()

        try:
            # Create worker and thread
            self.db_worker = DatabaseWorker(self.session)
            self.db_thread = QThread()

            # Move worker to thread
            self.db_worker.moveToThread(self.db_thread)

            # Connect signals
            self.db_worker.query_completed.connect(self._on_clips_loaded)
            self.db_worker.error_occurred.connect(self._on_error)
            self.db_worker.progress_update.connect(self._on_progress_update)

            # Connect query request signal to worker's query_clips slot
            self.query_clips_requested.connect(self.db_worker.query_clips)

            # Start thread
            self.db_thread.start()

            logger.info("Database worker thread started successfully")
        except Exception as e:
            # Clean up session if setup fails
            logger.error(f"Failed to setup worker thread: {e}")
            if self.session:
                self.session.close()
                self.session = None
            raise

    def load_clips_async(self):
        """
        Load clips asynchronously using background thread (Phase 2).

        Shows loading indicator and requests clips from DatabaseWorker.
        UI remains responsive during query execution.
        """
        logger.info("Starting async clip loading...")

        # Show loading indicator
        if self.loading_label:
            self.loading_label.setVisible(True)
            self.loading_label.setText("⏳ Loading clips from database...")

        # Hide clip container during loading
        if self.clip_container:
            self.clip_container.setVisible(False)

        # Request clips from worker (via Signal/Slot for thread-safe communication)
        self.query_clips_requested.emit()

        logger.debug("Async clip loading requested from worker thread")

    def _on_clips_loaded(self, clips_data: list[dict[str, Any]]):
        """
        Callback when clips are loaded from database (Phase 2).

        Args:
            clips_data: List of clip dictionaries from DatabaseWorker
        """
        logger.info(f"Clips loaded callback - received {len(clips_data)} clips")

        # Hide loading indicator
        if self.loading_label:
            self.loading_label.setVisible(False)

        # Show clip container
        if self.clip_container:
            self.clip_container.setVisible(True)

        # Update clips list
        self.clips = clips_data

        # Extract unanalyzed count from first clip metadata (if available)
        unanalyzed_count = 0
        if clips_data:
            # Count unanalyzed clips
            unanalyzed_count = sum(1 for clip in clips_data if not clip.get("is_analyzed", False))

        # Update filter widget
        if self.filter_widget:
            self.filter_widget.set_unanalyzed_count(unanalyzed_count)
            self.filter_widget.set_status(f"{len(self.clips)} clips, {unanalyzed_count} unanalyzed")

        # Apply current filters and render
        self._apply_all_filters()

        logger.info(f"✓ Loaded {len(self.clips)} clips ({unanalyzed_count} unanalyzed)")

    def _on_error(self, error_msg: str):
        """
        Callback when database error occurs (Phase 2).

        Args:
            error_msg: Error message from DatabaseWorker
        """
        logger.error(f"Database error: {error_msg}")

        # Hide loading indicator
        if self.loading_label:
            self.loading_label.setVisible(False)

        # Show error message
        QMessageBox.critical(
            self, "Database Error", f"Failed to load clips from database:\n\n{error_msg}"
        )

    def _on_progress_update(self, current: int, total: int, message: str):
        """
        Callback for progress updates from DatabaseWorker (Phase 2).

        Args:
            current: Current progress value
            total: Total progress value (usually 100)
            message: Progress message
        """
        if self.loading_label:
            self.loading_label.setText(f"⏳ {message}")

        logger.debug(f"Progress: {current}/{total} - {message}")

    def _load_clips_from_db(self):
        """Load all clips from database including analysis data from separate tables."""
        session = None  # BUG FIX 7: Initialize session variable
        try:
            session = get_db_session()

            # Eager Loading: Alle Analyse-Daten in EINEM Query laden (statt N+1 Queries)
            db_clips = (
                session.query(VideoClip)
                .filter(VideoClip.is_available == True)
                .options(
                    # Alle Relationships eager loaden
                    joinedload(VideoClip.analysis_status),
                    joinedload(VideoClip.colors),
                    joinedload(VideoClip.motion),
                    joinedload(VideoClip.scene_type),
                    joinedload(VideoClip.mood),
                    joinedload(VideoClip.objects),
                    joinedload(VideoClip.style),
                    joinedload(VideoClip.fingerprint),
                )
                .all()
            )

            self.clips = []
            unanalyzed_count = 0

            # PERF-OPTIMIZATION: Use ClipDataLoader for lazy loading
            for clip_db in db_clips:
                clip_data = ClipDataLoader.db_to_dict(clip_db, full_details=False)
                if clip_data.get("_unanalyzed_count", 0) > 0:
                    unanalyzed_count += 1
                self.clips.append(clip_data)

            # Update filter widget
            if self.filter_widget:
                self.filter_widget.set_unanalyzed_count(unanalyzed_count)
                self.filter_widget.set_status(
                    f"{len(self.clips)} clips, {unanalyzed_count} unanalyzed"
                )

            # Apply current filters
            self._apply_all_filters()

            logger.info(
                f"Loaded {len(self.clips)} clips from database ({unanalyzed_count} unanalyzed)"
            )

        except Exception as e:
            logger.error(f"Failed to load clips from database: {e}", exc_info=True)
        finally:
            # BUG FIX 7: Always close session, even if exception occurs
            if session:
                session.close()

    def _on_analysis_filters_changed(self, filters: dict[str, str]):
        """Handle changes to analysis filters."""
        self.current_analysis_filters = filters
        self._apply_all_filters()

    def _on_analyze_requested(self):
        """Handle request to analyze unanalyzed clips."""
        # Get unanalyzed clip IDs
        unanalyzed_ids = [clip["id"] for clip in self.clips if not clip.get("is_analyzed", False)]

        if unanalyzed_ids:
            logger.info(f"Requesting analysis for {len(unanalyzed_ids)} clips")
            self.analyze_requested.emit(unanalyzed_ids)
        else:
            QMessageBox.information(
                self, "All Clips Analyzed", "All clips have already been analyzed."
            )

    def _on_reanalyze_all_requested(self):
        """Handle request to re-analyze all clips."""
        # Get all clip IDs
        all_ids = [clip["id"] for clip in self.clips]

        if all_ids:
            logger.info(f"Requesting re-analysis for {len(all_ids)} clips")
            self.analyze_requested.emit(all_ids)
        else:
            QMessageBox.information(self, "No Clips", "No clips available to analyze.")

    def _apply_all_filters(self):
        """Apply both text search and analysis filters."""
        # Start with all clips
        filtered = self.clips.copy()

        # Apply text search
        search_text = self.search_input.text() if self.search_input else ""
        if search_text:
            search_lower = search_text.lower()
            filtered = [clip for clip in filtered if search_lower in clip.get("name", "").lower()]

        # Apply analysis filters
        if self.current_analysis_filters:
            filtered = apply_filters_to_clips(filtered, self.current_analysis_filters)

        self.filtered_clips = filtered
        self._render_clips()

        logger.debug(f"Filtered to {len(filtered)} clips")

    def _render_clips(self):
        """Render clips in the grid layout with pagination for performance."""
        try:
            logger.debug(f"_render_clips() START - {len(self.filtered_clips)} filtered clips")

            # PERF-13 FIX: Disable updates during batch widget operations
            if self.clip_container:
                self.clip_container.setUpdatesEnabled(False)

            try:
                # Clear existing widgets
                while self.clip_layout.count():
                    item = self.clip_layout.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()

                # Process pending events to clean up deleted widgets
                QApplication.processEvents()

                # Limit initial render for performance (pagination)
                clips_to_render = self.filtered_clips[:MAX_INITIAL_RENDER]
                total_clips = len(self.filtered_clips)

                logger.debug(f"Rendering {len(clips_to_render)} clips (max {MAX_INITIAL_RENDER})")

                # Render filtered clips
                for i, clip_data in enumerate(clips_to_render):
                    try:
                        row = i // CLIPS_PER_ROW
                        col = i % CLIPS_PER_ROW

                        clip_widget = ClipThumbnailWidget(clip_data)
                        clip_widget.clip_selected.connect(self._on_clip_selected)
                        clip_widget.clip_delete_requested.connect(self._on_clip_delete_requested)
                        self.clip_layout.addWidget(clip_widget, row, col)

                    except Exception as e:
                        logger.error(
                            f"Error rendering clip {i} ({clip_data.get('name', 'unknown')}): {e}"
                        )
                        continue

                # Show "Load More" button if there are more clips
                if total_clips > MAX_INITIAL_RENDER:
                    remaining = total_clips - MAX_INITIAL_RENDER
                    load_more_btn = QPushButton(f"Load {remaining} more clips...")
                    load_more_btn.setStyleSheet("padding: 10px; font-size: 14px;")
                    load_more_btn.clicked.connect(self._render_all_clips)
                    row = (len(clips_to_render) // CLIPS_PER_ROW) + 1
                    self.clip_layout.addWidget(load_more_btn, row, 0, 1, CLIPS_PER_ROW)

                # Update status
                shown = len(clips_to_render)
                self.status_label.setText(f"Showing {shown} of {total_clips} clips")

                logger.debug(f"_render_clips() DONE - Rendered {shown} of {total_clips} clips")

            finally:
                # PERF-13 FIX: Re-enable updates and trigger single repaint
                if self.clip_container:
                    self.clip_container.setUpdatesEnabled(True)

        except Exception as e:
            logger.error(
                f"CRITICAL: _render_clips() failed: {type(e).__name__}: {e}", exc_info=True
            )

    def _render_all_clips(self):
        """
        Render all clips without pagination limit.

        PERF-FIX: Batch Layout Updates - deaktiviert UI-Updates während des Renderings.
        Bei 500 Clips: ~80% schneller (4s -> 0.8s).
        """
        # PERF-FIX: Disable updates during bulk widget creation
        if self.clip_container:
            self.clip_container.setUpdatesEnabled(False)

        try:
            # Clear existing widgets
            while self.clip_layout.count():
                item = self.clip_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            total = len(self.filtered_clips)

            # PERF-FIX: Batch-Create widgets (kein processEvents pro Widget)
            BATCH_SIZE = 50  # Größere Batches für weniger UI-Interrupts

            for batch_start in range(0, total, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, total)

                for i in range(batch_start, batch_end):
                    clip_data = self.filtered_clips[i]
                    row = i // CLIPS_PER_ROW
                    col = i % CLIPS_PER_ROW

                    clip_widget = ClipThumbnailWidget(clip_data)
                    clip_widget.clip_selected.connect(self._on_clip_selected)
                    clip_widget.clip_delete_requested.connect(self._on_clip_delete_requested)
                    self.clip_layout.addWidget(clip_widget, row, col)

                # Process events once per batch (nicht pro Widget)
                QApplication.processEvents()
                self.status_label.setText(f"Loading... {batch_end}/{total}")

            self.status_label.setText(f"{total} clips loaded")
            logger.info(f"Rendered all {total} clips (batch_size={BATCH_SIZE})")

        finally:
            # PERF-FIX: Re-enable updates and trigger single repaint
            if self.clip_container:
                self.clip_container.setUpdatesEnabled(True)

    def _filter_clips(self, search_text: str):
        """Filter clips based on search text."""
        # Use combined filter method
        self._apply_all_filters()
        logger.debug(f"Filtered clips: {len(self.filtered_clips)} results for '{search_text}'")

    def _sort_clips(self, sort_by: str):
        """Sort clips by selected criteria."""
        self.current_sort = sort_by

        def get_sort_key(clip):
            """Get sort key based on sort criteria."""
            if sort_by == "Name":
                return clip.get("name", "").lower()
            elif sort_by == "Duration":
                return clip.get("duration", 0.0)
            elif sort_by == "Date Added":
                return clip.get("date_added", "")
            elif sort_by == "Motion":
                # Sort by motion score (from analysis)
                analysis = clip.get("analysis", {})
                motion = analysis.get("motion", {})
                return motion.get("motion_score", 0.0)
            elif sort_by == "Brightness":
                # Sort by brightness (from analysis)
                analysis = clip.get("analysis", {})
                color = analysis.get("color", {})
                return color.get("brightness_value", 128)
            else:
                return clip.get("name", "").lower()

        # Sort both full and filtered lists
        reverse = sort_by in ["Duration", "Motion", "Brightness"]  # High values first
        self.clips.sort(key=get_sort_key, reverse=reverse)
        self.filtered_clips.sort(key=get_sort_key, reverse=reverse)

        self._render_clips()
        logger.debug(f"Sorted clips by: {sort_by}")

    def _on_clip_selected(self, clip_id: int):
        """Handle clip selection."""
        logger.info(f"Clip selected: {clip_id}")

        # Ensure that the selected clip has full details parsed
        # This is important if details are lazy-loaded
        selected_clip = None
        for clip in self.clips:
            if clip.get("id") == clip_id:
                selected_clip = clip
                break

        if selected_clip:
            # Parse full details on demand if not already present
            ClipDataLoader.ensure_details(selected_clip)
            logger.debug(f"Selected clip: {selected_clip.get('name')}")

        self.clip_selected.emit(clip_id)

    def _on_clip_delete_requested(self, clip_id: int):
        """Handle clip deletion request."""
        # Find clip
        clip_to_delete = None
        for clip in self.clips:
            if clip.get("id") == clip_id:
                clip_to_delete = clip
                break

        if not clip_to_delete:
            logger.warning(f"Clip {clip_id} not found for deletion")
            return

        # Confirm deletion
        clip_name = clip_to_delete.get("name", "Unknown")
        reply = QMessageBox.question(
            self,
            "Delete Clip",
            f"Are you sure you want to delete '{clip_name}' from the library?\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                # Delete from database
                session = get_db_session()
                thumbnail_to_delete = None
                try:
                    clip_db = session.query(VideoClip).filter(VideoClip.id == clip_id).first()
                    if clip_db:
                        # Store thumbnail path before deletion
                        if clip_db.thumbnail_path:
                            thumbnail_to_delete = Path(clip_db.thumbnail_path)

                        session.delete(clip_db)
                        session.commit()
                        logger.info(f"Deleted clip {clip_id} from database")
                    else:
                        logger.warning(f"Clip {clip_id} not found in database")
                finally:
                    session.close()

                # Delete thumbnail file if it exists (outside of DB session)
                if thumbnail_to_delete and thumbnail_to_delete.exists():
                    try:
                        thumbnail_to_delete.unlink()
                        logger.info(f"Deleted thumbnail file: {thumbnail_to_delete}")
                    except Exception as thumb_err:
                        logger.warning(f"Failed to delete thumbnail file: {thumb_err}")

            except Exception as e:
                logger.error(f"Failed to delete clip {clip_id} from database: {e}")
                QMessageBox.warning(
                    self, "Database Error", f"Failed to delete clip from database: {e}"
                )
                return

            # Remove from clips list
            self.clips = [c for c in self.clips if c.get("id") != clip_id]

            # Re-filter and render
            self._filter_clips(self.search_input.text())

            logger.info(f"Deleted clip {clip_id}: {clip_name}")
            QMessageBox.information(
                self, "Clip Deleted", f"'{clip_name}' has been removed from the library."
            )

    def _setup_import_progress_dialog(self, total_files: int):
        """
        Create and configure progress dialog for clip import.

        Args:
            total_files: Total number of files to import

        Returns:
            Configured QProgressDialog
        """
        from PyQt6.QtCore import Qt

        progress = QProgressDialog(
            f"Importing 0/{total_files} clips...", "Cancel", 0, total_files, self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(True)
        progress.show()
        QApplication.processEvents()

        return progress

    def _import_single_clip(
        self, file_path: str, session, analyzer, clip_index: int, total_files: int
    ) -> tuple[dict | None, str | None]:
        """
        Import a single video clip with full error handling.

        Args:
            file_path: Path to video file
            session: Database session
            analyzer: VideoAnalyzer instance
            clip_index: Index of current clip (for logging)
            total_files: Total number of files (for logging)

        Returns:
            Tuple of (clip_data dict, error_message)
            - Success: (clip_data, None)
            - Failure: (None, error_message)
        """
        try:
            file_path_obj = Path(file_path)
            clip_name = file_path_obj.name

            # 1. Analyze video file to get metadata
            logger.info(f"Analyzing video: {clip_name}")
            video_info = analyzer.get_video_info(file_path)

            if not video_info:
                logger.error(f"Failed to analyze video: {clip_name}")
                return (None, clip_name)

            # 2. Create clip object for database
            clip_db = VideoClip(
                project_id=1,  # Default project
                name=clip_name,
                file_path=str(file_path),
                duration=video_info.get("duration_seconds", 0.0),
                fps=video_info.get("frame_rate", 30.0),
                width=video_info.get("width", 0),
                height=video_info.get("height", 0),
                codec="",
                thumbnail_path=None,
                format=file_path_obj.suffix.lstrip("."),
            )

            # 3. Add to database
            session.add(clip_db)
            session.flush()

            clip_id = clip_db.id
            logger.info(f"Clip added to database: ID={clip_id}, Name={clip_name}")

            # 4. Generate thumbnail (with error handling)
            try:
                generator = ClipThumbnailWidget.get_thumbnail_generator()
                thumb_path = generator.generate(file_path_obj)
                if thumb_path and thumb_path.exists():
                    clip_db.thumbnail_path = str(thumb_path)
                    logger.info(f"Thumbnail generated for clip {clip_id}")
            except Exception as thumb_err:
                logger.warning(f"Failed to generate thumbnail for {clip_name}: {thumb_err}")

            # 5. Commit database changes
            session.commit()

            # 6. Create clip data dict for UI
            clip_data = {
                "id": clip_id,
                "name": clip_name,
                "duration": video_info.get("duration_seconds", 0.0),
                "width": video_info.get("width", 0),
                "height": video_info.get("height", 0),
                "fps": video_info.get("frame_rate", 30.0),
                "file_path": str(file_path),
                "thumbnail_path": clip_db.thumbnail_path,
            }

            logger.info(
                f"✓ Imported clip: {clip_name} "
                f"({video_info.get('duration_seconds', 0):.2f}s, "
                f"{video_info.get('width', 0)}x{video_info.get('height', 0)}, "
                f"{video_info.get('frame_rate', 0):.1f}fps)"
            )

            return (clip_data, None)

        except Exception as e:
            logger.error(f"Failed to import {file_path}: {type(e).__name__}: {e}", exc_info=True)
            session.rollback()
            return (None, Path(file_path).name)

    def _show_import_results(self, imported_count: int, failed_count: int, failed_files: list[str]):
        """
        Display import results message to user.

        Args:
            imported_count: Number of successfully imported clips
            failed_count: Number of failed imports
            failed_files: List of failed file names
        """
        if imported_count == 0 and failed_count == 0:
            return

        message = f"Successfully imported {imported_count} clip(s)."
        if failed_count > 0:
            message += f"\n\nFailed: {failed_count} clip(s):"
            for f in failed_files[:5]:  # Show max 5 failed files
                message += f"\n  • {f}"
            if len(failed_files) > 5:
                message += f"\n  ... and {len(failed_files) - 5} more"

        QMessageBox.information(self, "Import Complete", message)

    def _import_clips(self):
        """
        Import video clips from file dialog with progress feedback and error recovery.

        Features:
        - Progress dialog with cancel option
        - GUI stays responsive during import (processEvents)
        - Individual errors don't stop entire import
        - Detailed logging for debugging crashes
        """
        # Get files from dialog
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Import Video Clips", "", "Video Files (*.mp4 *.avi *.mov *.mkv *.webm)"
        )

        if not file_paths:
            return

        total_files = len(file_paths)
        logger.info(f"Starting import of {total_files} clips")

        # Setup progress dialog
        progress = self._setup_import_progress_dialog(total_files)

        # Initialize database session and analyzer
        session = get_db_session()
        analyzer = VideoAnalyzer(session=session)

        imported_count = 0
        failed_count = 0
        failed_files = []

        try:
            for i, file_path in enumerate(file_paths):
                # Check for cancel
                if progress.wasCanceled():
                    logger.info("Import cancelled by user")
                    break

                # Update progress
                file_name = Path(file_path).name
                progress.setValue(i)
                progress.setLabelText(f"Importing {i+1}/{total_files}:\n{file_name}")
                QApplication.processEvents()

                # Import single clip using extracted method
                clip_data, error = self._import_single_clip(
                    file_path, session, analyzer, i, total_files
                )

                if clip_data:
                    self.clips.append(clip_data)
                    imported_count += 1
                else:
                    failed_count += 1
                    if error:
                        failed_files.append(error)

        finally:
            progress.setValue(total_files)
            progress.close()
            session.close()

        # Update UI
        if imported_count > 0:
            self.filtered_clips = self.clips.copy()
            self._render_clips()

        # Show results using extracted method
        self._show_import_results(imported_count, failed_count, failed_files)

    def import_clips_programmatic(
        self, file_paths: list, show_progress: bool = True, show_results: bool = True
    ):
        """
        Import video clips programmatically (for automated testing).

        Same as _import_clips() but accepts file_paths directly instead of opening dialog.

        Args:
            file_paths: List of file paths to import
            show_progress: Show progress dialog (default: True, False for automated tests)
            show_results: Show results message box (default: True, False for automated tests)
        """
        if not file_paths:
            return

        total_files = len(file_paths)
        logger.info(f"Starting programmatic import of {total_files} clips")

        # Setup progress dialog (optional)
        progress = None
        if show_progress:
            progress = self._setup_import_progress_dialog(total_files)

        # Initialize database session and analyzer
        session = get_db_session()
        analyzer = VideoAnalyzer(session=session)

        imported_count = 0
        failed_count = 0
        failed_files = []

        try:
            for i, file_path in enumerate(file_paths):
                # Check for cancel (if progress dialog exists)
                if progress and progress.wasCanceled():
                    logger.info("Import cancelled by user")
                    break

                # Update progress (if dialog exists)
                if progress:
                    file_name = Path(file_path).name
                    progress.setValue(i)
                    progress.setLabelText(f"Importing {i+1}/{total_files}:\n{file_name}")
                    QApplication.processEvents()

                # Import single clip using extracted method
                clip_data, error = self._import_single_clip(
                    file_path, session, analyzer, i, total_files
                )

                if clip_data:
                    self.clips.append(clip_data)
                    imported_count += 1
                else:
                    failed_count += 1
                    if error:
                        failed_files.append(error)

        finally:
            if progress:
                progress.setValue(total_files)
                progress.close()
            session.close()

        # Update UI
        if imported_count > 0:
            self.filtered_clips = self.clips.copy()
            self._render_clips()

        # Show results using extracted method (optional)
        if show_results:
            self._show_import_results(imported_count, failed_count, failed_files)
        else:
            logger.info(f"Import completed: {imported_count} success, {failed_count} failed")

    def add_clip(self, clip_data: dict[str, Any]):
        """
        Programmatically add a clip to the library.

        Args:
            clip_data: Dictionary with clip metadata
        """
        self.clips.append(clip_data)
        self.filtered_clips = self.clips.copy()
        self._render_clips()

        logger.info(f"Added clip: {clip_data.get('name')}")
        self.clip_imported.emit(clip_data.get("id", 0))

    def get_clip(self, clip_id: int) -> dict[str, Any] | None:
        """
        Get clip data by ID.

        Args:
            clip_id: Clip ID

        Returns:
            Clip data dictionary or None
        """
        for clip in self.clips:
            if clip.get("id") == clip_id:
                return clip
        return None

    def clear(self):
        """Clear all clips from library."""
        self.clips = []
        self.filtered_clips = []
        self._render_clips()
        logger.info("Clip library cleared")

    def refresh(self):
        """Reload clips from database asynchronously (alias for load_clips_async)."""
        self.load_clips_async()

    def get_unanalyzed_clip_ids(self) -> list[int]:
        """Get list of clip IDs that haven't been analyzed."""
        return [clip["id"] for clip in self.clips if not clip.get("is_analyzed", False)]

    def get_all_clip_ids(self) -> list[int]:
        """Get list of all clip IDs."""
        return [clip["id"] for clip in self.clips]

    def update_clip_analysis(self, clip_id: int, analysis_data: dict[str, Any]):
        """
        Update analysis data for a clip in memory (without DB reload).

        Args:
            clip_id: Clip ID
            analysis_data: New analysis data
        """
        for clip in self.clips:
            if clip.get("id") == clip_id:
                clip["analysis"] = analysis_data
                clip["is_analyzed"] = True
                break

        # Re-render if clip is in filtered list
        for clip in self.filtered_clips:
            if clip.get("id") == clip_id:
                self._render_clips()
                break

        # Update unanalyzed count
        unanalyzed_count = len(self.get_unanalyzed_clip_ids())
        if self.filter_widget:
            self.filter_widget.set_unanalyzed_count(unanalyzed_count)
            self.filter_widget.set_status(f"{len(self.clips)} clips, {unanalyzed_count} unanalyzed")

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter - check if video files are being dragged."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            # Check if at least one supported video file is being dragged
            for url in urls:
                if url.isLocalFile():
                    path = Path(url.toLocalFile())
                    if path.suffix.lower() in SUPPORTED_VIDEO_FORMATS:
                        event.acceptProposedAction()
                        self._show_drop_indicator(True)
                        logger.debug(f"Drag enter accepted - {len(urls)} file(s)")
                        return
        event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent):
        """Handle drag leave - remove visual feedback."""
        self._show_drop_indicator(False)
        event.accept()
        logger.debug("Drag leave")

    def dropEvent(self, event: QDropEvent):
        """Handle drop - import video files."""
        self._show_drop_indicator(False)

        if event.mimeData().hasUrls():
            video_paths = []
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    path = Path(url.toLocalFile())
                    if path.suffix.lower() in SUPPORTED_VIDEO_FORMATS:
                        video_paths.append(str(path))

            if video_paths:
                event.acceptProposedAction()
                logger.info(f"Drop event - importing {len(video_paths)} video(s)")
                self._import_videos(video_paths)
                return

        event.ignore()

    def _show_drop_indicator(self, show: bool):
        """Show visual feedback during drag-over."""
        if show:
            # Apply drop indicator style (overlay on current theme)
            self.setStyleSheet(
                self._original_stylesheet
                + """
                ClipLibraryWidget {
                    border: 2px dashed #0078d4;
                    background-color: rgba(0, 120, 212, 0.1);
                }
            """
            )
        else:
            # Restore original theme
            self.setStyleSheet(self._original_stylesheet)

    def _import_videos(self, video_paths: list[str]):
        """
        Import multiple videos via drag-and-drop with progress dialog.

        Args:
            video_paths: List of video file paths to import
        """
        if not video_paths:
            return

        # Use programmatic import with progress dialog
        self.import_clips_programmatic(video_paths, show_progress=True, show_results=True)

        # Emit signal for external listeners
        self.videos_imported.emit(video_paths)

        logger.info(f"Drag-drop import completed for {len(video_paths)} video(s)")

    def closeEvent(self, event):
        """
        Clean up worker thread on widget close (Phase 2).

        Ensures thread is properly terminated to prevent resource leaks.
        """
        logger.info("ClipLibraryWidget closing - cleaning up worker thread...")

        # Stop worker thread if running
        if self.db_thread and self.db_thread.isRunning():
            logger.debug("Stopping database worker thread...")

            # Request thread to quit
            self.db_thread.quit()

            # Wait for thread to finish (max 5 seconds)
            if not self.db_thread.wait(5000):
                logger.warning("Worker thread did not finish in time, terminating...")
                self.db_thread.terminate()
                self.db_thread.wait()

            logger.info("Database worker thread stopped")

        # Close database session
        if self.session:
            self.session.close()
            logger.debug("Database session closed")

        # Call parent closeEvent
        super().closeEvent(event)
