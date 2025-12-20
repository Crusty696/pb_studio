"""
Clip Library Widget for PB_studio

Production-ready PyQt6 widget for video clip management with thumbnails and metadata.

Features:
- Scrollable list/grid view of video clips
- Thumbnail display with caching
- Metadata display (filename, duration, resolution, FPS, codec)
- Drag-and-drop functionality to timeline
- Search and filter capabilities
- Performance-optimized thumbnail loading

Usage:
    from pb_studio.gui.widgets import ClipLibraryWidget

    library = ClipLibraryWidget()
    library.load_clips(video_clips)
    library.clip_selected.connect(on_clip_selected)
    library.clip_dragged.connect(on_clip_dragged)

Dependencies:
- PyQt6
- VideoManager (Task 23)
"""

import json
from pathlib import Path

from PyQt6.QtCore import QByteArray, QMimeData, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QDrag, QFont, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...video import VideoManager


class ClipLibraryWidget(QWidget):
    """
    High-performance video clip library with thumbnails and drag-and-drop.

    Signals:
        clip_selected(dict): Emitted when user selects a clip (clip metadata dict)
        clip_dragged(dict): Emitted when user starts dragging a clip (clip metadata dict)
        filter_changed(str): Emitted when search/filter changes (filter text)
    """

    # Signals
    clip_selected = pyqtSignal(dict)
    clip_dragged = pyqtSignal(dict)
    filter_changed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None):
        """
        Initialize the ClipLibraryWidget.

        Args:
            parent: Parent widget (optional)
        """
        super().__init__(parent)

        # Clip data
        self.clips: list[dict] = []
        self.filtered_clips: list[dict] = []
        self.video_manager = VideoManager()

        # Thumbnail cache
        self.thumbnail_cache: dict[str, QPixmap] = {}
        self.thumbnail_size = QSize(160, 90)  # 16:9 aspect ratio

        # Filter state
        self.current_filter: str = ""
        self.current_sort: str = "name"  # name, duration, date

        # Styling
        self.background_color = QColor(30, 30, 30)
        self.item_background = QColor(40, 40, 40)
        self.item_hover = QColor(50, 50, 50)
        self.item_selected = QColor(60, 100, 140)
        self.text_color = QColor(220, 220, 220)
        self.metadata_color = QColor(150, 150, 150)

        # Setup UI
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Search and filter toolbar
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setSpacing(5)

        # Search field
        self.search_field = QLineEdit()
        self.search_field.setPlaceholderText("Search clips...")
        self.search_field.textChanged.connect(self._on_search_changed)
        toolbar_layout.addWidget(self.search_field)

        # Sort dropdown
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Name", "Duration", "Date Added"])
        self.sort_combo.currentIndexChanged.connect(self._on_sort_changed)
        toolbar_layout.addWidget(QLabel("Sort:"))
        toolbar_layout.addWidget(self.sort_combo)

        # Clear filter button
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_filter)
        toolbar_layout.addWidget(clear_btn)

        layout.addLayout(toolbar_layout)

        # Clip list widget
        self.clip_list = QListWidget()
        self.clip_list.setIconSize(self.thumbnail_size)
        self.clip_list.setSpacing(5)
        self.clip_list.setDragEnabled(True)
        self.clip_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.clip_list.itemClicked.connect(self._on_clip_clicked)
        self.clip_list.itemDoubleClicked.connect(self._on_clip_double_clicked)

        # Enable custom drag-and-drop
        self.clip_list.startDrag = self._start_drag

        layout.addWidget(self.clip_list)

        # Status label
        self.status_label = QLabel("No clips loaded")
        self.status_label.setStyleSheet(f"color: {self.metadata_color.name()};")
        layout.addWidget(self.status_label)

        # Widget setup
        self.setMinimumWidth(300)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

    def load_clips(self, clips: list[dict]) -> None:
        """
        Load video clips into the library.

        Args:
            clips: List of clip metadata dicts with keys:
                   'id', 'name', 'file_path', 'duration', 'width', 'height',
                   'fps', 'codec', 'thumbnail_path' (optional)
        """
        self.clips = clips
        self.filtered_clips = clips.copy()
        self._refresh_clip_list()
        self._update_status()

    def add_clip(self, clip: dict) -> None:
        """
        Add a single clip to the library.

        Args:
            clip: Clip metadata dict
        """
        self.clips.append(clip)

        # Check if clip passes current filter
        if self._passes_filter(clip):
            self.filtered_clips.append(clip)
            self._add_clip_item(clip)

        self._update_status()

    def remove_clip(self, clip_id: int | str) -> None:
        """
        Remove a clip from the library.

        Args:
            clip_id: ID of the clip to remove
        """
        # Remove from main list
        self.clips = [c for c in self.clips if c.get("id") != clip_id]

        # Remove from filtered list
        self.filtered_clips = [c for c in self.filtered_clips if c.get("id") != clip_id]

        # Remove from thumbnail cache
        cache_key = str(clip_id)
        if cache_key in self.thumbnail_cache:
            del self.thumbnail_cache[cache_key]

        # Refresh display
        self._refresh_clip_list()
        self._update_status()

    def filter_clips(self, search_text: str) -> None:
        """
        Filter clips by search text.

        Args:
            search_text: Text to filter by (searches name, codec, resolution)
        """
        self.current_filter = search_text.lower()

        # Apply filter
        self.filtered_clips = [clip for clip in self.clips if self._passes_filter(clip)]

        # Refresh display
        self._refresh_clip_list()
        self._update_status()

        # Emit signal
        self.filter_changed.emit(search_text)

    def clear_library(self) -> None:
        """Clear all clips from the library."""
        self.clips.clear()
        self.filtered_clips.clear()
        self.thumbnail_cache.clear()
        self.clip_list.clear()
        self._update_status()

    def get_selected_clip(self) -> dict | None:
        """
        Get currently selected clip metadata.

        Returns:
            Clip metadata dict or None if no selection
        """
        current_item = self.clip_list.currentItem()
        if current_item:
            clip_id = current_item.data(Qt.ItemDataRole.UserRole)
            return self._get_clip_by_id(clip_id)
        return None

    def _refresh_clip_list(self) -> None:
        """Refresh the clip list display."""
        # PERF-13 FIX: Disable updates during batch operations
        self.clip_list.setUpdatesEnabled(False)
        try:
            self.clip_list.clear()

            # Sort clips
            sorted_clips = self._sort_clips(self.filtered_clips)

            # Add items
            for clip in sorted_clips:
                self._add_clip_item(clip)
        finally:
            # Re-enable updates and trigger single repaint
            self.clip_list.setUpdatesEnabled(True)

    def _add_clip_item(self, clip: dict) -> None:
        """
        Add a clip item to the list.

        Args:
            clip: Clip metadata dict
        """
        # Create list item
        item = QListWidgetItem()
        item.setSizeHint(QSize(self.thumbnail_size.width() + 10, self.thumbnail_size.height() + 60))

        # Store clip ID in item data
        item.setData(Qt.ItemDataRole.UserRole, clip.get("id"))

        # Set thumbnail icon
        thumbnail = self._get_or_create_thumbnail(clip)
        if thumbnail:
            item.setIcon(QIcon(thumbnail))

        # Set display text with metadata
        display_text = self._format_clip_text(clip)
        item.setText(display_text)

        # Set tooltip with full info
        tooltip = self._format_clip_tooltip(clip)
        item.setToolTip(tooltip)

        # Add to list
        self.clip_list.addItem(item)

    def _get_or_create_thumbnail(self, clip: dict) -> QPixmap | None:
        """
        Get cached thumbnail or create new one.

        Args:
            clip: Clip metadata dict

        Returns:
            QPixmap thumbnail or None
        """
        clip_id = str(clip.get("id", ""))

        # Check cache first
        if clip_id in self.thumbnail_cache:
            return self.thumbnail_cache[clip_id]

        # Try to load from thumbnail_path
        thumbnail_path = clip.get("thumbnail_path")
        if thumbnail_path and Path(thumbnail_path).exists():
            pixmap = QPixmap(thumbnail_path)
            if not pixmap.isNull():
                # Scale to thumbnail size
                pixmap = pixmap.scaled(
                    self.thumbnail_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.thumbnail_cache[clip_id] = pixmap
                return pixmap

        # Generate thumbnail from video file
        file_path = clip.get("file_path")
        if file_path and Path(file_path).exists():
            thumbnail = self.video_manager.generate_thumbnail(
                Path(file_path),
                time_offset=1.0,  # 1 second into video
            )
            if thumbnail:
                pixmap = QPixmap(thumbnail)
                if not pixmap.isNull():
                    pixmap = pixmap.scaled(
                        self.thumbnail_size,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    self.thumbnail_cache[clip_id] = pixmap
                    return pixmap

        # Return placeholder if thumbnail generation failed
        return self._create_placeholder_thumbnail(clip)

    def _create_placeholder_thumbnail(self, clip: dict) -> QPixmap:
        """
        Create a placeholder thumbnail.

        Args:
            clip: Clip metadata dict

        Returns:
            QPixmap placeholder
        """
        pixmap = QPixmap(self.thumbnail_size)
        pixmap.fill(QColor(60, 60, 60))

        painter = QPainter(pixmap)
        painter.setPen(QPen(self.text_color))
        painter.setFont(QFont("Arial", 10))

        # Draw resolution text
        resolution = f"{clip.get('width', '?')}x{clip.get('height', '?')}"
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, resolution)

        painter.end()

        return pixmap

    def _format_clip_text(self, clip: dict) -> str:
        """
        Format clip text for display.

        Args:
            clip: Clip metadata dict

        Returns:
            Formatted display text
        """
        name = Path(clip.get("file_path", "Unknown")).name
        duration = clip.get("duration", 0)
        width = clip.get("width", 0)
        height = clip.get("height", 0)
        fps = clip.get("fps", 0)

        # Format duration
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        duration_str = f"{minutes:02d}:{seconds:02d}"

        return f"{name}\n{width}x{height} @ {fps:.1f}fps\nDuration: {duration_str}"

    def _format_clip_tooltip(self, clip: dict) -> str:
        """
        Format clip tooltip with full metadata.

        Args:
            clip: Clip metadata dict

        Returns:
            Formatted tooltip text
        """
        return (
            f"Name: {Path(clip.get('file_path', 'Unknown')).name}\n"
            f"Path: {clip.get('file_path', 'Unknown')}\n"
            f"Duration: {clip.get('duration', 0):.2f}s\n"
            f"Resolution: {clip.get('width', 0)}x{clip.get('height', 0)}\n"
            f"FPS: {clip.get('fps', 0):.2f}\n"
            f"Codec: {clip.get('codec', 'Unknown')}\n"
            f"Size: {clip.get('size_bytes', 0) / (1024 * 1024):.1f} MB"
        )

    def _passes_filter(self, clip: dict) -> bool:
        """
        Check if clip passes current filter.

        Args:
            clip: Clip metadata dict

        Returns:
            True if clip passes filter
        """
        if not self.current_filter:
            return True

        # Search in name, codec, resolution
        searchable = (
            f"{Path(clip.get('file_path', '')).name} "
            f"{clip.get('codec', '')} "
            f"{clip.get('width', '')}x{clip.get('height', '')}"
        ).lower()

        return self.current_filter in searchable

    def _sort_clips(self, clips: list[dict]) -> list[dict]:
        """
        Sort clips by current sort mode.

        Args:
            clips: List of clip dicts

        Returns:
            Sorted list of clips
        """
        if self.current_sort == "name":
            return sorted(clips, key=lambda c: Path(c.get("file_path", "")).name.lower())
        elif self.current_sort == "duration":
            return sorted(clips, key=lambda c: c.get("duration", 0), reverse=True)
        elif self.current_sort == "date":
            return sorted(clips, key=lambda c: c.get("id", 0), reverse=True)
        return clips

    def _get_clip_by_id(self, clip_id: int | str) -> dict | None:
        """
        Get clip metadata by ID.

        Args:
            clip_id: Clip ID

        Returns:
            Clip metadata dict or None
        """
        for clip in self.clips:
            if clip.get("id") == clip_id:
                return clip
        return None

    def _update_status(self) -> None:
        """Update status label."""
        total = len(self.clips)
        filtered = len(self.filtered_clips)

        if total == 0:
            self.status_label.setText("No clips loaded")
        elif filtered == total:
            self.status_label.setText(f"{total} clip{'s' if total != 1 else ''}")
        else:
            self.status_label.setText(f"{filtered} of {total} clips")

    def _on_search_changed(self, text: str) -> None:
        """Handle search field text change."""
        self.filter_clips(text)

    def _on_sort_changed(self, index: int) -> None:
        """Handle sort combo box change."""
        sort_modes = ["name", "duration", "date"]
        self.current_sort = sort_modes[index]
        self._refresh_clip_list()

    def _clear_filter(self) -> None:
        """Clear search filter."""
        self.search_field.clear()
        self.filter_clips("")

    def _on_clip_clicked(self, item: QListWidgetItem) -> None:
        """Handle clip item click."""
        clip = self._get_clip_by_id(item.data(Qt.ItemDataRole.UserRole))
        if clip:
            self.clip_selected.emit(clip)

    def _on_clip_double_clicked(self, item: QListWidgetItem) -> None:
        """Handle clip item double-click (could trigger preview)."""
        clip = self._get_clip_by_id(item.data(Qt.ItemDataRole.UserRole))
        if clip:
            # Double-click behavior can be customized
            # For now, just emit selection signal
            self.clip_selected.emit(clip)

    def _start_drag(self, supported_actions) -> None:
        """
        Custom drag handler for clip items.

        Args:
            supported_actions: Qt supported drag actions
        """
        current_item = self.clip_list.currentItem()
        if not current_item:
            return

        clip = self._get_clip_by_id(current_item.data(Qt.ItemDataRole.UserRole))
        if not clip:
            return

        # Emit drag signal
        self.clip_dragged.emit(clip)

        # Create drag object
        drag = QDrag(self.clip_list)

        # Set mime data with clip information
        mime_data = QMimeData()
        clip_json = json.dumps(clip)
        mime_data.setData("application/x-pb-studio-clip", QByteArray(clip_json.encode("utf-8")))
        mime_data.setText(str(clip.get("id", "")))
        drag.setMimeData(mime_data)

        # Set drag pixmap (thumbnail)
        thumbnail = self._get_or_create_thumbnail(clip)
        if thumbnail:
            drag.setPixmap(thumbnail)
            drag.setHotSpot(thumbnail.rect().center())

        # Execute drag
        drag.exec(Qt.DropAction.CopyAction)
