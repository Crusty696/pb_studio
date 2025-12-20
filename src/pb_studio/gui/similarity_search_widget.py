"""
Similarity Search Widget - UI fuer Aehnlichkeitssuche.

Ermoeglicht:
- Suche nach aehnlichen Clips zu einem ausgewaehlten Clip
- Duplikat-Erkennung
- Visuelle Darstellung der Aehnlichkeit
"""

from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..analysis.similarity import SimilarityResult, SimilaritySearch
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SimilarityWorker(QThread):
    """Worker Thread fuer Aehnlichkeitssuche."""

    progress = pyqtSignal(int, int)  # current, total
    result_ready = pyqtSignal(list)  # List[SimilarityResult]
    error = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, similarity_search: SimilaritySearch, clip_id: int, k: int = 10):
        super().__init__()
        self.similarity_search = similarity_search
        self.clip_id = clip_id
        self.k = k

    def run(self):
        """Fuehrt die Suche aus."""
        try:
            results = self.similarity_search.find_similar(
                self.clip_id, k=self.k, min_similarity=0.3
            )
            self.result_ready.emit(results)
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            self.error.emit(str(e))
        finally:
            self.finished_signal.emit()


class SimilarClipWidget(QFrame):
    """Widget fuer einen aehnlichen Clip."""

    clicked = pyqtSignal(int)  # clip_id

    def __init__(self, result: SimilarityResult, parent=None):
        super().__init__(parent)
        self.result = result
        self.clip_id = result.clip_id

        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        self.setStyleSheet(
            """
            SimilarClipWidget {
                background-color: #3a3a3a;
                border: 1px solid #555;
                border-radius: 5px;
                padding: 8px;
            }
            SimilarClipWidget:hover {
                background-color: #4a4a4a;
                border-color: #2a82da;
            }
        """
        )
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._init_ui()

    def _init_ui(self):
        """Initialisiert die UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Thumbnail Placeholder
        thumb_label = QLabel()
        thumb_label.setFixedSize(80, 45)
        thumb_label.setStyleSheet(
            """
            background-color: #2b2b2b;
            border: 1px solid #555;
            color: #888;
            font-size: 10px;
        """
        )
        thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb_label.setText("📹")
        layout.addWidget(thumb_label)

        # Info
        info_layout = QVBoxLayout()

        # Filename
        filename = self.result.details.get("filename", f"Clip {self.clip_id}")
        name_label = QLabel(filename[:30] + "..." if len(filename) > 30 else filename)
        name_label.setStyleSheet("font-weight: bold; color: #fff;")
        info_layout.addWidget(name_label)

        # Match Type & Similarity
        match_color = self._get_match_color(self.result.match_type)
        similarity_pct = self.result.similarity_score * 100
        match_label = QLabel(
            f"{self.result.match_type.replace('_', ' ').title()} - {similarity_pct:.1f}%"
        )
        match_label.setStyleSheet(f"color: {match_color}; font-size: 11px;")
        info_layout.addWidget(match_label)

        # Duration
        duration = self.result.details.get("duration", 0)
        if duration:
            duration_label = QLabel(f"{duration:.1f}s")
            duration_label.setStyleSheet("color: #888; font-size: 10px;")
            info_layout.addWidget(duration_label)

        layout.addLayout(info_layout)
        layout.addStretch()

        # Similarity Bar
        similarity_bar = QProgressBar()
        similarity_bar.setRange(0, 100)
        similarity_bar.setValue(int(similarity_pct))
        similarity_bar.setFixedWidth(60)
        similarity_bar.setFixedHeight(16)
        similarity_bar.setTextVisible(False)
        similarity_bar.setStyleSheet(
            f"""
            QProgressBar {{
                background-color: #2b2b2b;
                border: 1px solid #555;
                border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background-color: {match_color};
                border-radius: 2px;
            }}
        """
        )
        layout.addWidget(similarity_bar)

    def _get_match_color(self, match_type: str) -> str:
        """Gibt Farbe fuer Match-Type zurueck."""
        colors = {
            "exact": "#27ae60",
            "near_duplicate": "#2ecc71",
            "similar": "#f39c12",
            "low_similarity": "#e74c3c",
        }
        return colors.get(match_type, "#95a5a6")

    def mousePressEvent(self, event):
        """Handle click."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.clip_id)
        super().mousePressEvent(event)


class SimilaritySearchWidget(QWidget):
    """
    Widget fuer Aehnlichkeitssuche.

    Signals:
        clip_selected: Emittiert wenn ein aehnlicher Clip angeklickt wird
    """

    clip_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._similarity_search: SimilaritySearch | None = None
        self._current_clip_id: int | None = None
        self._results: list[SimilarityResult] = []
        self._worker: SimilarityWorker | None = None

        self._init_ui()

    def _init_ui(self):
        """Initialisiert die UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = QHBoxLayout()

        title = QLabel("🔍 Similar Clips")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #fff;")
        header.addWidget(title)

        header.addStretch()

        # Result count slider
        header.addWidget(QLabel("Results:"))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(5, 50)
        self.count_spin.setValue(10)
        self.count_spin.setFixedWidth(60)
        header.addWidget(self.count_spin)

        layout.addLayout(header)

        # Current clip info
        self.current_clip_label = QLabel("Select a clip to find similar ones")
        self.current_clip_label.setStyleSheet("color: #888; font-style: italic; padding: 10px;")
        layout.addWidget(self.current_clip_label)

        # Search button
        self.search_btn = QPushButton("🔍 Find Similar Clips")
        self.search_btn.setEnabled(False)
        self.search_btn.clicked.connect(self._start_search)
        self.search_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #2a82da;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }
        """
        )
        layout.addWidget(self.search_btn)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # Results area
        self.results_scroll = QScrollArea()
        self.results_scroll.setWidgetResizable(True)
        self.results_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.results_layout.setSpacing(8)

        self.results_scroll.setWidget(self.results_container)
        layout.addWidget(self.results_scroll)

        # No results label
        self.no_results_label = QLabel("No similar clips found")
        self.no_results_label.setStyleSheet("color: #888; padding: 20px; text-align: center;")
        self.no_results_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.no_results_label.hide()
        layout.addWidget(self.no_results_label)

    def set_similarity_search(self, similarity_search: SimilaritySearch):
        """Setzt die SimilaritySearch-Instanz."""
        self._similarity_search = similarity_search

    def set_current_clip(self, clip_id: int, clip_name: str = ""):
        """Setzt den aktuellen Clip fuer die Suche."""
        self._current_clip_id = clip_id
        self.current_clip_label.setText(f"Reference: {clip_name or f'Clip {clip_id}'}")
        self.current_clip_label.setStyleSheet("color: #fff; padding: 10px; font-weight: bold;")
        self.search_btn.setEnabled(True)

    def _start_search(self):
        """Startet die Aehnlichkeitssuche."""
        if not self._similarity_search:
            logger.warning("SimilaritySearch not initialized")
            QMessageBox.warning(
                self,
                "Not Ready",
                "Similarity search is not initialized. Please analyze clips first.",
            )
            return

        if not self._current_clip_id:
            return

        # Clear previous results
        self._clear_results()

        # Show progress
        self.progress_bar.show()
        self.search_btn.setEnabled(False)

        # Start worker
        self._worker = SimilarityWorker(
            self._similarity_search, self._current_clip_id, self.count_spin.value()
        )
        self._worker.result_ready.connect(self._on_results)
        self._worker.error.connect(self._on_error)
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.start()

    def _clear_results(self):
        """Loescht alle Ergebnisse."""
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.no_results_label.hide()

    @pyqtSlot(list)
    def _on_results(self, results: list[SimilarityResult]):
        """Handler fuer Suchergebnisse."""
        self._results = results

        if not results:
            self.no_results_label.show()
            return

        # Create result widgets
        for result in results:
            widget = SimilarClipWidget(result)
            widget.clicked.connect(self.clip_selected.emit)
            self.results_layout.addWidget(widget)

        logger.info(f"Found {len(results)} similar clips")

    @pyqtSlot(str)
    def _on_error(self, error_msg: str):
        """Handler fuer Fehler."""
        logger.error(f"Similarity search error: {error_msg}")
        QMessageBox.warning(self, "Search Error", f"Similarity search failed:\n{error_msg}")

    @pyqtSlot()
    def _on_finished(self):
        """Handler wenn Suche abgeschlossen."""
        self.progress_bar.hide()
        self.search_btn.setEnabled(True)


class DuplicateFinderDialog(QDialog):
    """Dialog zum Finden von Duplikaten."""

    def __init__(self, similarity_search: SimilaritySearch, parent=None):
        super().__init__(parent)
        self.similarity_search = similarity_search
        self._duplicates: list[list[int]] = []

        self.setWindowTitle("Find Duplicates")
        self.setMinimumSize(500, 400)

        self._init_ui()

    def _init_ui(self):
        """Initialisiert die UI."""
        layout = QVBoxLayout(self)

        # Info
        info_label = QLabel(
            "This will scan all clips in your library to find potential duplicates.\n"
            "Clips with 95%+ similarity will be grouped together."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #888; padding: 10px;")
        layout.addWidget(info_label)

        # Threshold slider
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Similarity Threshold:"))

        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(80, 99)
        self.threshold_slider.setValue(95)
        self.threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        threshold_layout.addWidget(self.threshold_slider)

        self.threshold_label = QLabel("95%")
        self.threshold_slider.valueChanged.connect(lambda v: self.threshold_label.setText(f"{v}%"))
        threshold_layout.addWidget(self.threshold_label)

        layout.addLayout(threshold_layout)

        # Scan button
        self.scan_btn = QPushButton("🔍 Scan for Duplicates")
        self.scan_btn.clicked.connect(self._scan_duplicates)
        layout.addWidget(self.scan_btn)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # Results
        self.results_list = QListWidget()
        self.results_list.setStyleSheet(
            """
            QListWidget {
                background-color: #2b2b2b;
                border: 1px solid #555;
                color: #fff;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #444;
            }
            QListWidget::item:selected {
                background-color: #2a82da;
            }
        """
        )
        layout.addWidget(self.results_list)

        # Status
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888;")
        layout.addWidget(self.status_label)

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _scan_duplicates(self):
        """Scannt nach Duplikaten."""
        self.results_list.clear()
        self.scan_btn.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()

        try:
            threshold = self.threshold_slider.value() / 100.0
            self._duplicates = self.similarity_search.find_duplicates(threshold)

            if self._duplicates:
                for i, group in enumerate(self._duplicates):
                    item = QListWidgetItem(
                        f"Group {i + 1}: {len(group)} clips (IDs: {', '.join(map(str, group))})"
                    )
                    self.results_list.addItem(item)

                self.status_label.setText(f"Found {len(self._duplicates)} duplicate groups")
                self.status_label.setStyleSheet("color: #f39c12;")
            else:
                self.status_label.setText("No duplicates found!")
                self.status_label.setStyleSheet("color: #27ae60;")

        except Exception as e:
            logger.error(f"Duplicate scan failed: {e}")
            self.status_label.setText(f"Error: {e}")
            self.status_label.setStyleSheet("color: #e74c3c;")

        finally:
            self.progress_bar.hide()
            self.scan_btn.setEnabled(True)

    def get_duplicates(self) -> list[list[int]]:
        """Gibt gefundene Duplikat-Gruppen zurueck."""
        return self._duplicates
