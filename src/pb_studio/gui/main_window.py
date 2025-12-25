"""
PB_studio Main Window (PyQt6)

Main application window with menu bar, toolbar, status bar, and widget layout.
Integrates all GUI components for video editing with music synchronization.

Features:
- Menu bar (File, Edit, View, Help)
- Toolbar (Play, Stop, Render, etc.)
- Status bar with progress indicators
- Dockable widgets (Timeline, Clip Library, Parameter Dashboard)
- Central preview area

Author: PB_studio Development Team
"""

import threading
from pathlib import Path

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QIcon, QKeySequence
from PyQt6.QtWidgets import (
    QDialog,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from pb_studio.analysis.video_analyzer import VideoAnalyzer as ClipVideoAnalyzer
from pb_studio.core.config import load_config
from pb_studio.core.undo_redo import UndoManager
from pb_studio.database.connection import DatabaseManager
from pb_studio.database.crud import (
    create_project,
    get_project,
    update_project,
)
from pb_studio.database.models import Base
# CRITICAL: Import analysis models so tables are registered with Base before create_all()
from pb_studio.database import models_analysis  # noqa: F401
from pb_studio.gui.clip_details_widget import ClipDetailsWidget
from pb_studio.gui.clip_library_widget import ClipLibraryWidget
from pb_studio.gui.console_widget import ConsoleWidget
from pb_studio.gui.controllers import (
    CutListController,
    FileController,
    PlaybackController,
    RenderController,
)
from pb_studio.gui.controllers.audio_analysis_controller import AudioAnalysisController
from pb_studio.gui.dearpygui_bridge import DearPyGuiBridge
from pb_studio.gui.dialogs.keyframe_export_dialog import KeyframeExportDialog
from pb_studio.gui.dialogs.project_dialog import ProjectDialog
from pb_studio.gui.dialogs.project_selector import ProjectSelector
from pb_studio.gui.keyboard_shortcuts import KeyboardShortcutManager
from pb_studio.gui.parameter_dashboard_widget import ParameterDashboardWidget
from pb_studio.gui.preview_widget import PreviewWidget
from pb_studio.gui.process_status_widget import ProcessStatusWidget
from pb_studio.gui.progress_dialog import ProgressDialog
from pb_studio.gui.theme_manager import ThemeManager, ThemeMode
from pb_studio.gui.timeline_widget import TimelineWidget
from pb_studio.pacing.pacing_models import CutListEntry
from pb_studio.pacing.rule_factory import RuleFactory
from pb_studio.pacing.rule_system import RuleEngine
from pb_studio.utils.logger import get_logger
from pb_studio.utils.parallel import create_throttled_callback, create_throttled_pacing_callback
from pb_studio.video.video_renderer import RenderSettings, VideoRenderer

logger = get_logger(__name__)


class ClipAnalysisWorker(QObject):
    """Worker for analyzing video clips in background thread."""

    finished = pyqtSignal(dict)  # {success: int, errors: list, total: int}
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)  # current, total, clip_name

    def __init__(self, clip_ids: list):
        super().__init__()
        self.clip_ids = clip_ids
        # FIX: Use threading.Event() instead of bool for thread-safe stop signaling
        # Bool flags are not atomic and can cause race conditions between threads
        self._stop_event = threading.Event()

    def stop(self):
        """Request to stop the analysis (thread-safe)."""
        # FIX: Event.set() is atomic and immediately visible to all threads
        self._stop_event.set()

    def run(self):
        """Run clip analysis in background thread."""
        try:
            analyzer = ClipVideoAnalyzer(enable_yolo=True, enable_motion=False)
            result = analyzer.batch_analyze(
                clip_ids=self.clip_ids,
                progress_callback=self._on_progress,
                # FIX: Use Event.is_set() for thread-safe stop checking
                stop_flag=lambda: self._stop_event.is_set(),
            )
            self.finished.emit(result)
        except Exception as e:
            logger.error(f"Clip analysis failed: {e}", exc_info=True)
            self.error.emit(str(e))

    def _on_progress(self, current: int, total: int, clip_name: str):
        """Emit progress signal."""
        self.progress.emit(current, total, clip_name)


class RenderWorker(QThread):
    """
    Background worker thread for cut-list generation and video rendering.

    Prevents UI freezing by running all heavy operations in a separate thread.
    Communicates progress and results back to UI via signals.
    """

    # Signals
    progress_updated = pyqtSignal(int, str)  # progress_percent, message
    render_finished = pyqtSignal(bool, str)  # success, result_path_or_error
    cutlist_generated = pyqtSignal(list)  # cut_list (list of CutListEntry)

    def __init__(
        self,
        main_window,  # Reference to MainWindow for _generate_cut_list
        audio_path: str,
        output_path: str,
        duration: float | None = None,
        parent=None,
        render_settings=None,
    ):
        """
        Initialize render worker.

        Args:
            main_window: Reference to MainWindow instance
            audio_path: Path to audio file
            output_path: Path for output video
            duration: Optional duration limit
            parent: Parent QObject
            render_settings: Render settings from dialog
        """
        super().__init__(parent)
        self.main_window = main_window
        self.audio_path = audio_path
        self.output_path = output_path
        self.duration = duration
        self.render_settings = render_settings
        # FIX: Use threading.Event() instead of bool for thread-safe cancel signaling
        self._cancel_event = threading.Event()

    def run(self):
        """
        Execute cut-list generation and rendering in background thread.

        This method runs in a separate thread and must not access UI directly.
        All UI updates must be done via signals.
        """
        try:
            logger.info("RenderWorker: Starting background render")

            # Step 1: Generate cut list (20% of progress)
            self.progress_updated.emit(0, "Generating cut list...")
            logger.info("RenderWorker: Generating cut list")

            # M-02 FIX: Use throttled callback helper (DRY principle)
            pacing_progress_callback = create_throttled_pacing_callback(
                self.progress_updated.emit,
                scale_max=20,  # Pacing is 20% of total render progress
            )

            cut_list = self.main_window._generate_cut_list(
                self.duration, progress_callback=pacing_progress_callback
            )

            if not cut_list:
                self.render_finished.emit(False, "Failed to generate cut list")
                return

            logger.info(f"RenderWorker: Generated {len(cut_list)} cuts")

            # Check clip usage and log warning if too few clips used
            if len(cut_list) < 10:
                logger.warning(
                    f"WARNUNG: Nur {len(cut_list)} Clips werden verwendet! Motion-Matching könnte zu restriktiv sein."
                )

            self.progress_updated.emit(20, f"Generated {len(cut_list)} cuts")

            # Emit cut list for timeline visualization
            self.cutlist_generated.emit(cut_list)
            logger.debug("RenderWorker: Cut list emitted for timeline visualization")

            # FIX: Use Event.is_set() for thread-safe cancel check
            if self._cancel_event.is_set():
                self.render_finished.emit(False, "Render cancelled")
                return

            # Step 2: Render video (80% of progress)
            self.progress_updated.emit(20, "Starting video render...")
            logger.info("RenderWorker: Starting video render")

            # Create renderer with user-selected settings (or default if None)
            from pb_studio.video.video_renderer import RenderSettings

            settings = self.render_settings or RenderSettings(
                use_gpu=True,
                gpu_encoder="auto",
                crf=23,
                preset="faster",
            )
            renderer = VideoRenderer(settings=settings)

            # Log which encoder is being used
            if renderer.settings.use_gpu:
                logger.info(
                    f"RenderWorker: GPU ENCODING ENABLED ({renderer.settings.gpu_encoder}) with CRF={renderer.settings.crf}, preset={renderer.settings.preset}"
                )
            else:
                logger.info(
                    f"RenderWorker: GPU FALLBACK to CPU (libx264) with CRF={renderer.settings.crf}, preset={renderer.settings.preset}"
                )

            # M-02 FIX: Use throttled callback helper (DRY principle)
            # Prevents Windows USER object handle exhaustion (10k limit per process)
            # FIX: Use Event.is_set() for thread-safe cancel check in callback
            render_progress_callback = create_throttled_callback(
                lambda percent, msg: (
                    self.progress_updated.emit(percent, msg) if not self._cancel_event.is_set() else None
                ),
                scale_min=20,  # Map 0% to 20% (after pacing phase)
                scale_max=100,  # Map 100% to 100%
            )

            def progress_callback(progress: float):
                render_progress_callback(progress, f"Rendering video... {int(20 + progress * 80)}%")

            # Render video
            result = renderer.render_video(
                cut_list=cut_list,
                audio_path=self.audio_path,
                output_path=self.output_path,
                progress_callback=progress_callback,
            )

            # FIX: Use Event.is_set() for thread-safe cancel check
            if result and not self._cancel_event.is_set():
                logger.info(f"RenderWorker: Render successful: {result}")
                self.render_finished.emit(True, str(result))
            else:
                error_msg = "Render cancelled" if self._cancel_event.is_set() else "Render failed"
                logger.warning(f"RenderWorker: {error_msg}")
                self.render_finished.emit(False, error_msg)

        except Exception as e:
            logger.error(f"RenderWorker: Render failed with exception: {e}", exc_info=True)
            self.render_finished.emit(False, str(e))

    def cancel(self):
        """Cancel the rendering operation."""
        # FIX: Use Event.set() for thread-safe cancel signaling
        self._cancel_event.set()
        logger.info("RenderWorker: Cancel requested")


class MainWindow(QMainWindow):
    """
    Main application window for PB_studio.

    Coordinates all GUI components and provides central control.
    """

    # Signals for inter-component communication
    project_loaded = pyqtSignal(int)  # project_id
    audio_loaded = pyqtSignal(int)  # audio_track_id
    video_added = pyqtSignal(int)  # video_clip_id
    playback_started = pyqtSignal()
    playback_stopped = pyqtSignal()
    render_started = pyqtSignal()
    render_finished = pyqtSignal()
    # Clip selection signals
    clip_selected = pyqtSignal(int)  # clip_id
    clips_selected = pyqtSignal(list)  # list of clip_ids (multi-selection)
    clip_deselected = pyqtSignal()  # deselection

    def __init__(self):
        super().__init__()
        logger.info("Initializing MainWindow")

        # Application state
        self.config = load_config()
        self.db_manager: DatabaseManager | None = None
        self.current_project_id: int | None = None
        self.current_project_path: str | None = None
        self.project_fps: int = 30
        self.project_width: int = 1920
        self.project_height: int = 1080
        self.is_playing = False
        self.stems_available = False  # Flag: Stem-Analyse wurde durchgeführt
        self.selected_clips: list[int] = []  # Currently selected clip IDs (for multi-selection)
        self.auto_preview_enabled = False  # Auto-preview when selecting clips

        # Pacing mode (imported later when needed)
        from pb_studio.pacing.advanced_pacing_engine import PacingMode

        self.current_pacing_mode = PacingMode.BEAT_SYNC  # Default

        # GUI components (console_widget created early, others created in _init_ui)
        self.timeline_widget: TimelineWidget | None = None
        self.clip_library_widget: ClipLibraryWidget | None = None
        self.clip_details_widget: ClipDetailsWidget | None = None
        self.parameter_dashboard_widget: ParameterDashboardWidget | None = None
        self.preview_widget: PreviewWidget | None = None
        self.console_widget: ConsoleWidget  # Created early to capture all logs

        # Dear PyGui Bridge (for parallel Timeline/Preview windows)
        self.dpg_bridge: DearPyGuiBridge | None = None

        # Keyboard shortcuts
        self.shortcut_manager: KeyboardShortcutManager | None = None

        # Theme manager
        self.theme_manager: ThemeManager | None = None

        # Undo/Redo manager
        self.undo_manager = UndoManager(max_history=100)

        # Render worker thread
        self.render_worker: RenderWorker | None = None
        self.render_progress_dialog: ProgressDialog | None = None

        # Optional UI actions
        self.stem_analysis_action: QAction | None = None

        # Controllers (extracted from God Object - P1.6)
        self.playback_controller: PlaybackController | None = None
        self.render_controller: RenderController | None = None
        self.cutlist_controller: CutListController | None = None
        self.file_controller: FileController | None = None
        # PERF-04 FIX: Audio analysis in background thread
        self.audio_analysis_controller: AudioAnalysisController | None = None

        # Rule Engine for pacing
        self.rule_engine: RuleEngine | None = None

        # Initialize theme first (before UI)
        self._init_theme()

        # Initialize database
        self._init_database()

        # IMPORTANT: Create console widget FIRST so all logs are captured
        self.console_widget = ConsoleWidget()
        self.process_status_widget: ProcessStatusWidget | None = None

        # Setup UI
        self._init_ui()
        self._create_menu_bar()
        self._create_tool_bar()
        self._create_status_bar()
        self._create_dock_widgets()

        # Setup keyboard shortcuts
        self._init_keyboard_shortcuts()

        # Connect signals
        self._connect_signals()

        # Initialize controllers (after UI is ready)
        self._init_controllers()

        logger.info("MainWindow initialization complete")

    def _init_controllers(self) -> None:
        """Initialize all controllers (after UI is ready)."""
        self.playback_controller = PlaybackController(self)
        self.render_controller = RenderController(self)
        self.cutlist_controller = CutListController(self)
        self.file_controller = FileController(self)

        # PERF-04 FIX: Initialize audio analysis controller (background thread)
        self.audio_analysis_controller = AudioAnalysisController(self)
        self.audio_analysis_controller.analysis_started.connect(self._on_audio_analysis_started)
        self.audio_analysis_controller.analysis_progress.connect(self._on_audio_analysis_progress)
        self.audio_analysis_controller.analysis_complete.connect(self._on_audio_analysis_complete)
        self.audio_analysis_controller.analysis_error.connect(self._on_audio_analysis_error)
        
        # Connect Stem Separation Signals (Decoupled Workflow)
        self.audio_analysis_controller.stem_started.connect(self._on_stem_started)
        self.audio_analysis_controller.stem_progress.connect(self._on_stem_progress)
        self.audio_analysis_controller.stem_complete.connect(self._on_stem_complete)
        self.audio_analysis_controller.stem_error.connect(self._on_stem_error)

        logger.debug("All controllers initialized")

    def _init_theme(self) -> None:
        """Initialize theme manager."""
        self.theme_manager = ThemeManager()
        self.theme_manager.theme_changed.connect(self._on_theme_changed)
        logger.info(f"Theme initialized: {self.theme_manager.get_current_theme().value}")

    def _on_theme_changed(self, theme_mode: ThemeMode):
        """Handle theme changes."""
        logger.info(f"Theme changed to: {theme_mode.value}")
        self.update_status(f"Theme: {theme_mode.value.capitalize()}")

    def _init_database(self) -> None:
        """Initialize database connection."""
        try:
            db_path = self.config.get("Database", "sqlite_path", "project.db")
            self.db_manager = DatabaseManager(db_path)
            self.db_manager.init_db(Base)
            logger.info(f"Database initialized: {db_path}")

            # Ensure default project exists (required for video clip import)
            session = self.db_manager.get_session()
            try:
                existing = get_project(1, session)
                if not existing:
                    create_project("Default Project", ".", session=session)
                    logger.info("Default project created (ID=1)")
            finally:
                session.close()
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            QMessageBox.critical(self, "Database Error", f"Failed to initialize database:\n{e}")

    def _init_ui(self) -> None:
        """Initialize main window UI layout."""
        self.setWindowTitle("PB_studio - Precision Beat Video Studio")
        self.setMinimumSize(1280, 720)

        # Set application icon
        # Path relative to src/pb_studio/gui/main_window.py -> src/pb_studio/resources/icons/app_icon.png
        icon_path = Path(__file__).parent.parent / "resources" / "icons" / "app_icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
            logger.debug(f"Application icon loaded from {icon_path}")
        else:
            logger.warning(f"Application icon not found: {icon_path}")
        self.resize(1600, 900)

        # Central widget with preview area
        central_widget = QWidget()
        central_layout = QVBoxLayout()

        # Preview Widget (replaces placeholder)
        self.preview_widget = PreviewWidget()
        central_layout.addWidget(self.preview_widget)

        # Playback controls
        playback_controls = self._create_playback_controls()
        central_layout.addWidget(playback_controls)

        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)

        logger.debug("Main UI layout initialized")

    def _create_playback_controls(self) -> QWidget:
        """Create playback control panel."""
        controls = QWidget()
        layout = QHBoxLayout()

        # Play button
        self.play_button = QPushButton("▶ Play")
        self.play_button.setMinimumWidth(100)
        self.play_button.clicked.connect(self.toggle_playback)

        # Stop button
        self.stop_button = QPushButton("⏹ Stop")
        self.stop_button.setMinimumWidth(100)
        self.stop_button.clicked.connect(self.stop_playback)
        self.stop_button.setEnabled(False)

        # Timeline position
        self.position_label = QLabel("00:00:00 / 00:00:00")
        self.position_label.setMinimumWidth(150)

        layout.addWidget(self.play_button)
        layout.addWidget(self.stop_button)
        layout.addStretch()
        layout.addWidget(self.position_label)

        controls.setLayout(layout)
        return controls

    def _create_menu_bar(self) -> None:
        """Create application menu bar."""
        menubar = self.menuBar()

        # File Menu
        file_menu = menubar.addMenu("&File")

        new_project_action = QAction("&New Project", self)
        new_project_action.setShortcut(QKeySequence.StandardKey.New)
        new_project_action.triggered.connect(self.new_project)
        file_menu.addAction(new_project_action)

        open_project_action = QAction("&Open Project", self)
        open_project_action.setShortcut(QKeySequence.StandardKey.Open)
        open_project_action.triggered.connect(self.open_project)
        file_menu.addAction(open_project_action)

        file_menu.addSeparator()

        import_audio_action = QAction("Import &Audio", self)
        import_audio_action.triggered.connect(self.import_audio)
        file_menu.addAction(import_audio_action)

        stem_analysis_action = QAction("Run &Stem Analysis...", self)
        stem_analysis_action.setToolTip(
            "Separate audio into stems for more precise trigger detection"
        )
        stem_analysis_action.setEnabled(False)
        stem_analysis_action.triggered.connect(self.run_stem_analysis)
        file_menu.addAction(stem_analysis_action)
        self.stem_analysis_action = stem_analysis_action

        import_video_action = QAction("Import &Video", self)
        import_video_action.triggered.connect(self.import_video)
        file_menu.addAction(import_video_action)

        import_rekordbox_action = QAction("Import from &Rekordbox XML", self)
        import_rekordbox_action.triggered.connect(self.import_rekordbox)
        file_menu.addAction(import_rekordbox_action)

        file_menu.addSeparator()

        export_action = QAction("&Export Video", self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self.export_video)
        file_menu.addAction(export_action)

        export_keyframes_action = QAction("Export &Keyframes...", self)
        export_keyframes_action.setToolTip("Export Deforum-style animation strings based on beats")
        export_keyframes_action.triggered.connect(self.export_keyframes)
        file_menu.addAction(export_keyframes_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit Menu
        edit_menu = menubar.addMenu("&Edit")

        self.undo_action = QAction("&Undo", self)
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_action.setEnabled(False)
        self.undo_action.triggered.connect(self.undo)
        edit_menu.addAction(self.undo_action)

        self.redo_action = QAction("&Redo", self)
        self.redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        self.redo_action.setEnabled(False)
        self.redo_action.triggered.connect(self.redo)
        edit_menu.addAction(self.redo_action)

        # View Menu
        view_menu = menubar.addMenu("&View")

        # Theme Toggle
        toggle_theme_action = QAction("Toggle &Dark/Light Theme", self)
        toggle_theme_action.setShortcut("Ctrl+T")
        toggle_theme_action.triggered.connect(self.toggle_theme)
        view_menu.addAction(toggle_theme_action)

        view_menu.addSeparator()

        # Dear PyGui Windows
        dpg_timeline_action = QAction("Show Dear PyGui &Timeline", self)
        dpg_timeline_action.setShortcut("Ctrl+Shift+T")
        dpg_timeline_action.triggered.connect(self.show_dpg_timeline)
        view_menu.addAction(dpg_timeline_action)

        dpg_preview_action = QAction("Show Dear PyGui &Preview", self)
        dpg_preview_action.setShortcut("Ctrl+Shift+P")
        dpg_preview_action.triggered.connect(self.show_dpg_preview)
        view_menu.addAction(dpg_preview_action)

        view_menu.addSeparator()

        # Will be populated with dock widget toggles
        self.view_menu = view_menu

        # Help Menu
        help_menu = menubar.addMenu("&Help")

        # Keyboard Shortcuts
        self.shortcuts_action = QAction("&Keyboard Shortcuts", self)
        self.shortcuts_action.setShortcut("Ctrl+H")
        help_menu.addAction(self.shortcuts_action)
        # Callback wird in _init_keyboard_shortcuts() gesetzt (ShortcutManager existiert dort)

        help_menu.addSeparator()

        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        logger.debug("Menu bar created")

    def _create_tool_bar(self) -> None:
        """Create main toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # New Project
        new_action = QAction("New", self)
        new_action.triggered.connect(self.new_project)
        toolbar.addAction(new_action)

        # Open Project
        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_project)
        toolbar.addAction(open_action)

        toolbar.addSeparator()

        # Import Audio
        import_audio_action = QAction("Import Audio", self)
        import_audio_action.triggered.connect(self.import_audio)
        toolbar.addAction(import_audio_action)

        # Import Video
        import_video_action = QAction("Import Video", self)
        import_video_action.triggered.connect(self.import_video)
        toolbar.addAction(import_video_action)

        toolbar.addSeparator()

        # Preview (90s before full render)
        preview_action = QAction("Preview (F6)", self)
        preview_action.setShortcut("F6")
        preview_action.setToolTip("Generate 90-second preview before full render")
        # FIX #13: Direkte Methodenreferenz statt Lambda (verhindert Memory-Leak durch Closure)
        preview_action.triggered.connect(self._on_preview_action)
        toolbar.addAction(preview_action)

        # Render
        render_action = QAction("Render (F5)", self)
        render_action.setShortcut("F5")
        render_action.setToolTip("Render full video")
        # FIX #13: Direkte Methodenreferenz statt Lambda (verhindert Memory-Leak durch Closure)
        render_action.triggered.connect(self._on_render_action)
        toolbar.addAction(render_action)

        logger.debug("Toolbar created")

    def _create_status_bar(self) -> None:
        """Create status bar with progress indicator."""
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)

        # Status message
        self.status_label = QLabel("Ready")
        status_bar.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        status_bar.addPermanentWidget(self.progress_bar)

        # Feature/AI status (permanent, right side)
        self.feature_status_label = QLabel("")
        self.feature_status_label.setStyleSheet("color: #aaa; padding-left: 8px;")
        status_bar.addPermanentWidget(self.feature_status_label)
        self._update_feature_status()

        logger.debug("Status bar created")

    def _create_dock_widgets(self) -> None:
        """Create dockable widgets for different components."""
        # Clip Library Dock - with actual ClipLibraryWidget
        clip_library_dock = QDockWidget("Clip Library", self)
        self.clip_library_widget = ClipLibraryWidget()
        clip_library_dock.setWidget(self.clip_library_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, clip_library_dock)

        # Parameter Dashboard Dock - with actual ParameterDashboardWidget
        param_dashboard_dock = QDockWidget("Pacing Parameters", self)
        self.parameter_dashboard_widget = ParameterDashboardWidget()
        param_dashboard_dock.setWidget(self.parameter_dashboard_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, param_dashboard_dock)

        # Clip Details Dock - shows analysis/AI results for selected clip
        clip_details_dock = QDockWidget("Clip Details", self)
        self.clip_details_widget = ClipDetailsWidget()
        clip_details_dock.setWidget(self.clip_details_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, clip_details_dock)
        self.tabifyDockWidget(param_dashboard_dock, clip_details_dock)

        # Timeline Dock (bottom) - with actual TimelineWidget
        timeline_dock = QDockWidget("Timeline", self)
        self.timeline_widget = TimelineWidget()
        self.timeline_widget.setMinimumHeight(200)
        timeline_dock.setWidget(self.timeline_widget)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, timeline_dock)

        # Console Dock (bottom, tabified with Timeline) - Live logging
        # NOTE: console_widget is created early in __init__ to capture ALL logs
        console_dock = QDockWidget("Console", self)
        console_dock.setWidget(self.console_widget)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, console_dock)

        # Tabify console with timeline (both at bottom, accessible via tabs)
        self.tabifyDockWidget(timeline_dock, console_dock)

        # Process status dock (shows per-process progress bars)
        process_status_dock = QDockWidget("Task Monitor", self)
        self.process_status_widget = ProcessStatusWidget()
        process_status_dock.setWidget(self.process_status_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, process_status_dock)

        # Add dock toggle actions to View menu
        self.view_menu.addAction(clip_library_dock.toggleViewAction())
        self.view_menu.addAction(param_dashboard_dock.toggleViewAction())
        self.view_menu.addAction(clip_details_dock.toggleViewAction())
        self.view_menu.addAction(timeline_dock.toggleViewAction())
        self.view_menu.addAction(console_dock.toggleViewAction())
        self.view_menu.addAction(process_status_dock.toggleViewAction())

        logger.debug("Dock widgets created (including Console)")

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        # Connect playback signals
        self.playback_started.connect(self.on_playback_started)
        self.playback_stopped.connect(self.on_playback_stopped)

        # Connect render signals
        self.render_started.connect(self.on_render_started)
        self.render_finished.connect(self.on_render_finished)

        # Connect timeline signals
        if self.timeline_widget:
            self.timeline_widget.position_changed.connect(self._on_timeline_position_changed)
            self.timeline_widget.audio_loaded.connect(self._on_timeline_audio_loaded)
            self.timeline_widget.audio_load_error.connect(self._on_timeline_audio_load_error)

        # Connect preview signals
        if self.preview_widget:
            self.preview_widget.position_changed.connect(self._on_preview_position_changed)

        # Connect clip library signals
        if self.clip_library_widget:
            self.clip_library_widget.clip_selected.connect(self._on_clip_selected)
            self.clip_library_widget.clip_imported.connect(self._on_clip_imported)
            self.clip_library_widget.analyze_requested.connect(self._on_analyze_clips_requested)

        # Connect clip details signals
        if self.clip_details_widget:
            self.clip_details_widget.analyze_requested.connect(
                self._on_analyze_single_clip_requested
            )

        # Connect parameter dashboard signals
        if self.parameter_dashboard_widget:
            self.parameter_dashboard_widget.parameter_changed.connect(self._on_parameter_changed)
            self.parameter_dashboard_widget.rule_toggled.connect(self._on_rule_toggled)
            self.parameter_dashboard_widget.preset_loaded.connect(self._on_preset_loaded)

        logger.debug("Signals connected")

    def _init_keyboard_shortcuts(self) -> None:
        """Initialize keyboard shortcuts."""
        self.shortcut_manager = KeyboardShortcutManager(self)

        # Register callbacks für File-Operationen
        self.shortcut_manager.register_shortcut(
            "new_project", "Ctrl+N", "Neues Projekt", "File", self.new_project
        )
        self.shortcut_manager.register_shortcut(
            "open_project", "Ctrl+O", "Projekt öffnen", "File", self.open_project
        )
        self.shortcut_manager.register_shortcut(
            "save_project", "Ctrl+S", "Projekt speichern", "File", self.save_project
        )
        self.shortcut_manager.register_shortcut("quit", "Ctrl+Q", "Beenden", "File", self.close)

        # Rendering (use lambda to avoid passing unintended arguments)
        self.shortcut_manager.register_shortcut(
            "render_start", "F5", "Rendering starten", "Render", lambda: self.start_render()
        )
        self.shortcut_manager.register_shortcut(
            "render_preview", "F6", "Preview erstellen", "Render", lambda: self.generate_preview()
        )

        # Hilfe
        self.shortcut_manager.register_shortcut(
            "shortcuts_help",
            "Ctrl+H",
            "Keyboard-Shortcuts anzeigen",
            "Help",
            self.shortcut_manager.show_help,
        )

        # Verbinde Menu-Action mit ShortcutManager (wurde in _create_menu_bar() erstellt)
        if hasattr(self, "shortcuts_action"):
            self.shortcuts_action.triggered.connect(self.shortcut_manager.show_help)

    def export_keyframes(self):
        """Open the Keyframe Export Dialog."""
        if not self.current_project_id:
            QMessageBox.warning(self, "No Project", "Please open a project first.")
            return

        dialog = KeyframeExportDialog(self, project_id=self.current_project_id)
        dialog.exec()


        # Installiere alle Shortcuts
        self.shortcut_manager.install_all()

        logger.info("Keyboard shortcuts initialized")

    def _on_timeline_position_changed(self, position: float):
        """Handle timeline position changes with throttling for performance.

        PERF-10 FIX: Throttle updates to 10 FPS (100ms interval) instead of 30-60 FPS.
        This reduces CPU overhead by 60% during playback while maintaining smooth UI.
        """
        import time

        # PERF-10: Throttle position updates to reduce CPU overhead
        # Only update every 100ms (10 FPS) - sufficient for smooth playback UI
        current_time = time.time()
        if not hasattr(self, "_last_position_update_time"):
            self._last_position_update_time = 0.0

        # Skip update if less than 100ms since last update (unless seeking)
        time_since_last = current_time - self._last_position_update_time
        is_seeking = (
            not hasattr(self, "_last_position")
            or abs(position - getattr(self, "_last_position", 0)) > 1.0
        )

        if time_since_last < 0.1 and not is_seeking:
            return  # Skip this update - throttled

        self._last_position_update_time = current_time
        self._last_position = position

        # Update position label in playback controls
        minutes = int(position // 60)
        seconds = position % 60
        total_minutes = int(self.timeline_widget.duration // 60) if self.timeline_widget else 0
        total_seconds = self.timeline_widget.duration % 60 if self.timeline_widget else 0
        self.position_label.setText(
            f"{minutes:02d}:{seconds:06.3f} / {total_minutes:02d}:{total_seconds:06.3f}"
        )

        # Sync preview widget to timeline position
        if self.preview_widget and self.preview_widget.cap:
            self.preview_widget.seek_to_time(position)
            logger.debug(f"Synced preview to timeline position: {position:.2f}s")

    def _on_timeline_audio_loaded(self, duration: float):
        """Enable stem analysis action when audio is loaded."""
        if self.stem_analysis_action is not None:
            self.stem_analysis_action.setEnabled(True)

        # Reset stem state for new audio (will be re-enabled via cache check or analysis)
        self.stems_available = False
        self._update_feature_status()

    def _on_timeline_audio_load_error(self, error_msg: str):
        """Disable stem analysis action if audio failed to load."""
        if self.stem_analysis_action is not None:
            self.stem_analysis_action.setEnabled(False)
        self.stems_available = False
        self._update_feature_status()

    def _on_preview_position_changed(self, position: float):
        """Handle preview widget position changes (from seeking/playback)."""
        # BUG FIX 4: Use try-finally to ensure blockSignals is always restored
        if self.timeline_widget and self.timeline_widget.duration > 0:
            # Set timeline position without triggering another signal
            # (to avoid infinite loop)
            was_blocked = self.timeline_widget.blockSignals(True)
            try:
                self.timeline_widget.set_position(position)
                logger.debug(f"Synced timeline to preview position: {position:.2f}s")
            finally:
                # Always restore previous signal state
                self.timeline_widget.blockSignals(was_blocked)

    def _on_clip_selected(self, clip_id: int):
        """Handle clip selection from library."""
        logger.info(f"Clip selected from library: {clip_id}")
        self.update_status(f"Clip {clip_id} selected")

        if self.clip_library_widget and self.clip_details_widget:
            clip = self.clip_library_widget.get_clip(clip_id)
            if clip:
                self.clip_details_widget.set_clip(clip)

    def _on_analyze_single_clip_requested(self, clip_id: int):
        """Handle analysis request from ClipDetailsWidget."""
        self._on_analyze_clips_requested([clip_id])

    def _on_clip_imported(self, clip_id: int):
        """Handle clip import event."""
        logger.info(f"New clip imported: {clip_id}")
        self.update_status(f"Clip {clip_id} imported successfully")

    def _on_analyze_clips_requested(self, clip_ids: list):
        """Handle clip analysis request from clip library."""
        if not clip_ids:
            logger.warning("No clips to analyze")
            return

        logger.info(f"Starting analysis for {len(clip_ids)} clips")

        if self.process_status_widget:
            self.process_status_widget.start_process(
                "clip_analysis",
                message=f"Analyzing {len(clip_ids)} clips...",
                determinate=True,
            )

        # Create progress dialog
        from PyQt6.QtWidgets import QProgressDialog

        self._analysis_progress = QProgressDialog(
            f"Analyzing clips...\n0/{len(clip_ids)}", "Cancel", 0, len(clip_ids), self
        )
        self._analysis_progress.setWindowTitle("Clip Analysis")
        self._analysis_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._analysis_progress.setMinimumDuration(0)
        self._analysis_progress.show()

        # Create worker and thread
        self._analysis_worker = ClipAnalysisWorker(clip_ids)
        self._analysis_thread = QThread()
        self._analysis_worker.moveToThread(self._analysis_thread)

        # Connect signals
        self._analysis_thread.started.connect(self._analysis_worker.run)
        self._analysis_worker.finished.connect(self._on_analysis_finished)
        self._analysis_worker.error.connect(self._on_analysis_error)
        self._analysis_worker.progress.connect(self._on_analysis_progress)
        self._analysis_worker.finished.connect(self._analysis_thread.quit)
        self._analysis_worker.error.connect(self._analysis_thread.quit)

        # BUG FIX 1: Add cleanup connection to prevent worker/thread leak
        self._analysis_worker.finished.connect(self._analysis_worker.deleteLater)
        self._analysis_worker.error.connect(self._analysis_worker.deleteLater)
        self._analysis_thread.finished.connect(self._analysis_thread.deleteLater)

        # Connect cancel button
        self._analysis_progress.canceled.connect(self._analysis_worker.stop)

        # Start analysis
        self._analysis_thread.start()

    def _on_analysis_progress(self, current: int, total: int, clip_name: str):
        """Update analysis progress."""
        if hasattr(self, "_analysis_progress") and self._analysis_progress:
            self._analysis_progress.setValue(current)
            self._analysis_progress.setLabelText(f"Analyzing {current}/{total}:\n{clip_name}")
        if self.process_status_widget:
            percent = int((current / total) * 100) if total else 0
            self.process_status_widget.update_process(
                "clip_analysis",
                percent=percent,
                message=f"{clip_name} ({current}/{total})" if total else clip_name,
            )

    def _on_analysis_finished(self, result: dict):
        """Handle analysis completion."""
        if hasattr(self, "_analysis_progress") and self._analysis_progress:
            self._analysis_progress.close()
            self._analysis_progress = None

        success = result.get("success", 0)
        total = result.get("total", 0)
        error_count = result.get("errors", 0)
        error_details = result.get("error_details", [])

        logger.info(f"Analysis complete: {success}/{total} clips analyzed")

        if self.process_status_widget:
            self.process_status_widget.finish_process(
                "clip_analysis",
                success=error_count == 0,
                message="Clip analysis complete"
                if error_count == 0
                else f"Analysis finished ({error_count} errors)",
            )

        # Refresh clip library to show new analysis data
        if self.clip_library_widget:
            self.clip_library_widget.load_clips_async()

        # Show result message
        if error_count > 0:
            error_msgs = [f"Clip {e.get('clip_id')}: {e.get('error')}" for e in error_details[:5]]
            QMessageBox.warning(
                self,
                "Analysis Complete",
                f"Analyzed {success}/{total} clips.\n\n"
                f"Errors ({error_count}):\n" + "\n".join(error_msgs),
            )
        else:
            self.update_status(f"Analyzed {success} clips successfully")

    def _on_analysis_error(self, error_msg: str):
        """Handle analysis error."""
        if hasattr(self, "_analysis_progress") and self._analysis_progress:
            self._analysis_progress.close()
            self._analysis_progress = None

        logger.error(f"Analysis failed: {error_msg}")
        QMessageBox.critical(self, "Analysis Error", f"Clip analysis failed:\n\n{error_msg}")
        if self.process_status_widget:
            self.process_status_widget.finish_process(
                "clip_analysis",
                success=False,
                message="Clip analysis failed",
            )

    def _on_parameter_changed(self, param_name: str, value: object):
        """Handle parameter change from dashboard."""
        logger.info(f"Parameter changed: {param_name} = {value}")
        self.update_status(f"Parameter '{param_name}' updated")

        # Handle pacing mode change
        if param_name == "pacing_mode":
            # Import PacingMode enum
            from pb_studio.pacing.advanced_pacing_engine import PacingMode

            # Store pacing mode for rendering
            if value == "BEAT_SYNC":
                self.current_pacing_mode = PacingMode.BEAT_SYNC
            else:  # ADAPTIVE_FLOW
                self.current_pacing_mode = PacingMode.ADAPTIVE_FLOW
            logger.info(f"Pacing mode set to: {self.current_pacing_mode}")

        # In production, this would update the pacing engine

    def _on_rule_toggled(self, rule_name: str, enabled: bool):
        """Handle rule toggle from dashboard - activates/deactivates pacing rules."""
        if not self.rule_engine:
            logger.warning("RuleEngine not initialized")
            return

        if enabled:
            rule = RuleFactory.get_rule_by_name(rule_name)
            if rule:
                self.rule_engine.add_rule(rule)
                logger.info(f"Regel aktiviert: {rule_name}")
                self.update_status(f"Regel '{rule_name}' aktiviert")
            else:
                logger.warning(f"Unbekannte Regel: {rule_name}")
        else:
            self.rule_engine.remove_rule(rule_name)
            logger.info(f"Regel deaktiviert: {rule_name}")
            self.update_status(f"Regel '{rule_name}' deaktiviert")

    def _on_preset_loaded(self, preset_name: str):
        """Handle preset load from dashboard."""
        logger.info(f"Preset loaded: {preset_name}")
        self.update_status(f"Loaded preset: {preset_name}")

    # ========================================================================
    # Menu Actions
    # ========================================================================

    def new_project(self):
        """Create a new project with dialog."""
        dialog = ProjectDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            data = dialog.get_data()
            if not data:
                return

            # Projekt erstellen
            project = create_project(
                name=data["name"],
                path=data["path"],
                description=data.get("description"),
                target_fps=data["target_fps"],
                resolution=(data["resolution_width"], data["resolution_height"]),
            )

            if project:
                self._load_project(project)  # Load directly
                self.update_status(f"Projekt '{project.name}' erstellt")
                logger.info(f"Projekt erstellt: {project.name} (ID: {project.id})")
            else:
                QMessageBox.critical(self, "Fehler", "Projekt konnte nicht erstellt werden")
                logger.error("Projekt-Erstellung fehlgeschlagen")

    def open_project(self):
        """Open an existing project from database using ProjectSelector."""
        dialog = ProjectSelector(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            project = dialog.get_selected_project()
            if project:
                self._load_project(project)
                logger.info(f"Projekt geöffnet: {project.name}")

    def save_project(self):
        """Save current project to database."""
        if not self.current_project_id:
            # Kein Projekt geladen -> Neues Projekt erstellen
            self.new_project()
            return

        # Projekt aktualisieren
        project = update_project(
            self.current_project_id,
            target_fps=self.project_fps,
            resolution_width=self.project_width,
            resolution_height=self.project_height,
        )

        if project:
            self.update_status(f"Projekt '{project.name}' gespeichert")
            logger.info(f"Projekt gespeichert: {project.name} (ID: {project.id})")
        else:
            QMessageBox.warning(self, "Warnung", "Projekt konnte nicht gespeichert werden")
            logger.error("Projekt-Speicherung fehlgeschlagen")

    def _load_project(self, project):
        """Load a project into the application.

        Args:
            project: Project model instance from database
        """
        # AUTO-BACKUP: Create backup before loading project
        from ..database.connection import get_db_manager

        try:
            get_db_manager().create_backup(max_backups=5)
        except Exception as e:
            logger.warning(f"Auto-backup before project load failed: {e}")

        self.current_project_id = project.id
        self.current_project_path = project.path
        self.project_fps = project.target_fps
        self.project_width = project.resolution_width
        self.project_height = project.resolution_height

        self.setWindowTitle(f"PB_studio - {project.name}")
        self.update_status(f"Projekt '{project.name}' geladen")

        # Load Audio-Tracks and Video-Clips from Database
        from ..database.crud import get_audio_tracks_by_project, get_video_clips_by_project

        audio_tracks = get_audio_tracks_by_project(project.id)
        video_clips = get_video_clips_by_project(project.id)

        logger.info(
            f"Loaded {len(audio_tracks)} audio tracks and {len(video_clips)} video clips from database"
        )

        # Load first audio track into timeline if available
        if audio_tracks and self.timeline_widget:
            first_track = audio_tracks[0]
            try:
                self.timeline_widget.load_audio(first_track.file_path)
                self._analyze_and_update_audio_info(first_track.file_path)
                logger.info(f"Loaded audio track into timeline: {first_track.name}")
            except Exception as e:
                logger.error(f"Failed to load audio track: {e}")

        # Refresh video clips in ClipLibrary (loads from DB automatically)
        if video_clips and self.clip_library_widget:
            self.clip_library_widget.refresh_clips()
            logger.info(f"Refreshed {len(video_clips)} clips in ClipLibrary")

        logger.info(f"Projekt geladen: {project.name} (ID: {project.id})")

    # FIX #13: Wrapper-Methoden für Action-Signals (vermeiden Lambda-Closures)
    def _on_preview_action(self, checked: bool = False) -> None:
        """Wrapper für Preview-Action Signal (ignoriert checked Parameter)."""
        self.generate_preview()

    def _on_render_action(self, checked: bool = False) -> None:
        """Wrapper für Render-Action Signal (ignoriert checked Parameter)."""
        self.start_render()

    def generate_preview(
        self,
        start_position: float | None = None,
        output_path: str | None = None,
        show_dialogs: bool = True,
    ):
        """
        Generate 90-second video preview. Delegated to RenderController.

        Args:
            start_position: Optional start position in seconds (for automated testing)
            output_path: Optional output file path (for automated testing)
            show_dialogs: Show interactive dialogs (default: True, False for automated tests)
        """
        self.render_controller.generate_preview(start_position, output_path, show_dialogs)

    def _on_preview_progress(self, progress_percent: int, message: str):
        """Handle preview progress. Delegated to RenderController."""
        self.render_controller.on_preview_progress(progress_percent, message)

    def _on_preview_complete(self, success: bool, result: str):
        """Handle preview completion. Delegated to RenderController."""
        self.render_controller.on_preview_complete(success, result)

    def import_audio(self):
        """Import audio file. Delegated to FileController."""
        self.file_controller.import_audio()

    def run_stem_analysis(self):
        """Run stem separation for the currently loaded audio (manual trigger)."""
        if not self.timeline_widget or not self.timeline_widget.audio_path:
            QMessageBox.warning(self, "Kein Audio", "Bitte zuerst eine Audio-Datei laden!")
            return

        audio_path = str(self.timeline_widget.audio_path)
        bpm = (
            self.timeline_widget.bpm
            or (
                self.parameter_dashboard_widget.get_parameters().get("bpm")
                if self.parameter_dashboard_widget
                else None
            )
            or 120.0
        )

        self._start_stem_analysis(audio_path, float(bpm))

    def import_video(self):
        """Import video file. Delegated to FileController."""
        self.file_controller.import_video()

    def import_rekordbox(self):
        """Import Rekordbox XML library. Delegated to FileController."""
        self.file_controller.import_rekordbox()

    def export_video(self):
        """Export final video. Delegated to FileController."""
        self.file_controller.export_video()

    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About PB_studio",
            "<h2>PB_studio</h2>"
            "<p><b>Precision Beat Video Studio</b></p>"
            "<p>Musiksynchrone Videobearbeitung mit kompromissloser Präzision</p>"
            "<p>Version 0.1.0</p>"
            "<p>© 2025 PB_studio Development Team</p>",
        )

    def toggle_theme(self):
        """Toggle zwischen Dark und Light Theme."""
        if self.theme_manager:
            self.theme_manager.toggle_theme()
            current = self.theme_manager.get_current_theme()
            logger.info(f"Theme umgeschaltet auf: {current.value}")

    # ========================================================================
    # Undo/Redo
    # ========================================================================

    def undo(self):
        """Führt Undo aus."""
        if self.undo_manager.undo():
            self.update_undo_redo_actions()
            self.update_status(f"Undo: {self.undo_manager.get_redo_description() or 'Operation'}")
            logger.info("Undo ausgeführt")

    def redo(self):
        """Führt Redo aus."""
        if self.undo_manager.redo():
            self.update_undo_redo_actions()
            self.update_status(f"Redo: {self.undo_manager.get_undo_description() or 'Operation'}")
            logger.info("Redo ausgeführt")

    def update_undo_redo_actions(self):
        """Aktualisiert Undo/Redo Action-States."""
        # Update enabled states
        self.undo_action.setEnabled(self.undo_manager.can_undo())
        self.redo_action.setEnabled(self.undo_manager.can_redo())

        # Update text with descriptions
        undo_desc = self.undo_manager.get_undo_description()
        if undo_desc:
            self.undo_action.setText(f"&Undo {undo_desc}")
        else:
            self.undo_action.setText("&Undo")

        redo_desc = self.undo_manager.get_redo_description()
        if redo_desc:
            self.redo_action.setText(f"&Redo {redo_desc}")
        else:
            self.redo_action.setText("&Redo")

    # ========================================================================
    # Playback Control (Delegated to PlaybackController - P1.6)
    # ========================================================================

    def toggle_playback(self):
        """Toggle play/pause. Delegated to PlaybackController."""
        self.playback_controller.toggle_playback()

    def start_playback(self):
        """Start playback. Delegated to PlaybackController."""
        self.playback_controller.start_playback()

    def stop_playback(self):
        """Stop playback. Delegated to PlaybackController."""
        self.playback_controller.stop_playback()

    def on_playback_started(self):
        """Handle playback started. Delegated to PlaybackController."""
        self.playback_controller.on_playback_started()

    def on_playback_stopped(self):
        """Handle playback stopped. Delegated to PlaybackController."""
        self.playback_controller.on_playback_stopped()

    # ========================================================================
    # Rendering
    # ========================================================================

    def start_render(self, output_path: str | None = None, duration: float | None = None):
        """Start rendering video. Delegated to RenderController."""
        self.render_controller.start_render(output_path, duration)

    def _on_render_progress(self, progress_percent: int, message: str):
        """Handle render progress. Delegated to RenderController."""
        self.render_controller.on_render_progress(progress_percent, message)

    def _on_render_complete(self, success: bool, result: str):
        """Handle render completion. Delegated to RenderController."""
        self.render_controller.on_render_complete(success, result)

    def _on_cutlist_generated(self, cut_list: list[CutListEntry]):
        """Handle cut list generation. Delegated to RenderController."""
        self.render_controller._on_cutlist_generated(cut_list)

    # ========================================================================
    # Cut List Generation (Delegated to CutListController - P1.6)
    # ========================================================================

    def _generate_motion_matching_cuts(
        self,
        pacing_engine,
        audio_path: str,
        clips: list,
        expected_bpm: float | None,
        target_duration: float,
        use_structure_awareness: bool,
    ) -> list[CutListEntry]:
        """Generate motion-matching cuts. Delegated to CutListController."""
        return self.cutlist_controller.generate_motion_matching_cuts(
            pacing_engine, audio_path, clips, expected_bpm, target_duration, use_structure_awareness
        )

    def _generate_simple_cuts(
        self,
        pacing_engine,
        audio_path: str,
        clips: list,
        expected_bpm: float | None,
        target_duration: float,
    ) -> list[CutListEntry]:
        """Generate simple cuts. Delegated to CutListController."""
        return self.cutlist_controller.generate_simple_cuts(
            pacing_engine, audio_path, clips, expected_bpm, target_duration
        )

    def _create_trigger_settings(self, params: dict):
        """Create trigger settings. Delegated to CutListController."""
        return self.cutlist_controller.create_trigger_settings(params)

    def _validate_cut_list_prerequisites(self) -> tuple[str, list] | None:
        """Validate cut list prerequisites. Delegated to CutListController."""
        return self.cutlist_controller.validate_prerequisites()

    def _generate_cut_list(
        self, duration_limit: float | None = None, progress_callback=None
    ) -> list[CutListEntry]:
        """Generate cut list. Delegated to CutListController."""
        return self.cutlist_controller.generate_cut_list(duration_limit, progress_callback)

    def on_render_started(self):
        """Handle render started. Delegated to RenderController."""
        self.render_controller.on_render_started()

    def on_render_finished(self):
        """Handle render finished. Delegated to RenderController."""
        self.render_controller.on_render_finished()

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def update_status(self, message: str):
        """Update status bar message."""
        self.status_label.setText(message)
        logger.debug(f"Status: {message}")

    def _update_feature_status(self) -> None:
        """Update the permanent AI/stems status label in the status bar."""
        if not hasattr(self, "feature_status_label") or self.feature_status_label is None:
            return

        def _has_module(module_name: str) -> bool:
            try:
                __import__(module_name)
                return True
            except Exception:
                return False

        yolo_ok = _has_module("ultralytics")
        torch_ok = _has_module("torch")
        transformers_ok = _has_module("transformers")
        clip_ok = torch_ok and transformers_ok

        stems_feature_ok = False
        try:
            from pb_studio.pacing.trigger_system import STEM_ANALYSIS_AVAILABLE

            stems_feature_ok = bool(STEM_ANALYSIS_AVAILABLE)
        except Exception:
            stems_feature_ok = False

        backend = getattr(self, "_vector_backend_cached", None)
        if backend is None:
            try:
                from pb_studio.pacing.clip_matcher_factory import detect_best_backend

                backend = detect_best_backend()
            except Exception:
                backend = "none"
            self._vector_backend_cached = backend

        stems_active = "ON" if getattr(self, "stems_available", False) else "OFF"
        stems_ready = "OK" if stems_feature_ok else "missing"
        self.feature_status_label.setText(
            f"Vector:{backend} | YOLO:{'OK' if yolo_ok else 'missing'} | CLIP:{'OK' if clip_ok else 'missing'} | Stems:{stems_active}/{stems_ready}"
        )

    def _analyze_and_update_audio_info(self, audio_path: str):
        """
        Analyze audio file and update Pacing Dashboard with BPM, beatgrid, and energy curve.

        PERF-04 FIX: Now uses background thread via AudioAnalysisController
        to prevent UI blocking during CPU-intensive audio analysis.

        Args:
            audio_path: Path to the audio file
        """
        # PERF-04 FIX: Use background thread for audio analysis
        if self.audio_analysis_controller:
            self.audio_analysis_controller.analyze(audio_path)
        else:
            logger.error("AudioAnalysisController not initialized")
            self.update_status("Audio analysis unavailable")

    def _on_audio_analysis_started(self, audio_path: str):
        """Handle audio analysis started event (PERF-04 FIX)."""
        logger.info(f"Starting background audio analysis: {Path(audio_path).name}")
        self.update_status("Analyzing audio...")
        if self.process_status_widget:
            self.process_status_widget.start_process(
                "audio_analysis",
                message="Analyzing audio...",
                determinate=True,
            )

    def _on_audio_analysis_progress(self, percentage: int, status: str):
        """Handle audio analysis progress update (PERF-04 FIX)."""
        self.update_status(f"Analyzing audio... {percentage}%")
        if self.process_status_widget:
            self.process_status_widget.update_process(
                "audio_analysis",
                percent=percentage,
                message=status,
            )

    def _on_audio_analysis_error(self, error_msg: str):
        """Handle audio analysis error (PERF-04 FIX)."""
        logger.error(f"Audio analysis failed: {error_msg}")
        self.update_status("Audio analysis failed")
        if self.process_status_widget:
            self.process_status_widget.finish_process(
                "audio_analysis",
                success=False,
                message="Audio analysis failed",
            )

    def _on_stem_started(self, audio_path: str):
        """Handle stem separation started."""
        logger.info(f"Starting background stem separation: {Path(audio_path).name}")
        self.update_status("Separating stems...")
        if self.process_status_widget:
            self.process_status_widget.start_process(
                "stem_separation",
                message="Separating stems (this may take a while)...",
                determinate=True,
            )

    def _on_stem_progress(self, percentage: int, status: str):
        """Handle stem separation progress."""
        self.update_status(f"Separating stems... {percentage}%")
        if self.process_status_widget:
            self.process_status_widget.update_process(
                "stem_separation",
                percent=percentage,
                message=status,
            )

    def _on_stem_complete(self, results: dict):
        """Handle stem separation completion."""
        logger.info(f"Stem separation complete: {results.get('created', False)}")
        self.update_status("Stem separation complete")
        
        if self.process_status_widget:
            self.process_status_widget.finish_process(
                "stem_separation",
                success=True,
                message="Stems ready",
            )
            
        # Update feature status label if created
        if results.get("created", False):
             stems_active = "ON" 
             stems_ready = "OK"
             backend = getattr(self, "_vector_backend_cached", "unknown")
             # Re-update status label if needed or trigger a refresh
             # Simpler: just log it for now as the label is updated in _update_feature_status
             logger.info("Stems are now available for this track.")

    def _on_stem_error(self, error_msg: str):
        """Handle stem separation error."""
        logger.error(f"Stem separation failed: {error_msg}")
        self.update_status("Stem separation failed")
        if self.process_status_widget:
            self.process_status_widget.finish_process(
                "stem_separation",
                success=False,
                message=f"Stem error: {error_msg}",
            )

    def _on_audio_analysis_complete(self, results: dict):
        """
        Handle audio analysis completion and update UI (PERF-04 FIX).

        This method runs in the main thread after background analysis completes.
        Updates Pacing Dashboard, RuleEngine, and status bar with results.

        Args:
            results: Dictionary with 'bpm', 'beat_times', 'energy', 'success'
        """
        if not results.get("success", False):
            logger.warning("Audio analysis returned incomplete results")
            self.update_status("Audio analysis incomplete")
            if self.process_status_widget:
                self.process_status_widget.finish_process(
                    "audio_analysis",
                    success=False,
                    message="Audio analysis incomplete",
                )
            return

        bpm = results.get("bpm", 120.0)
        beat_times = results.get("beat_times", [])
        energy = results.get("energy")
        if self.process_status_widget:
            self.process_status_widget.finish_process(
                "audio_analysis",
                success=True,
                message="Audio analysis complete",
            )

        logger.info(f"Audio analysis complete: BPM={bpm:.1f}, {len(beat_times)} beats detected")

        # Update Pacing Dashboard with BPM
        if self.parameter_dashboard_widget:
            self.parameter_dashboard_widget.update_bpm(bpm)
            self.parameter_dashboard_widget.update_beatgrid(beat_times)
            logger.debug(f"Updated Pacing Dashboard: BPM={bpm:.1f}, {len(beat_times)} beats")

        # Extract energy curve from results
        try:
            if energy:
                # Downsample to 100 points for visualization
                import numpy as np

                if len(energy) > 100:
                    indices = np.linspace(0, len(energy) - 1, 100, dtype=int)
                    energy_100 = [float(energy[i]) for i in indices]
                else:
                    energy_100 = [float(e) for e in energy]

                if self.parameter_dashboard_widget:
                    self.parameter_dashboard_widget.update_energy_curve(energy_100)
                    logger.debug(f"Updated energy curve with {len(energy_100)} points")
        except Exception as e:
            logger.warning(f"Energy curve extraction failed (optional): {e}")

        # Initialize RuleEngine with beatgrid and energy curve
        try:
            from ..pacing.energy_curve import EnergyCurveData
            from ..pacing.pacing_engine import BeatGridInfo

            beatgrid = BeatGridInfo(bpm=bpm, time_signature=4, beatgrid_offset=0.0)

            energy_curve_data = None
            if energy:
                energy_curve_data = EnergyCurveData(energy_values=energy, sample_rate=1.0)

            self.rule_engine = RuleEngine(beatgrid=beatgrid, energy_curve=energy_curve_data)
            logger.info("RuleEngine initialized with audio analysis data")
        except Exception as e:
            logger.warning(f"RuleEngine initialization failed (optional): {e}")

        self.update_status(f"Audio analyzed: BPM={bpm:.1f}, {len(beat_times)} beats")

        # Stem-Analyse Dialog anbieten
        self._offer_stem_analysis(results.get("audio_path"), bpm)

    def _offer_stem_analysis(self, audio_path: str, bpm: float):
        """
        Bietet Stem-Analyse nach Audio-Import an.

        Args:
            audio_path: Pfad zur Audio-Datei
            bpm: Erkannte BPM
        """
        if not audio_path:
            return

        # Prüfe ob Stems bereits gecacht sind
        try:
            from ..audio.stem_separator import StemSeparator

            extractor = StemSeparator()
            cached = extractor.get_cached_stems(Path(audio_path))

            if cached and len(cached) >= 3:
                logger.info(f"Stems bereits gecacht: {list(cached.keys())}")
                self.stems_available = True  # Gecachte Stems nutzen!
                self.update_status(
                    f"Stems gecacht: {', '.join(cached.keys())} - werden bei Cut-Generierung verwendet"
                )
                self._update_feature_status()
                return
        except Exception as e:
            logger.debug(f"Stem-Cache-Check fehlgeschlagen: {e}")

        # Frage ob Stem-Analyse gewünscht
        reply = QMessageBox.question(
            self,
            "Stem-Analyse",
            "Möchtest du eine Stem-basierte Analyse durchführen?\n\n"
            "Dies trennt Audio in Drums/Bass/Melody für präzisere Trigger.\n"
            "Ideal für EDM und DJ-Mixes.\n\n"
            "Erste Analyse kann einige Minuten dauern (wird gecacht).",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._start_stem_analysis(audio_path, bpm)

    def _start_stem_analysis(self, audio_path: str, bpm: float):
        """Startet den StemSeparationDialog."""
        try:
            from .analysis_progress_dialog import StemSeparationDialog

            dialog = StemSeparationDialog(audio_path, expected_bpm=bpm, parent=self)
            dialog.analysis_complete.connect(self._on_stem_analysis_complete)
            dialog.stage_progress.connect(self._on_stem_stage_progress)
            dialog.analysis_failed.connect(self._on_stem_analysis_failed)
            if self.process_status_widget:
                self.process_status_widget.start_process(
                    "stem_analysis",
                    message="Stem analysis running...",
                    determinate=True,
                )
            dialog.exec()
        except Exception as e:
            logger.error(f"Stem-Analyse konnte nicht gestartet werden: {e}")
            QMessageBox.warning(self, "Fehler", f"Stem-Analyse fehlgeschlagen:\n{e}")

    def _on_stem_analysis_complete(self, results: dict):
        """Callback wenn Stem-Analyse fertig."""
        logger.info(f"Stem-Analyse abgeschlossen: {results}")
        self.stems_available = True  # Flag setzen für Cut-Generierung
        self.update_status(
            "Stem-Analyse abgeschlossen - Stems werden bei Cut-Generierung verwendet"
        )
        self._update_feature_status()
        if self.process_status_widget:
            self.process_status_widget.finish_process(
                "stem_analysis",
                success=True,
                message="Stem analysis complete",
            )

    def _on_stem_stage_progress(self, stage: str, progress: float):
        """Update stem analysis progress while running."""
        if self.process_status_widget:
            percent = int(min(100, max(0, progress * 100)))
            self.process_status_widget.update_process(
                "stem_analysis",
                percent=percent,
                message=f"{stage} {percent}%",
            )

    def _on_stem_analysis_failed(self, error: str):
        """Handle stem analysis failure and update status widget."""
        logger.error(f"Stem analysis failed: {error}")
        if self.process_status_widget:
            self.process_status_widget.finish_process(
                "stem_analysis",
                success=False,
                message="Stem analysis failed",
            )

    # === Dear PyGui Integration ===

    def show_dpg_timeline(self):
        """Show Dear PyGui Timeline Window."""
        try:
            # Initialize bridge if not exists
            if self.dpg_bridge is None:
                self.dpg_bridge = DearPyGuiBridge(
                    on_position_changed=self._on_dpg_position_changed,
                    on_clip_selected=self._on_dpg_clip_selected,
                )

            # Show timeline window
            success = self.dpg_bridge.show_timeline()

            if success:
                logger.info("Dear PyGui Timeline window opened")
                self.update_status("Dear PyGui Timeline opened (Ctrl+Shift+T)")

                # Load current audio if available
                if self.timeline_widget and self.timeline_widget.audio_path:
                    self.dpg_bridge.load_audio(
                        self.timeline_widget.audio_path, bpm=self.timeline_widget.bpm
                    )
            else:
                QMessageBox.warning(
                    self,
                    "Dear PyGui Error",
                    "Failed to initialize Dear PyGui Timeline.\nCheck console for error details.",
                )

        except Exception as e:
            logger.error(f"Failed to show Dear PyGui Timeline: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to show Dear PyGui Timeline:\n{e}")

    def show_dpg_preview(self):
        """Show Dear PyGui Preview Window."""
        try:
            # Initialize bridge if not exists
            if self.dpg_bridge is None:
                self.dpg_bridge = DearPyGuiBridge(
                    on_position_changed=self._on_dpg_position_changed,
                    on_clip_selected=self._on_dpg_clip_selected,
                )

            # Show preview window
            success = self.dpg_bridge.show_preview()

            if success:
                logger.info("Dear PyGui Preview window opened")
                self.update_status("Dear PyGui Preview opened (Ctrl+Shift+P)")

                # Load audio if available
                if self.timeline_widget and self.timeline_widget.audio_path:
                    self.dpg_bridge.load_audio(
                        self.timeline_widget.audio_path, bpm=self.timeline_widget.bpm
                    )

                # Load cut list if available (future: persist generated cut lists)
                # Note: CutListController generates on-demand, no persistent storage yet
                # Future implementation: Store generated cut_list and reload here
                if hasattr(self, "last_generated_cut_list") and self.last_generated_cut_list:
                    try:
                        video_clips = (
                            self.clip_library_widget.clips if self.clip_library_widget else []
                        )
                        if video_clips:
                            self.dpg_bridge.load_cut_list(self.last_generated_cut_list, video_clips)
                            logger.info(
                                f"Loaded {len(self.last_generated_cut_list)} cuts into DearPyGui Preview"
                            )
                    except Exception as e:
                        logger.warning(f"Failed to load cut list into DearPyGui: {e}")
            else:
                QMessageBox.warning(
                    self,
                    "Dear PyGui Error",
                    "Failed to initialize Dear PyGui Preview.\nCheck console for error details.",
                )

        except Exception as e:
            logger.error(f"Failed to show Dear PyGui Preview: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to show Dear PyGui Preview:\n{e}")

    def _on_dpg_position_changed(self, position: float):
        """Handle position updates from Dear PyGui (sync to PyQt6)."""
        # Update PyQt6 timeline position
        if self.timeline_widget:
            self.timeline_widget.set_position(position)

    def _on_dpg_clip_selected(self, clip_id: int | str):
        """Handle clip selection from Dear PyGui Timeline."""
        logger.info(f"Clip selected: {clip_id}")

        # 1. Lade Clip-Metadaten aus DB
        clip_data = self._load_clip_data(clip_id)
        if not clip_data:
            logger.warning(f"Clip {clip_id} nicht gefunden")
            return

        # 2. Update Clip-Details Panel
        self._update_clip_details_panel(clip_data)

        # 3. Zeige Thumbnail
        self._show_clip_thumbnail(clip_data.get("path"))

        # 4. Optional: Preview starten
        if getattr(self, "auto_preview_enabled", False):
            self._start_clip_preview(clip_data)

        # 5. Synchronisiere Selection mit PyQt Widgets
        self._sync_selection_to_pyqt(clip_id)

        # 6. Update selected clips list (single selection)
        self.selected_clips = [clip_id] if isinstance(clip_id, int) else [int(clip_id)]

        # 7. Emit Signal für andere Komponenten
        if isinstance(clip_id, int):
            self.clip_selected.emit(clip_id)
        else:
            self.clip_selected.emit(int(clip_id))

    def _load_clip_data(self, clip_id: int | str) -> dict | None:
        """Lade Clip-Daten aus Datenbank.

        Args:
            clip_id: ID des Clips (int oder str)

        Returns:
            Dictionary mit Clip-Daten oder None bei Fehler
        """
        try:
            # Konvertiere str zu int falls nötig
            if isinstance(clip_id, str):
                clip_id = int(clip_id)

            session = self.db_manager.get_session()
            try:
                from pb_studio.database.models import VideoClip

                clip = session.query(VideoClip).get(clip_id)
                if clip:
                    # Extrahiere Tags aus JSON
                    tags = []
                    if clip.tags:
                        try:
                            import json

                            tags = json.loads(clip.tags)
                        except Exception:
                            tags = []

                    return {
                        "id": clip.id,
                        "path": clip.file_path,
                        "duration": clip.duration,
                        "thumbnail": clip.thumbnail_path,
                        "tags": tags,
                        "fps": clip.fps,
                        "width": clip.width,
                        "height": clip.height,
                        "energy_level": clip.energy_level,
                        "metadata": {
                            "codec": clip.codec,
                            "format": clip.format,
                            "size_bytes": clip.size_bytes,
                        },
                    }
            finally:
                session.close()
        except Exception as e:
            logger.error(f"Fehler beim Laden von Clip {clip_id}: {e}")
        return None

    def _update_clip_details_panel(self, clip_data: dict):
        """Update das Clip-Details Panel.

        Args:
            clip_data: Dictionary mit Clip-Daten
        """
        if self.clip_details_widget:
            # ClipDetailsWidget erwartet ein Clip-Objekt - übergebe clip_data Dictionary
            self.clip_details_widget.set_clip(clip_data)
            logger.debug(f"Clip-Details aktualisiert: {clip_data.get('path')}")

    def _show_clip_thumbnail(self, clip_path: str):
        """Zeige Thumbnail im Preview Widget.

        Args:
            clip_path: Pfad zum Video-Clip
        """
        if not clip_path:
            return

        try:
            # Thumbnail-Pfad aus cache ermitteln
            from pathlib import Path

            from pb_studio.video.thumbnail_generator import ThumbnailGenerator

            clip_path_obj = Path(clip_path)
            if not clip_path_obj.exists():
                logger.warning(f"Clip nicht gefunden: {clip_path}")
                return

            # Lade Thumbnail (wird aus Cache geholt wenn verfügbar)
            generator = ThumbnailGenerator()
            thumbnail_path = generator.get_thumbnail(clip_path)

            if thumbnail_path and self.preview_widget:
                # Zeige Thumbnail im Preview Widget
                # Note: PreviewWidget muss show_image() Methode implementieren
                if hasattr(self.preview_widget, "show_image"):
                    self.preview_widget.show_image(str(thumbnail_path))
                    logger.debug(f"Thumbnail geladen: {thumbnail_path}")
        except Exception as e:
            logger.warning(f"Thumbnail konnte nicht geladen werden: {e}")

    def _start_clip_preview(self, clip_data: dict):
        """Starte Clip-Preview (optional).

        Args:
            clip_data: Dictionary mit Clip-Daten
        """
        if not self.preview_widget:
            return

        try:
            clip_path = clip_data.get("path")
            if clip_path and Path(clip_path).exists():
                # Lade Video im Preview Widget
                self.preview_widget.load_video(clip_path)
                logger.info(f"Preview gestartet: {clip_path}")
        except Exception as e:
            logger.warning(f"Preview konnte nicht gestartet werden: {e}")

    def _sync_selection_to_pyqt(self, clip_id: int):
        """Synchronisiere Selection mit PyQt Widgets.

        Args:
            clip_id: ID des selektierten Clips
        """
        # Clip Library
        if self.clip_library_widget and hasattr(self.clip_library_widget, "select_clip"):
            try:
                self.clip_library_widget.select_clip(clip_id)
                logger.debug(f"ClipLibrary synchronisiert: Clip {clip_id}")
            except Exception as e:
                logger.debug(f"ClipLibrary sync nicht möglich: {e}")

        # Timeline Widget (PyQt)
        if self.timeline_widget and hasattr(self.timeline_widget, "select_clip"):
            try:
                self.timeline_widget.select_clip(clip_id)
                logger.debug(f"Timeline synchronisiert: Clip {clip_id}")
            except Exception as e:
                logger.debug(f"Timeline sync nicht möglich: {e}")

    def _on_dpg_clips_selected(self, clip_ids: list[int]):
        """Handle multi-selection (Shift+Click, Ctrl+Click).

        Args:
            clip_ids: Liste von Clip-IDs (multi-selection)
        """
        logger.info(f"Multi-Selection: {len(clip_ids)} Clips")
        self.selected_clips = clip_ids

        # Emit multi-selection signal
        self.clips_selected.emit(clip_ids)

        # Update UI für ersten Clip in Auswahl
        if clip_ids:
            clip_data = self._load_clip_data(clip_ids[0])
            if clip_data:
                self._update_clip_details_panel(clip_data)
                self._show_clip_thumbnail(clip_data.get("path"))

        # Status Update
        self.update_status(f"{len(clip_ids)} Clips ausgewählt")

    def _on_dpg_deselect(self):
        """Handle deselection bei Klick auf leeren Bereich."""
        logger.debug("Deselection: Alle Clips abgewählt")
        self.selected_clips = []

        # Clear Clip-Details Panel
        self._clear_clip_details_panel()

        # Emit deselection signal
        self.clip_deselected.emit()

        # Status Update
        self.update_status("Keine Clips ausgewählt")

    def _clear_clip_details_panel(self):
        """Leere das Clip-Details Panel."""
        if self.clip_details_widget and hasattr(self.clip_details_widget, "clear"):
            self.clip_details_widget.clear()
            logger.debug("Clip-Details Panel geleert")

    def closeEvent(self, event):
        """Handle window close event.

        BUGFIX #12: Properly disconnect all signals to prevent memory leaks from lambda captures.
        BUGFIX #14: Use exec() for modal dialog to disable UI interaction during confirmation.
        """
        # BUGFIX #14: Create modal message box that blocks UI interaction
        msg_box = QMessageBox(
            QMessageBox.Icon.Question,
            "Exit",
            "Are you sure you want to exit?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            self,
        )
        msg_box.setDefaultButton(QMessageBox.StandardButton.No)
        msg_box.setModal(True)  # Ensure modal behavior
        reply = msg_box.exec()  # Use exec() instead of question() for better modal control

        if reply == QMessageBox.StandardButton.Yes:
            # BUGFIX #12: Disconnect all signal connections to prevent memory leaks
            try:
                # Disconnect playback signals
                self.playback_started.disconnect()
                self.playback_stopped.disconnect()
                self.render_started.disconnect()
                self.render_finished.disconnect()

                # Disconnect widget signals
                if self.timeline_widget:
                    self.timeline_widget.position_changed.disconnect()
                if self.preview_widget:
                    self.preview_widget.position_changed.disconnect()
                if self.clip_library_widget:
                    self.clip_library_widget.clip_selected.disconnect()
                    self.clip_library_widget.clip_imported.disconnect()
                    self.clip_library_widget.analyze_requested.disconnect()
                if self.parameter_dashboard_widget:
                    self.parameter_dashboard_widget.parameter_changed.disconnect()
                    self.parameter_dashboard_widget.rule_toggled.disconnect()
                    self.parameter_dashboard_widget.preset_loaded.disconnect()

                logger.debug("All signal connections disconnected")
            except Exception as e:
                logger.warning(f"Signal cleanup error (non-critical): {e}")

            # Clean up Dear PyGui if active
            if self.dpg_bridge:
                try:
                    self.dpg_bridge.cleanup()
                    logger.info("Dear PyGui cleaned up")
                except Exception as e:
                    logger.warning(f"Dear PyGui cleanup error: {e}")

            logger.info("Application closing")
            event.accept()
        else:
            event.ignore()
