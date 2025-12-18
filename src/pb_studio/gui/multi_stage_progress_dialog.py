"""
Multi-Stage Progress Dialog - Hybrid aus Timeline und Dashboard.

Kombiniert Timeline-Workflow-Visualisierung mit Dashboard-Kacheln
für übersichtliche Fortschrittsanzeige bei komplexen Multi-Phasen-Operationen.

Design:
- Oben: Horizontale Timeline mit Phasen-Flow
- Mitte: 2x2 Dashboard mit Kacheln (Audio, Video, Pacing, Rendering)
- Unten: Gesamt-Fortschritt + Aktionen

Author: PB_studio Development Team
Date: 2025-12-02
"""

import time
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from ..utils.logger import get_logger

logger = get_logger(__name__)


class PhaseStatus:
    """Status-Enum für Phasen."""

    WAITING = "waiting"  # ⏸ Wartet
    RUNNING = "running"  # ⏳ Läuft
    COMPLETED = "completed"  # ✓ Fertig
    ERROR = "error"  # ✗ Fehler


class StageCard(QFrame):
    """
    Dashboard-Kachel für eine einzelne Phase.

    Zeigt:
    - Phase-Name + Icon
    - Status (Wartet/Läuft/Fertig/Fehler)
    - Fortschrittsbalken
    - Wichtige Metriken
    - Sub-Analysen (bei aktiver Phase)
    - ETA
    """

    def __init__(self, title: str, icon: str, color: str, parent: Optional["QWidget"] = None):
        """
        Initialize stage card.

        Args:
            title: Phase title (z.B. "AUDIO-ANALYSE")
            icon: Emoji icon (z.B. "🎵")
            color: Hex color (z.B. "#2E5C8A")
            parent: Parent widget
        """
        super().__init__(parent)

        self.title = title
        self.icon = icon
        self.color = color
        self.status = PhaseStatus.WAITING
        self.progress = 0
        self.metrics: dict[str, str] = {}
        self.sub_tasks: dict[str, int] = {}  # {name: progress %}
        self.eta_seconds = 0

        self._init_ui()

    def _init_ui(self):
        """Initialize UI components."""
        # Card styling
        self.setFrameShape(QFrame.Shape.Box)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.setLineWidth(2)
        self.setStyleSheet(
            f"""
            StageCard {{
                background-color: #2A2A2A;
                border: 2px solid {self.color};
                border-radius: 8px;
                padding: 10px;
            }}
        """
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Header: Icon + Title
        header_layout = QHBoxLayout()

        self.icon_label = QLabel(self.icon)
        icon_font = QFont()
        icon_font.setPointSize(16)
        self.icon_label.setFont(icon_font)
        header_layout.addWidget(self.icon_label)

        self.title_label = QLabel(self.title)
        title_font = QFont()
        title_font.setPointSize(10)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet(f"color: {self.color};")
        header_layout.addWidget(self.title_label)

        header_layout.addStretch()
        layout.addLayout(header_layout)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("background-color: #555;")
        layout.addWidget(separator)

        # Status
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Status: ⏸ Wartet")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        layout.addLayout(status_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimumHeight(20)
        self.progress_bar.setStyleSheet(
            f"""
            QProgressBar {{
                border: 1px solid #555;
                border-radius: 3px;
                text-align: center;
                background-color: #1A1A1A;
            }}
            QProgressBar::chunk {{
                background-color: {self.color};
            }}
        """
        )
        layout.addWidget(self.progress_bar)

        # Metrics container (expandable)
        self.metrics_container = QFrame()
        metrics_layout = QVBoxLayout(self.metrics_container)
        metrics_layout.setSpacing(2)
        metrics_layout.setContentsMargins(0, 5, 0, 0)
        self.metrics_labels: list[QLabel] = []
        layout.addWidget(self.metrics_container)

        # Sub-tasks container (only for active phase)
        self.subtasks_container = QFrame()
        self.subtasks_container.setVisible(False)
        subtasks_layout = QVBoxLayout(self.subtasks_container)
        subtasks_layout.setSpacing(3)
        subtasks_layout.setContentsMargins(0, 5, 0, 0)

        subtasks_header = QLabel("Sub-Analysen:")
        subtasks_header.setStyleSheet("color: #888; font-size: 9pt;")
        subtasks_layout.addWidget(subtasks_header)

        self.subtask_labels: dict[str, QLabel] = {}
        layout.addWidget(self.subtasks_container)

        # ETA
        self.eta_label = QLabel("")
        self.eta_label.setStyleSheet("color: #888; font-size: 9pt;")
        self.eta_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.eta_label)

        layout.addStretch()

    def set_status(self, status: str):
        """
        Update phase status.

        Args:
            status: One of PhaseStatus constants
        """
        self.status = status

        status_map = {
            PhaseStatus.WAITING: ("⏸ Wartet", "#888"),
            PhaseStatus.RUNNING: ("⏳ Läuft", self.color),
            PhaseStatus.COMPLETED: ("✓ Fertig", "#2D7A3E"),
            PhaseStatus.ERROR: ("✗ Fehler", "#C53030"),
        }

        text, color = status_map.get(status, ("?", "#888"))
        self.status_label.setText(f"Status: {text}")
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")

    def set_progress(self, value: int):
        """
        Update progress bar.

        Args:
            value: Progress value (0-100)
        """
        self.progress = max(0, min(100, value))
        self.progress_bar.setValue(self.progress)

    def set_metrics(self, metrics: dict[str, str]):
        """
        Update metrics display.

        HANDLE LEAK FIX: Reuse existing QLabel widgets instead of creating new ones!
        Creating/destroying thousands of QLabels during render exhausts Windows USER handles.

        Args:
            metrics: Dict of metric_name: value
        """
        self.metrics = metrics
        layout = self.metrics_container.layout()
        items = list(metrics.items())

        # HANDLE LEAK FIX: Reuse existing labels (no new widget creation!)
        for i, (key, value) in enumerate(items):
            if i < len(self.metrics_labels):
                # REUSE existing label - just update text (no handle allocation)
                self.metrics_labels[i].setText(f"• {key}: {value}")
            else:
                # Only create new label if we need more than we have
                label = QLabel(f"• {key}: {value}")
                label.setStyleSheet("color: #CCC; font-size: 9pt;")
                layout.addWidget(label)
                self.metrics_labels.append(label)

        # Remove excess labels only if metrics dict shrunk (rare)
        while len(self.metrics_labels) > len(items):
            label = self.metrics_labels.pop()
            layout.removeWidget(label)
            label.setParent(None)
            label.deleteLater()

    def set_subtasks(self, subtasks: dict[str, int], show: bool = True):
        """
        Update sub-tasks display.

        HANDLE LEAK FIX: Hide unused labels instead of deleting them.
        Reuse existing labels where possible to minimize handle allocations.

        Args:
            subtasks: Dict of task_name: progress%
            show: If True, show subtasks container
        """
        self.sub_tasks = subtasks
        self.subtasks_container.setVisible(show and len(subtasks) > 0)

        if not show:
            return

        # HANDLE LEAK FIX: Hide labels for removed tasks instead of deleting
        removed_tasks = [name for name in self.subtask_labels if name not in subtasks]
        for task_name in removed_tasks:
            label = self.subtask_labels.pop(task_name)
            label.setVisible(False)  # Hide instead of delete (no handle churn)

        # Update or reuse labels
        for task_name, progress in subtasks.items():
            if task_name not in self.subtask_labels:
                # Try to reuse a hidden label first
                label = QLabel()
                label.setStyleSheet("color: #AAA; font-size: 9pt;")
                self.subtasks_container.layout().addWidget(label)
                self.subtask_labels[task_name] = label
            else:
                # Ensure label is visible (may have been hidden)
                self.subtask_labels[task_name].setVisible(True)

            # Status icon based on progress
            if progress >= 100:
                icon = "✓"
                color = "#2D7A3E"
            elif progress > 0:
                icon = "⏳"
                color = self.color
            else:
                icon = "○"
                color = "#555"

            self.subtask_labels[task_name].setText(f"{icon} {task_name}  {progress}%")
            self.subtask_labels[task_name].setStyleSheet(f"color: {color}; font-size: 9pt;")

    def set_eta(self, seconds: int):
        """
        Update ETA display.

        Args:
            seconds: Remaining seconds
        """
        self.eta_seconds = seconds

        if seconds <= 0:
            self.eta_label.setText("")
            return

        if seconds < 60:
            eta_text = f"⏱ ETA: {seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            eta_text = f"⏱ ETA: {minutes}:{secs:02d} min"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            eta_text = f"⏱ ETA: {hours}h {minutes}m"

        self.eta_label.setText(eta_text)


class TimelineBar(QFrame):
    """
    Horizontale Timeline für Workflow-Visualisierung.

    Zeigt:
    - 4 Phasen-Icons mit Pfeilen
    - Mini-Fortschrittsbalken unter jedem Icon
    - Aktuelle Phase hervorgehoben
    """

    def __init__(self, parent: Optional["QWidget"] = None):
        super().__init__(parent)

        self.phases = [
            {"name": "AUDIO", "icon": "🎵", "color": "#2E5C8A", "progress": 0},
            {"name": "VIDEO", "icon": "🎬", "color": "#2D7A3E", "progress": 0},
            {"name": "PACING", "icon": "⚙️", "color": "#D97706", "progress": 0},
            {"name": "RENDER", "icon": "🎥", "color": "#C53030", "progress": 0},
        ]

        self.current_phase = 0

        self._init_ui()

    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout(self)
        layout.setSpacing(5)

        # Timeline icons + arrows
        icons_layout = QHBoxLayout()
        icons_layout.setSpacing(20)

        self.icon_labels: list[QLabel] = []
        self.arrow_labels: list[QLabel] = []

        for i, phase in enumerate(self.phases):
            # Phase icon
            icon_label = QLabel(phase["icon"])
            icon_font = QFont()
            icon_font.setPointSize(20)
            icon_label.setFont(icon_font)
            icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            icons_layout.addWidget(icon_label)
            self.icon_labels.append(icon_label)

            # Arrow (except after last phase)
            if i < len(self.phases) - 1:
                arrow = QLabel("─────▶")
                arrow.setStyleSheet("color: #555; font-size: 14pt;")
                arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
                icons_layout.addWidget(arrow)
                self.arrow_labels.append(arrow)

        layout.addLayout(icons_layout)

        # Phase names
        names_layout = QHBoxLayout()
        names_layout.setSpacing(20)

        self.name_labels: list[QLabel] = []

        for i, phase in enumerate(self.phases):
            name_label = QLabel(phase["name"])
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_label.setStyleSheet("color: #888; font-size: 9pt;")
            names_layout.addWidget(name_label)
            self.name_labels.append(name_label)

            if i < len(self.phases) - 1:
                # Spacer for arrow
                spacer = QLabel("")
                spacer.setFixedWidth(80)
                names_layout.addWidget(spacer)

        layout.addLayout(names_layout)

        # Mini progress bars
        progress_layout = QHBoxLayout()
        progress_layout.setSpacing(20)

        self.mini_progress_bars: list[QProgressBar] = []

        for i, phase in enumerate(self.phases):
            mini_bar = QProgressBar()
            mini_bar.setMinimum(0)
            mini_bar.setMaximum(100)
            mini_bar.setValue(0)
            mini_bar.setTextVisible(True)
            mini_bar.setMaximumHeight(12)
            mini_bar.setMaximumWidth(80)
            mini_bar.setStyleSheet(
                f"""
                QProgressBar {{
                    border: 1px solid #555;
                    border-radius: 3px;
                    text-align: center;
                    background-color: #1A1A1A;
                    font-size: 8pt;
                }}
                QProgressBar::chunk {{
                    background-color: {phase["color"]};
                }}
            """
            )
            progress_layout.addWidget(mini_bar, 0, Qt.AlignmentFlag.AlignCenter)
            self.mini_progress_bars.append(mini_bar)

            if i < len(self.phases) - 1:
                # Spacer for arrow
                spacer = QLabel("")
                spacer.setFixedWidth(80)
                progress_layout.addWidget(spacer)

        layout.addLayout(progress_layout)

    def set_phase_progress(self, phase_index: int, progress: int):
        """
        Update progress for a specific phase.

        Args:
            phase_index: Phase index (0-3)
            progress: Progress value (0-100)
        """
        if 0 <= phase_index < len(self.phases):
            self.phases[phase_index]["progress"] = progress
            self.mini_progress_bars[phase_index].setValue(progress)

    def set_current_phase(self, phase_index: int):
        """
        Highlight current active phase.

        Args:
            phase_index: Phase index (0-3)
        """
        self.current_phase = phase_index

        for i, (icon_label, name_label) in enumerate(zip(self.icon_labels, self.name_labels)):
            if i == phase_index:
                # Highlight active phase
                icon_label.setStyleSheet(
                    f"background-color: {self.phases[i]['color']}; padding: 5px; border-radius: 5px;"
                )
                name_label.setStyleSheet(
                    f"color: {self.phases[i]['color']}; font-size: 10pt; font-weight: bold;"
                )
            else:
                # Dim inactive phases
                icon_label.setStyleSheet("")
                name_label.setStyleSheet("color: #888; font-size: 9pt;")


class MultiStageProgressDialog(QDialog):
    """
    Multi-Stage Progress Dialog - Hybrid aus Timeline und Dashboard.

    Kombiniert:
    - Timeline-Workflow-Visualisierung (oben)
    - Dashboard-Kacheln für alle 4 Phasen (mitte)
    - Gesamt-Fortschritt + Aktionen (unten)

    Verwendung:
        >>> dialog = MultiStageProgressDialog("Video Production", parent=self)
        >>> dialog.show()
        >>>
        >>> # Audio-Phase
        >>> dialog.set_phase_progress(0, 100)
        >>> dialog.update_stage(0, "completed", metrics={"BPM": "128.5", "Beats": "512"})
        >>>
        >>> # Video-Phase
        >>> dialog.set_current_phase(1)
        >>> dialog.set_phase_progress(1, 42)
        >>> dialog.update_stage(1, "running", 42,
        >>>     metrics={"Clip": "42/99", "Aktuell": "motion_001.mp4"},
        >>>     subtasks={"Farben": 100, "Bewegung": 75, "Szene": 50, "Style": 0}
        >>> )
    """

    # Signals
    cancelled = pyqtSignal()  # User clicked cancel
    paused = pyqtSignal()  # User clicked pause
    details_requested = pyqtSignal()  # User clicked details

    def __init__(self, title: str = "Multi-Stage Operation", parent: Optional["QWidget"] = None):
        """
        Initialize multi-stage progress dialog.

        Args:
            title: Dialog window title
            parent: Parent widget
        """
        super().__init__(parent)

        self.setWindowTitle(title)
        self.setModal(False)  # Non-modal: User can interact with main window
        self.setMinimumSize(800, 700)

        # Normal dialog window (can be minimized and go behind other windows)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )

        self._cancelled = False
        self._paused = False
        self._start_time = time.time()
        self._overall_progress = 0

        self._init_ui()

    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Header
        header_layout = QHBoxLayout()
        header_label = QLabel("Video Production Pipeline")
        header_font = QFont()
        header_font.setPointSize(12)
        header_font.setBold(True)
        header_label.setFont(header_font)
        header_layout.addWidget(header_label)

        header_layout.addStretch()

        self.overall_percent_label = QLabel("Gesamt: 0%")
        overall_font = QFont()
        overall_font.setPointSize(11)
        overall_font.setBold(True)
        self.overall_percent_label.setFont(overall_font)
        self.overall_percent_label.setStyleSheet("color: #4A9EFF;")
        header_layout.addWidget(self.overall_percent_label)

        layout.addLayout(header_layout)

        # Timeline bar
        self.timeline = TimelineBar()
        layout.addWidget(self.timeline)

        # Separator
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.Shape.HLine)
        separator1.setFrameShadow(QFrame.Shadow.Sunken)
        separator1.setStyleSheet("background-color: #555;")
        layout.addWidget(separator1)

        # Dashboard: 2x2 Grid mit Stage-Cards
        dashboard_grid = QGridLayout()
        dashboard_grid.setSpacing(15)

        # Create 4 stage cards
        self.stage_cards: list[StageCard] = []

        stages = [
            ("AUDIO-ANALYSE", "🎵", "#2E5C8A", 0, 0),
            ("VIDEO-ANALYSE", "🎬", "#2D7A3E", 0, 1),
            ("PACING ENGINE", "⚙️", "#D97706", 1, 0),
            ("VIDEO-RENDERING", "🎥", "#C53030", 1, 1),
        ]

        for title, icon, color, row, col in stages:
            card = StageCard(title, icon, color)
            dashboard_grid.addWidget(card, row, col)
            self.stage_cards.append(card)

        layout.addLayout(dashboard_grid)

        # Separator
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.HLine)
        separator2.setFrameShadow(QFrame.Shadow.Sunken)
        separator2.setStyleSheet("background-color: #555;")
        layout.addWidget(separator2)

        # Overall progress bar (unten)
        overall_group = QGroupBox("Gesamt-Fortschritt")
        overall_layout = QVBoxLayout(overall_group)

        self.overall_progress_bar = QProgressBar()
        self.overall_progress_bar.setMinimum(0)
        self.overall_progress_bar.setMaximum(100)
        self.overall_progress_bar.setValue(0)
        self.overall_progress_bar.setTextVisible(True)
        self.overall_progress_bar.setMinimumHeight(30)
        self.overall_progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 2px solid #555;
                border-radius: 5px;
                text-align: center;
                background-color: #1A1A1A;
                font-size: 11pt;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2E5C8A, stop:0.33 #2D7A3E,
                    stop:0.66 #D97706, stop:1 #C53030
                );
            }
        """
        )
        overall_layout.addWidget(self.overall_progress_bar)

        # Overall stats
        stats_layout = QHBoxLayout()
        self.elapsed_label = QLabel("Gesamtzeit: 0:00")
        self.remaining_label = QLabel("Verbleibend: --:--")
        self.errors_label = QLabel("Fehler: 0")

        for label in [self.elapsed_label, self.remaining_label, self.errors_label]:
            label.setStyleSheet("color: #AAA; font-size: 9pt;")

        stats_layout.addWidget(self.elapsed_label)
        stats_layout.addWidget(QLabel("|"))
        stats_layout.addWidget(self.remaining_label)
        stats_layout.addWidget(QLabel("|"))
        stats_layout.addWidget(self.errors_label)
        stats_layout.addStretch()

        overall_layout.addLayout(stats_layout)
        layout.addWidget(overall_group)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.pause_button = QPushButton("⏸ Pause")
        self.pause_button.clicked.connect(self._on_pause)
        button_layout.addWidget(self.pause_button)

        self.cancel_button = QPushButton("✕ Abbrechen")
        self.cancel_button.clicked.connect(self._on_cancel)
        button_layout.addWidget(self.cancel_button)

        self.details_button = QPushButton("📋 Details >>")
        self.details_button.clicked.connect(self._on_details)
        button_layout.addWidget(self.details_button)

        layout.addLayout(button_layout)

        # Timer for elapsed time update
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_elapsed_time)
        self.timer.start(1000)  # Update every second

    def set_current_phase(self, phase_index: int):
        """
        Set current active phase.

        Args:
            phase_index: Phase index (0=Audio, 1=Video, 2=Pacing, 3=Render)
        """
        self.timeline.set_current_phase(phase_index)

    def set_phase_progress(self, phase_index: int, progress: int):
        """
        Update progress for a specific phase.

        Args:
            phase_index: Phase index (0-3)
            progress: Progress value (0-100)
        """
        # Update timeline
        self.timeline.set_phase_progress(phase_index, progress)

        # Update stage card
        if 0 <= phase_index < len(self.stage_cards):
            self.stage_cards[phase_index].set_progress(progress)

        # Recalculate overall progress
        self._update_overall_progress()

    def update_stage(
        self,
        stage_index: int,
        status: str,
        progress: int | None = None,
        metrics: dict[str, str] | None = None,
        subtasks: dict[str, int] | None = None,
        eta_seconds: int | None = None,
    ):
        """
        Update a stage card completely.

        Args:
            stage_index: Stage index (0-3)
            status: One of PhaseStatus constants
            progress: Optional progress (0-100)
            metrics: Optional metrics dict
            subtasks: Optional subtasks dict (name: progress%)
            eta_seconds: Optional ETA in seconds
        """
        if 0 <= stage_index < len(self.stage_cards):
            card = self.stage_cards[stage_index]

            card.set_status(status)

            if progress is not None:
                card.set_progress(progress)
                self.set_phase_progress(stage_index, progress)

            if metrics is not None:
                card.set_metrics(metrics)

            if subtasks is not None:
                # Show subtasks only for running phase
                show_subtasks = status == PhaseStatus.RUNNING
                card.set_subtasks(subtasks, show_subtasks)

            if eta_seconds is not None:
                card.set_eta(eta_seconds)

    def _update_overall_progress(self):
        """Recalculate overall progress from all phases."""
        # Simple average of all phase progresses
        total_progress = sum(card.progress for card in self.stage_cards)
        self._overall_progress = total_progress // len(self.stage_cards)

        self.overall_progress_bar.setValue(self._overall_progress)
        self.overall_percent_label.setText(f"Gesamt: {self._overall_progress}%")

        # Update remaining time estimate
        if self._overall_progress > 0:
            elapsed = time.time() - self._start_time
            estimated_total = elapsed / (self._overall_progress / 100.0)
            remaining = int(estimated_total - elapsed)

            if remaining > 0:
                if remaining < 60:
                    self.remaining_label.setText(f"Verbleibend: {remaining}s")
                elif remaining < 3600:
                    minutes = remaining // 60
                    seconds = remaining % 60
                    self.remaining_label.setText(f"Verbleibend: {minutes}:{seconds:02d}")
                else:
                    hours = remaining // 3600
                    minutes = (remaining % 3600) // 60
                    self.remaining_label.setText(f"Verbleibend: {hours}h {minutes}m")

    def _update_elapsed_time(self):
        """Update elapsed time label."""
        elapsed = int(time.time() - self._start_time)

        if elapsed < 60:
            self.elapsed_label.setText(f"Gesamtzeit: 0:{elapsed:02d}")
        elif elapsed < 3600:
            minutes = elapsed // 60
            seconds = elapsed % 60
            self.elapsed_label.setText(f"Gesamtzeit: {minutes}:{seconds:02d}")
        else:
            hours = elapsed // 3600
            minutes = (elapsed % 3600) // 60
            self.elapsed_label.setText(f"Gesamtzeit: {hours}h {minutes}m")

    def set_error_count(self, count: int):
        """
        Update error counter.

        Args:
            count: Number of errors
        """
        self.errors_label.setText(f"Fehler: {count}")
        if count > 0:
            self.errors_label.setStyleSheet("color: #C53030; font-weight: bold; font-size: 9pt;")
        else:
            self.errors_label.setStyleSheet("color: #AAA; font-size: 9pt;")

    def _on_pause(self):
        """Handle pause button click."""
        self._paused = not self._paused

        if self._paused:
            self.pause_button.setText("▶ Fortsetzen")
            self.paused.emit()
        else:
            self.pause_button.setText("⏸ Pause")
            # Resume signal would go here

    def _on_cancel(self):
        """Handle cancel button click."""
        self._cancelled = True
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("Abbrechen...")
        self.cancelled.emit()

    def _on_details(self):
        """Handle details button click."""
        self.details_requested.emit()

    def is_cancelled(self) -> bool:
        """Check if user cancelled operation."""
        return self._cancelled

    def is_paused(self) -> bool:
        """Check if operation is paused."""
        return self._paused

    def finish(self, success: bool = True, message: str = "Fertig!"):
        """
        Mark operation as complete.

        Args:
            success: If True, show success, else show error
            message: Completion message
        """
        if success:
            self.overall_progress_bar.setValue(100)
            self.overall_percent_label.setText("Gesamt: 100% ✓")
            self.overall_percent_label.setStyleSheet("color: #2D7A3E; font-weight: bold;")
        else:
            self.overall_percent_label.setText("Fehler!")
            self.overall_percent_label.setStyleSheet("color: #C53030; font-weight: bold;")

        self.pause_button.setEnabled(False)
        self.cancel_button.setEnabled(False)

        # Auto-close after 2 seconds (nur bei Erfolg)
        if success:
            QTimer.singleShot(2000, self.accept)

    def closeEvent(self, event):
        """Handle window close event."""
        if not self._cancelled and self._overall_progress < 100:
            # Ask for confirmation
            from PyQt6.QtWidgets import QMessageBox

            reply = QMessageBox.question(
                self,
                "Abbrechen?",
                "Möchten Sie den Vorgang wirklich abbrechen?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                self._on_cancel()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
