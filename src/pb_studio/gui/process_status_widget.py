from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from pb_studio.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class _ProcessEntry:
    """Internal helper storing widgets for a running process."""

    process_id: str
    label: QLabel
    status_label: QLabel
    progress_bar: QProgressBar
    wrapper: QFrame


class ProcessStatusWidget(QWidget):
    """Shows a dockable list of active pipeline tasks with per-process progress."""

    DEFAULT_PROCESSES: tuple[tuple[str, str, str], ...] = (
        ("audio_analysis", "Audio Analysis", "🎵"),
        ("clip_analysis", "Clip Analysis", "🎬"),
        ("stem_analysis", "Stem Analysis", "🥁"),
        ("preview_generation", "Preview Generation", "👀"),
        ("render_video", "Render Video", "🎥"),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(260)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        self._entries: dict[str, _ProcessEntry] = {}
        for process_id, display, icon in self.DEFAULT_PROCESSES:
            entry = self._create_entry(process_id, f"{icon} {display}")
            self._entries[process_id] = entry
            layout.addWidget(entry.wrapper)

        layout.addStretch()

    def _create_entry(self, process_id: str, display_name: str) -> _ProcessEntry:
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setFrameShadow(QFrame.Shadow.Raised)
        frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(6, 6, 6, 6)
        frame_layout.setSpacing(4)

        header = QHBoxLayout()
        name_label = QLabel(display_name)
        name_label.setStyleSheet("font-weight: bold;")
        status_label = QLabel("Idle")
        status_label.setStyleSheet("color: #888888; font-size: 11px;")
        status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(name_label)
        header.addWidget(status_label)
        frame_layout.addLayout(header)

        progress_bar = QProgressBar()
        progress_bar.setTextVisible(True)
        progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_bar.setMinimum(0)
        progress_bar.setMaximum(100)
        progress_bar.setValue(0)
        progress_bar.setMinimumHeight(14)
        frame_layout.addWidget(progress_bar)

        entry = _ProcessEntry(
            process_id=process_id,
            label=name_label,
            status_label=status_label,
            progress_bar=progress_bar,
            wrapper=frame,
        )
        return entry

    def start_process(
        self,
        process_id: str,
        message: str | None = None,
        determinate: bool = True,
    ) -> None:
        """Mark a process as started and reset its progress."""
        entry = self._entries.get(process_id)
        if not entry:
            logger.debug(f"Unknown process id '{process_id}'")
            return

        entry.progress_bar.setRange(0, 100 if determinate else 0)
        entry.progress_bar.setValue(0)
        entry.status_label.setText(message or "Running...")
        if not determinate:
            entry.progress_bar.setRange(0, 0)  # Indeterminate animation
        else:
            entry.progress_bar.setRange(0, 100)

    def update_process(
        self,
        process_id: str,
        percent: int | None = None,
        message: str | None = None,
    ) -> None:
        """Update the progress bar and message for a running process."""
        entry = self._entries.get(process_id)
        if not entry:
            logger.debug(f"Unknown process id '{process_id}'")
            return

        if percent is not None and entry.progress_bar.maximum() != 0:
            entry.progress_bar.setValue(max(0, min(100, percent)))
        if message:
            entry.status_label.setText(message)

    def finish_process(
        self,
        process_id: str,
        success: bool = True,
        message: str | None = None,
    ) -> None:
        """Mark a process as finished (success/failure)."""
        entry = self._entries.get(process_id)
        if not entry:
            logger.debug(f"Unknown process id '{process_id}'")
            return

        entry.progress_bar.setRange(0, 100)
        entry.progress_bar.setValue(100 if success else 0)
        suffix = "Done" if success else "Failed"
        entry.status_label.setText(message or suffix)
