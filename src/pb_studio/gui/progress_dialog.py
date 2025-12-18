"""
Progress Dialog for long-running operations.

Provides a non-modal dialog with progress bar, status message, and cancel functionality.
Stays on top but allows user to interact with main window (e.g., switch to Console tab).
Supports both determinate (0-100%) and indeterminate progress modes.
"""

import time
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)


class ProgressDialog(QDialog):
    """
    Non-modal progress dialog for long-running operations.

    Features:
    - Progress bar (0-100% or indeterminate)
    - Status message display
    - Cancel button
    - Auto-close on completion
    - Stays on top but allows main window interaction (Console tab access)

    Example:
        >>> dialog = ProgressDialog("Processing", "Analyzing clips...")
        >>> dialog.show()
        >>> # In worker thread:
        >>> dialog.update_progress(50, "Processing clip 5/10")
        >>> dialog.finish("Complete!")
    """

    # Signals
    cancelled = pyqtSignal()  # Emitted when user clicks Cancel

    def __init__(
        self,
        title: str,
        message: str = "",
        parent: Optional["QWidget"] = None,
        indeterminate: bool = False,
        show_cancel: bool = True,
        multi_stage: bool = False,
    ):
        """
        Initialize progress dialog.

        Args:
            title: Dialog window title
            message: Initial status message
            parent: Parent widget
            indeterminate: If True, show indeterminate progress (busy indicator)
            show_cancel: If True, show cancel button
            multi_stage: If True, show two-level progress (main + sub)
        """
        super().__init__(parent)

        self.setWindowTitle(title)
        self.setModal(False)  # Non-modal: User can switch to Console tab
        self.setMinimumWidth(500 if multi_stage else 400)

        # Normal dialog window (can be minimized and go behind other windows)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )

        self._cancelled = False
        self._indeterminate = indeterminate
        self._multi_stage = multi_stage
        self._start_time = time.time()
        self._last_progress = 0

        self._init_ui(message, show_cancel)

    def _init_ui(self, message: str, show_cancel: bool):
        """Initialize UI components."""
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # Main status message
        self.message_label = QLabel(message)
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        self.message_label.setFont(font)
        layout.addWidget(self.message_label)

        # Main progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100 if not self._indeterminate else 0)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(not self._indeterminate)
        self.progress_bar.setMinimumHeight(25)
        layout.addWidget(self.progress_bar)

        # Main progress info line (percentage + ETA)
        if not self._indeterminate:
            info_layout = QHBoxLayout()
            self.percent_label = QLabel("0%")
            self.percent_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            info_layout.addWidget(self.percent_label)

            info_layout.addStretch()

            self.eta_label = QLabel("ETA: Calculating...")
            self.eta_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.eta_label.setStyleSheet("color: #888;")
            info_layout.addWidget(self.eta_label)

            layout.addLayout(info_layout)

        # Multi-stage: Sub-progress section
        if self._multi_stage and not self._indeterminate:
            # Separator
            separator = QFrame()
            separator.setFrameShape(QFrame.Shape.HLine)
            separator.setFrameShadow(QFrame.Shadow.Sunken)
            separator.setStyleSheet("background-color: #555;")
            layout.addWidget(separator)

            # Sub-task label
            self.subtask_label = QLabel("...")
            self.subtask_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            sub_font = QFont()
            sub_font.setPointSize(9)
            self.subtask_label.setFont(sub_font)
            self.subtask_label.setStyleSheet("color: #aaa;")
            layout.addWidget(self.subtask_label)

            # Sub progress bar
            self.sub_progress_bar = QProgressBar()
            self.sub_progress_bar.setMinimum(0)
            self.sub_progress_bar.setMaximum(100)
            self.sub_progress_bar.setValue(0)
            self.sub_progress_bar.setTextVisible(True)
            self.sub_progress_bar.setMinimumHeight(20)
            layout.addWidget(self.sub_progress_bar)

            # Sub progress info
            sub_info_layout = QHBoxLayout()
            self.sub_percent_label = QLabel("0%")
            self.sub_percent_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            self.sub_percent_label.setStyleSheet("color: #aaa; font-size: 9pt;")
            sub_info_layout.addWidget(self.sub_percent_label)

            sub_info_layout.addStretch()

            self.sub_detail_label = QLabel("")
            self.sub_detail_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.sub_detail_label.setStyleSheet("color: #888; font-size: 9pt;")
            sub_info_layout.addWidget(self.sub_detail_label)

            layout.addLayout(sub_info_layout)

        # Buttons
        if show_cancel:
            button_layout = QHBoxLayout()
            button_layout.addStretch()

            self.cancel_button = QPushButton("Cancel")
            self.cancel_button.clicked.connect(self._on_cancel)
            button_layout.addWidget(self.cancel_button)

            layout.addLayout(button_layout)

        self.setLayout(layout)

    def _on_cancel(self):
        """Handle cancel button click."""
        self._cancelled = True
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("Cancelling...")
        self.cancelled.emit()

    def update_progress(self, value: int, message: str | None = None):
        """
        Update progress bar and optional message.

        Args:
            value: Progress value (0-100)
            message: Optional new status message
        """
        if self._indeterminate:
            # Indeterminate mode - just update message
            if message:
                self.message_label.setText(message)
            return

        # Clamp value to 0-100
        value = max(0, min(100, value))

        self.progress_bar.setValue(value)
        if hasattr(self, "percent_label"):
            self.percent_label.setText(f"{value}%")

        if message:
            self.message_label.setText(message)

        # Update ETA
        if hasattr(self, "eta_label") and value > 0:
            self._update_eta(value)

        self._last_progress = value

        # Process events to keep UI responsive
        from PyQt6.QtWidgets import QApplication

        QApplication.processEvents()

    def update_sub_progress(
        self, value: int, subtask: str | None = None, detail: str | None = None
    ):
        """
        Update sub-progress (only for multi-stage dialogs).

        Args:
            value: Sub-progress value (0-100)
            subtask: Optional sub-task description
            detail: Optional detail text (e.g., "1000/5000")
        """
        if not self._multi_stage:
            return

        # Clamp value to 0-100
        value = max(0, min(100, value))

        if hasattr(self, "sub_progress_bar"):
            self.sub_progress_bar.setValue(value)

        if hasattr(self, "sub_percent_label"):
            self.sub_percent_label.setText(f"{value}%")

        if subtask and hasattr(self, "subtask_label"):
            self.subtask_label.setText(subtask)

        if detail and hasattr(self, "sub_detail_label"):
            self.sub_detail_label.setText(detail)

        # Process events to keep UI responsive
        from PyQt6.QtWidgets import QApplication

        QApplication.processEvents()

    def _update_eta(self, current_progress: int):
        """
        Calculate and update ETA based on progress.

        Args:
            current_progress: Current progress (0-100)
        """
        if current_progress <= 0 or current_progress >= 100:
            if hasattr(self, "eta_label"):
                self.eta_label.setText("ETA: --:--")
            return

        # Calculate elapsed time
        elapsed = time.time() - self._start_time

        # Estimate total time based on current progress
        estimated_total = elapsed / (current_progress / 100.0)

        # Calculate remaining time
        remaining = estimated_total - elapsed

        # Format as MM:SS or HH:MM:SS
        if remaining < 60:
            eta_text = f"ETA: {int(remaining)}s"
        elif remaining < 3600:
            minutes = int(remaining / 60)
            seconds = int(remaining % 60)
            eta_text = f"ETA: {minutes}:{seconds:02d}"
        else:
            hours = int(remaining / 3600)
            minutes = int((remaining % 3600) / 60)
            eta_text = f"ETA: {hours}h {minutes}m"

        if hasattr(self, "eta_label"):
            self.eta_label.setText(eta_text)

    def set_indeterminate(self, indeterminate: bool):
        """
        Switch between determinate and indeterminate mode.

        Args:
            indeterminate: If True, show busy indicator
        """
        self._indeterminate = indeterminate
        self.progress_bar.setMaximum(0 if indeterminate else 100)
        self.progress_bar.setTextVisible(not indeterminate)

        if hasattr(self, "percent_label"):
            self.percent_label.setVisible(not indeterminate)

    def finish(self, message: str = "Complete!", auto_close: bool = True):
        """
        Mark operation as complete.

        Args:
            message: Completion message
            auto_close: If True, close dialog after 1 second
        """
        self.progress_bar.setValue(100)
        self.message_label.setText(message)

        if hasattr(self, "cancel_button"):
            self.cancel_button.setEnabled(False)

        if auto_close:
            # Close after 1 second
            QTimer.singleShot(1000, self.accept)

    def is_cancelled(self) -> bool:
        """
        Check if user cancelled the operation.

        Returns:
            True if cancelled
        """
        return self._cancelled


class ProgressCallback:
    """
    Callback wrapper for progress updates.

    Simplifies progress reporting from worker threads.

    Example:
        >>> dialog = ProgressDialog("Processing")
        >>> callback = ProgressCallback(dialog)
        >>>
        >>> for i, item in enumerate(items):
        >>>     if callback.is_cancelled():
        >>>         break
        >>>     process(item)
        >>>     callback.update(i, len(items), f"Processing {item}")
    """

    def __init__(self, dialog: ProgressDialog):
        """
        Initialize callback.

        Args:
            dialog: Progress dialog to update
        """
        self.dialog = dialog

    def update(self, current: int, total: int, message: str | None = None):
        """
        Update progress based on current/total.

        Args:
            current: Current item index (0-based)
            total: Total number of items
            message: Optional status message
        """
        if total > 0:
            percent = int((current / total) * 100)
            self.dialog.update_progress(percent, message)
        else:
            self.dialog.update_progress(0, message)

    def update_percent(self, percent: float, message: str | None = None):
        """
        Update progress with percentage.

        Args:
            percent: Progress percentage (0.0 - 1.0)
            message: Optional status message
        """
        self.dialog.update_progress(int(percent * 100), message)

    def is_cancelled(self) -> bool:
        """
        Check if operation was cancelled.

        Returns:
            True if cancelled
        """
        return self.dialog.is_cancelled()

    def finish(self, message: str = "Complete!"):
        """
        Mark operation as complete.

        Args:
            message: Completion message
        """
        self.dialog.finish(message)
