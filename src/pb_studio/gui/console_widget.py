"""
Console Widget für Live-Logging in der GUI

Zeigt alle Log-Messages in Echtzeit in einem scrollbaren Textfeld.
Farbcodiert nach Log-Level (INFO, WARNING, ERROR).

Author: PB_studio Development Team
"""

import logging

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QTextEdit, QVBoxLayout, QWidget


class LogSignal(QObject):
    """Qt Signal für Thread-safe Logging."""

    new_log = pyqtSignal(str, str)  # message, level


class ConsoleWidget(QWidget):
    """
    Console-Widget das alle Logs anzeigt.

    Features:
    - Echtzeit Log-Anzeige
    - Farbcodierung nach Level
    - Auto-Scroll
    - Clear-Button
    - Max. 1000 Zeilen (Performance)
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.max_lines = 1000
        self.log_signal = LogSignal()

        self._init_ui()
        self._setup_logging_handler()

    def _init_ui(self):
        """UI aufbauen."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Console TextEdit
        self.console_text = QTextEdit()
        self.console_text.setReadOnly(True)
        self.console_text.setMaximumHeight(200)

        # Dark Theme Styling
        self.console_text.setStyleSheet(
            """
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10pt;
                border: 1px solid #3c3c3c;
                padding: 4px;
            }
        """
        )

        layout.addWidget(self.console_text)

        # Buttons
        button_layout = QHBoxLayout()

        self.clear_btn = QPushButton("Clear Console")
        self.clear_btn.clicked.connect(self.clear_console)
        self.clear_btn.setMaximumWidth(120)

        self.auto_scroll_btn = QPushButton("Auto-Scroll: ON")
        self.auto_scroll_btn.setCheckable(True)
        self.auto_scroll_btn.setChecked(True)
        self.auto_scroll_btn.clicked.connect(self._toggle_auto_scroll)
        self.auto_scroll_btn.setMaximumWidth(120)

        button_layout.addWidget(self.clear_btn)
        button_layout.addWidget(self.auto_scroll_btn)
        button_layout.addStretch()

        layout.addLayout(button_layout)

        # Signal verbinden
        self.log_signal.new_log.connect(self._append_log)

    def _setup_logging_handler(self):
        """Logging Handler einrichten der alle Logs abfängt."""
        handler = GUILogHandler(self.log_signal)
        handler.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)

        # Root Logger erweitern
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)

    def _append_log(self, message: str, level: str):
        """Log-Nachricht hinzufügen (Thread-safe)."""
        # Farbcodierung nach Level
        if level == "ERROR" or level == "CRITICAL":
            color = "#f48771"  # Rot
        elif level == "WARNING":
            color = "#dcdcaa"  # Gelb
        elif level == "INFO":
            color = "#4ec9b0"  # Cyan
        else:
            color = "#d4d4d4"  # Weiß (DEBUG)

        # HTML formatieren
        html = f'<span style="color: {color};">{message}</span>'

        # Einfügen
        cursor = self.console_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertHtml(html + "<br>")

        # Auto-Scroll
        if self.auto_scroll_btn.isChecked():
            scrollbar = self.console_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

        # Performance: Max. Zeilen begrenzen
        self._limit_lines()

    def _limit_lines(self):
        """Alte Zeilen löschen wenn zu viele."""
        document = self.console_text.document()
        if document.lineCount() > self.max_lines:
            cursor = QTextCursor(document)
            cursor.movePosition(QTextCursor.MoveOperation.Start)

            # Erste 100 Zeilen löschen
            for _ in range(100):
                cursor.select(QTextCursor.SelectionType.LineUnderCursor)
                cursor.removeSelectedText()
                cursor.deleteChar()  # Newline

    def clear_console(self):
        """Console leeren."""
        self.console_text.clear()

    def _toggle_auto_scroll(self, checked: bool):
        """Auto-Scroll umschalten."""
        self.auto_scroll_btn.setText(f"Auto-Scroll: {'ON' if checked else 'OFF'}")


class GUILogHandler(logging.Handler):
    """
    Custom Logging Handler der Logs an das Console Widget sendet.
    """

    def __init__(self, log_signal: LogSignal):
        super().__init__()
        self.log_signal = log_signal

    def emit(self, record: logging.LogRecord):
        """Log Record verarbeiten."""
        try:
            msg = self.format(record)
            level = record.levelname

            # Signal emittieren (Thread-safe)
            self.log_signal.new_log.emit(msg, level)

        except Exception:
            self.handleError(record)
