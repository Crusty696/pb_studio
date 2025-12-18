"""
PB_studio Application Entry Point

Starts the PB_studio GUI application.

Usage:
    python -m pb_studio
    OR
    python src/pb_studio/__main__.py

Author: PB_studio Development Team
"""

import sys
from pathlib import Path

# Ensure proper imports work
if __name__ == "__main__":
    # Add src to path for development
    src_path = Path(__file__).parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def main():
    """Main application entry point."""
    from PyQt6.QtWidgets import QApplication

    from pb_studio.gui.main_window import MainWindow

    # Create QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("PB_studio")
    app.setOrganizationName("PB_studio")

    # Create and show main window
    window = MainWindow()
    window.show()

    # Start event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
