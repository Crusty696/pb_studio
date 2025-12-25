"""
PB_studio Application Entry Point

Main entry point for PB_studio - Precision Beat Video Studio.
Initializes logging, configuration, and launches the PyQt6 GUI.

Usage:
    python main.py

Author: PB_studio Development Team
"""

import logging
import os
import sys
import faulthandler
from pathlib import Path

# Enable faulthandler to dump traceback on hard crash
fault_log = open("fault_dump.log", "w")
faulthandler.enable(file=fault_log)

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Workaround für ältere GPUs: NVFuser deaktivieren, um DLL-Load-Fehler (nvfuser_codegen.dll) zu vermeiden
os.environ.setdefault("PYTORCH_JIT_USE_NNC_NOT_NVFUSER", "1")
os.environ.setdefault("PYTORCH_NVFUSER_DISABLE", "1")
os.environ.setdefault("TORCH_NVFUSER_DISABLE", "1")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from pb_studio.core.config import load_config
from pb_studio.gui.main_window import MainWindow
from pb_studio.utils.logger import get_logger, setup_logging


def main():
    """
    Main application entry point.

    Run Sequence:
    1. Hardware Bootstrapper (Critical)
    2. Startup Cleanup
    3. Logging Setup
    4. Config Loading
    5. GUI Launch

    Returns:
        Exit code (0 = success)
    """
    # 1. Run Hardware Bootstrapper
    try:
        from pb_studio.bootstrapper import Bootstrapper
        boot = Bootstrapper()
        if not boot.run():
            print("CRITICAL: Bootstrapper validation failed.")
            sys.exit(1)
    except Exception as e:
        print(f"CRITICAL: Bootstrapper failed unexpectedly: {e}")
        # Continue with caution or exit? 
        # For now, we print and continue, relying on fallback logic.
        

    # Setup logging
    # Note: Cleanup runs BEFORE logging to ensure logs are cleared if requested
    # and to provide a truly fresh state.
    try:
        from pb_studio.utils.cleanup import perform_startup_cleanup
        # Assuming run from project root: c:\GEMINI_PROJEKTE\_Pb-studio_V_2\pb_studio\main.py
        # root is the directory containing main.py
        project_root = Path(__file__).parent
        perform_startup_cleanup(project_root)
    except Exception as e:
        print(f"Startup cleanup failed: {e}")

    setup_logging(console_level=logging.INFO)
    logger = get_logger(__name__)

    logger.info("=" * 80)
    logger.info("PB_studio - Precision Beat Video Studio")
    logger.info("=" * 80)

    # Load configuration
    try:
        config = load_config()
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1

    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("PB_studio")
    app.setOrganizationName("PB_studio Development Team")
    app.setApplicationVersion("0.1.0")

    # Set application-wide style
    app.setStyle("Fusion")  # Use Fusion style for consistent cross-platform appearance

    # Apply dark theme
    apply_dark_theme(app)

    logger.info("Qt application initialized")

    # Create and show main window
    try:
        main_window = MainWindow()
        main_window.show()
        logger.info("Main window displayed")
    except Exception as e:
        logger.error(f"Failed to create main window: {e}")
        logger.exception(e)
        return 1

    # Run application event loop
    logger.info("Starting Qt event loop")
    exit_code = app.exec()

    logger.info(f"Application exited with code {exit_code}")
    return exit_code


def apply_dark_theme(app: QApplication):
    """
    Apply dark theme to the application.

    Args:
        app: QApplication instance
    """
    # Dark color palette
    from PyQt6.QtGui import QColor, QPalette

    palette = QPalette()

    # Window colors
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)

    # Base colors
    palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))

    # Tooltip colors
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)

    # Text colors
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)

    # Bright text
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)

    # Link colors
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(35, 35, 35))

    # Disabled colors
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(127, 127, 127))
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(127, 127, 127)
    )
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(127, 127, 127))
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(127, 127, 127)
    )

    app.setPalette(palette)

    # Additional stylesheet for specific widgets
    app.setStyleSheet(
        """
        QToolTip {
            color: #ffffff;
            background-color: #2a2a2a;
            border: 1px solid #555555;
            padding: 5px;
        }

        QMenuBar {
            background-color: #2b2b2b;
            color: #ffffff;
        }

        QMenuBar::item {
            background-color: transparent;
            padding: 4px 8px;
        }

        QMenuBar::item:selected {
            background-color: #3a3a3a;
        }

        QMenu {
            background-color: #2b2b2b;
            color: #ffffff;
            border: 1px solid #555555;
        }

        QMenu::item:selected {
            background-color: #3a3a3a;
        }

        QToolBar {
            background-color: #2b2b2b;
            border: none;
            padding: 4px;
            spacing: 4px;
        }

        QPushButton {
            background-color: #3a3a3a;
            border: 1px solid #555555;
            padding: 5px 15px;
            border-radius: 3px;
        }

        QPushButton:hover {
            background-color: #4a4a4a;
        }

        QPushButton:pressed {
            background-color: #2a2a2a;
        }

        QDockWidget {
            color: #ffffff;
            titlebar-close-icon: url(close.png);
            titlebar-normal-icon: url(float.png);
        }

        QDockWidget::title {
            background-color: #3a3a3a;
            padding: 4px;
        }

        QStatusBar {
            background-color: #2b2b2b;
            color: #ffffff;
        }

        QProgressBar {
            border: 1px solid #555555;
            border-radius: 3px;
            background-color: #2b2b2b;
            text-align: center;
            color: #ffffff;
        }

        QProgressBar::chunk {
            background-color: #2a82da;
        }
    """
    )


if __name__ == "__main__":
    sys.exit(main())
