"""
File Controller for PB_studio.

Handles file import/export operations for audio, video, and Rekordbox XML.
Extracted from MainWindow to reduce God Object pattern.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QFileDialog, QMessageBox

if TYPE_CHECKING:
    from ..main_window import MainWindow

logger = logging.getLogger(__name__)


class FileController:
    """
    Controller for file import/export operations.

    Handles:
    - Audio file import with analysis
    - Video file import to clip library
    - Rekordbox XML import
    - Video export
    """

    def __init__(self, main_window: "MainWindow"):
        """
        Initialize FileController.

        Args:
            main_window: Reference to MainWindow for widget access
        """
        self.main_window = main_window

    def import_audio(self) -> str | None:
        """
        Import audio file and trigger analysis.

        Returns:
            Path to imported file or None if cancelled
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window, "Import Audio", "", "Audio Files (*.mp3 *.wav *.flac *.ogg *.m4a)"
        )

        if not file_path:
            return None

        logger.info(f"Importing audio: {file_path}")
        self.main_window.update_status(f"Loading audio: {Path(file_path).name}...")

        # Load audio into timeline
        if self.main_window.timeline_widget:
            try:
                self.main_window.timeline_widget.load_audio(file_path)
                self.main_window.update_status(f"Imported audio: {Path(file_path).name}")
                logger.info("Audio loaded into timeline successfully")

                # Analyze audio and update Pacing Dashboard
                self.main_window._analyze_and_update_audio_info(file_path)
                return file_path

            except Exception as e:
                logger.error(f"Failed to load audio into timeline: {e}")
                self.main_window.update_status(f"Failed to load audio: {e}")
                QMessageBox.critical(
                    self.main_window, "Audio Load Error", f"Failed to load audio file:\n{e}"
                )
                return None

        return None

    def import_video(self) -> str | None:
        """
        Import video file to clip library.

        Returns:
            Path to imported file or None if cancelled
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window, "Import Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv *.webm)"
        )

        if not file_path:
            return None

        logger.info(f"Importing video: {file_path}")
        self.main_window.update_status(f"Importing video: {Path(file_path).name}...")

        # Import video using ClipLibraryWidget's programmatic import
        if self.main_window.clip_library_widget:
            try:
                self.main_window.clip_library_widget.import_clips_programmatic([file_path])
                self.main_window.update_status(f"Imported video: {Path(file_path).name}")
                logger.info(f"Video imported successfully: {file_path}")
                return file_path
            except Exception as e:
                logger.error(f"Failed to import video: {e}")
                self.main_window.update_status(f"Failed to import video: {e}")
                QMessageBox.critical(
                    self.main_window, "Video Import Error", f"Failed to import video file:\n{e}"
                )
                return None
        else:
            logger.error("ClipLibraryWidget not available for video import")
            QMessageBox.warning(
                self.main_window,
                "Import Error",
                "Clip Library not initialized. Cannot import video.",
            )
            return None

    def import_rekordbox(self) -> int:
        """
        Import Rekordbox XML library.

        Returns:
            Number of tracks imported, or 0 if cancelled/failed
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window, "Import Rekordbox XML", "", "XML Files (*.xml)"
        )

        if not file_path:
            return 0

        if not self.main_window.current_project_id:
            QMessageBox.warning(
                self.main_window, "No Project", "Please create or open a project first."
            )
            return 0

        try:
            from ...importers.rekordbox_importer import RekordboxImporter

            logger.info(f"Importing Rekordbox XML: {file_path}")
            self.main_window.update_status("Parsing Rekordbox XML...")

            importer = RekordboxImporter(file_path)
            if importer.parse():
                # Show import dialog with playlist selection
                count = importer.import_to_database(self.main_window.current_project_id)

                self.main_window.update_status(f"Imported {count} tracks from Rekordbox")
                QMessageBox.information(
                    self.main_window,
                    "Import Complete",
                    f"Successfully imported {count} tracks from Rekordbox XML.\n\n"
                    f"Total tracks found: {len(importer.tracks)}\n"
                    f"Playlists found: {len(importer.playlists)}",
                )
                logger.info(f"Rekordbox import complete: {count} tracks")
                return count
            else:
                QMessageBox.critical(
                    self.main_window,
                    "Import Failed",
                    "Failed to parse Rekordbox XML file.\nCheck the log for details.",
                )
                logger.error("Rekordbox XML parse failed")
                return 0

        except Exception as e:
            logger.error(f"Rekordbox import error: {e}", exc_info=True)
            QMessageBox.critical(self.main_window, "Import Error", f"Rekordbox import failed:\n{e}")
            return 0

    def export_video(self) -> str | None:
        """
        Export final video with file dialog.

        Returns:
            Path to export file or None if cancelled
        """
        file_path, _ = QFileDialog.getSaveFileName(
            self.main_window, "Export Video", "", "Video Files (*.mp4)"
        )

        if file_path:
            logger.info(f"Exporting video to: {file_path}")
            self.main_window.start_render(output_path=file_path)
            return file_path

        return None
