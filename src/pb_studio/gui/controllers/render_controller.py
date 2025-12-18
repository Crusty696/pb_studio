"""
Render Controller - Handles video rendering operations.

Extracted from MainWindow God Object (P1.6).
"""
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QFileDialog, QMessageBox

if TYPE_CHECKING:
    from ..main_window import MainWindow

logger = logging.getLogger(__name__)

from ..dialogs.render_settings_dialog import RenderSettingsDialog
from pb_studio.video.video_renderer import RenderSettings

# P3-FIX: Phase-Mapping Constants (consistent with MultiStageProgressDialog)
# These indices correspond to the phase order in MultiStageProgressDialog.phases[]
PHASE_AUDIO = 0  # 🎵 Audio Analysis (pre-render, not tracked in render worker)
PHASE_VIDEO = 1  # 🎬 Video Selection (pre-render, not tracked in render worker)
PHASE_PACING = 2  # ⚙️ Pacing Engine (Cut-List Generation, 0-20% of render progress)
PHASE_RENDER = 3  # 🎥 Video Rendering (Final MP4 creation, 20-100% of render progress)


class RenderController:
    """
    Manages video rendering operations.

    Responsibilities:
    - Start/stop rendering
    - Progress tracking and updates
    - Worker thread management
    - Result handling (success/failure)
    """

    def __init__(self, main_window: "MainWindow"):
        """
        Initialize render controller.

        Args:
            main_window: Reference to main window for UI access
        """
        self.main_window = main_window

    def start_render(self, output_path: str | None = None, duration: float | None = None):
        """
        Start rendering video in background thread.

        Args:
            output_path: Output file path (prompts if None)
            duration: Duration limit for preview (None = full video)
        """
        from ..main_window import RenderWorker
        from ..multi_stage_progress_dialog import MultiStageProgressDialog, PhaseStatus

        logger.info("Starting render")

        # DEFENSIVE: Reject boolean output_path (common Qt signal bug)
        if isinstance(output_path, bool):
            logger.warning(f"Ignoring invalid output_path (boolean: {output_path})")
            output_path = None

        # Validate prerequisites
        if not self.main_window.timeline_widget.audio_path:
            QMessageBox.warning(self.main_window, "No Audio", "Please load an audio file first.")
            return

        if len(self.main_window.clip_library_widget.clips) == 0:
            QMessageBox.warning(self.main_window, "No Clips", "Please import video clips first.")
            return

        if self.main_window.process_status_widget:
            self.main_window.process_status_widget.start_process(
                "render_video",
                message="Rendering video...",
                determinate=True,
            )

        # Check if already rendering
        if self.main_window.render_worker and self.main_window.render_worker.isRunning():
            QMessageBox.warning(self.main_window, "Rendering", "A render is already in progress.")
            return

        # Get output path if not provided
        if output_path is None:
            output_path, _ = QFileDialog.getSaveFileName(
                self.main_window, "Save Video", "", "Video Files (*.mp4)"
            )
            if not output_path or output_path == "":
                logger.info("Render cancelled by user (no output path)")
                return

        # Ensure output_path is a string
        output_path = str(output_path)

        # Show Render Settings Dialog
        settings_dialog = RenderSettingsDialog(self.main_window)
        if settings_dialog.exec() != QDialog.DialogCode.Accepted:
            logger.info("Render cancelled by user (settings dialog)")
            return

        render_settings = settings_dialog.get_settings()
        logger.info(
            f"Render Settings: GPU={render_settings.use_gpu}, Encoder={render_settings.gpu_encoder}, CRF={render_settings.crf}"
        )

        # Create NEW multi-stage progress dialog
        self.main_window.render_progress_dialog = MultiStageProgressDialog(
            title="Video Production Pipeline", parent=self.main_window
        )

        # Connect cancel signal
        self.main_window.render_progress_dialog.cancelled.connect(self._on_render_cancelled)

        # Initialize all phases as waiting
        # P3-FIX: Use explicit phase count for clarity
        NUM_PHASES = 4  # AUDIO, VIDEO, PACING, RENDER
        for i in range(NUM_PHASES):
            self.main_window.render_progress_dialog.update_stage(i, PhaseStatus.WAITING)

        # FIX: Audio & Video Analysen sind bereits abgeschlossen VOR dem Render!
        # Diese Phasen müssen sofort auf COMPLETED gesetzt werden, sonst bleiben sie bei 0%
        # Audio-Analyse wurde bereits beim Laden der Audio-Datei durchgeführt
        audio_name = "N/A"
        if self.main_window.timeline_widget and self.main_window.timeline_widget.audio_path:
            audio_name = self.main_window.timeline_widget.audio_path.name

        self.main_window.render_progress_dialog.update_stage(
            PHASE_AUDIO,
            PhaseStatus.COMPLETED,
            progress=100,
            metrics={"Status": "Bereits analysiert", "Datei": audio_name},
        )
        # Auch Timeline-Balken aktualisieren
        self.main_window.render_progress_dialog.set_phase_progress(PHASE_AUDIO, 100)

        # Video-Clips wurden bereits beim Import analysiert
        clip_count = (
            len(self.main_window.clip_library_widget.clips)
            if self.main_window.clip_library_widget
            else 0
        )
        self.main_window.render_progress_dialog.update_stage(
            PHASE_VIDEO,
            PhaseStatus.COMPLETED,
            progress=100,
            metrics={"Status": "Clips geladen", "Anzahl": f"{clip_count} Clips"},
        )
        # Auch Timeline-Balken aktualisieren
        self.main_window.render_progress_dialog.set_phase_progress(PHASE_VIDEO, 100)

        # Setze aktive Phase auf PACING (die erste echte Render-Phase)
        self.main_window.render_progress_dialog.set_current_phase(PHASE_PACING)
        self.main_window.render_progress_dialog.update_stage(
            PHASE_PACING,
            PhaseStatus.RUNNING,
            progress=0,
            metrics={"Status": "Starte Cut-List Generation..."},
        )

        self.main_window.render_progress_dialog.show()

        self.main_window.progress_bar.setVisible(True)
        self.main_window.progress_bar.setValue(0)
        self.main_window.render_started.emit()
        self.main_window.update_status("Starting render...")

        # Create and configure worker thread
        logger.info("Starting background render thread (with cut-list generation)")
        self.main_window.render_worker = RenderWorker(
            main_window=self.main_window,
            audio_path=str(self.main_window.timeline_widget.audio_path),
            output_path=output_path,
            duration=duration,
            output_path=output_path,
            duration=duration,
            parent=self.main_window,
            render_settings=render_settings,  # Pass settings
        )

        # Connect signals
        self.main_window.render_worker.progress_updated.connect(self.on_render_progress)
        self.main_window.render_worker.render_finished.connect(self.on_render_complete)
        self.main_window.render_worker.cutlist_generated.connect(self._on_cutlist_generated)

        # Start the worker
        self.main_window.render_worker.start()
        logger.info("Render worker thread started")

    def on_render_progress(self, progress_percent: int, message: str):
        """
        Handle render progress updates from worker thread.

        Args:
            progress_percent: Progress percentage (0-100)
            message: Progress message
        """
        from ..multi_stage_progress_dialog import PhaseStatus

        self.main_window.progress_bar.setValue(progress_percent)

        if self.main_window.process_status_widget:
            self.main_window.process_status_widget.update_process(
                "render_video",
                percent=progress_percent,
                message=message,
            )

        # BUG FIX 6: Check if dialog still exists before updating
        if self.main_window.render_progress_dialog:
            try:
                # P3-FIX: Determine current phase based on progress (consistent mapping)
                # 0-20%: Cut-list generation (PHASE_PACING)
                # 20-100%: Video rendering (PHASE_RENDER)
                PACING_THRESHOLD = 20  # P3-FIX: Extract magic number

                if progress_percent <= PACING_THRESHOLD:
                    # Pacing Engine phase (0-20% → 0-100% phase progress)
                    phase_progress = int((progress_percent / PACING_THRESHOLD) * 100)
                    self.main_window.render_progress_dialog.set_current_phase(PHASE_PACING)
                    self.main_window.render_progress_dialog.update_stage(
                        PHASE_PACING,
                        PhaseStatus.RUNNING,
                        progress=phase_progress,
                        metrics={"Schritt": "Cut-List Generation", "Status": message},
                    )
                else:
                    # Mark Pacing as completed
                    self.main_window.render_progress_dialog.update_stage(
                        PHASE_PACING, PhaseStatus.COMPLETED, progress=100
                    )

                    # Rendering phase (20-100% → 0-100% phase progress)
                    phase_progress = int(
                        ((progress_percent - PACING_THRESHOLD) / (100 - PACING_THRESHOLD)) * 100
                    )
                    self.main_window.render_progress_dialog.set_current_phase(PHASE_RENDER)
                    self.main_window.render_progress_dialog.update_stage(
                        PHASE_RENDER,
                        PhaseStatus.RUNNING,
                        progress=phase_progress,
                        metrics={
                            "Auflösung": "1080p",
                            "GPU": "AMD RX 7800 XT",
                            "Encoder": "h264_amf",
                            "Status": message,
                        },
                    )
            except RuntimeError as e:
                # BUG FIX 6: Dialog may have been deleted (user closed or error)
                logger.warning(f"Progress dialog no longer available: {e}")
                self.main_window.render_progress_dialog = None

    def on_render_complete(self, success: bool, result: str):
        """
        Handle render completion from worker thread.

        Args:
            success: Whether render was successful
            result: Output path on success, error message on failure
        """
        from ..multi_stage_progress_dialog import PhaseStatus

        # BUG FIX 2: Clean up worker with proper deletion
        if self.main_window.render_worker:
            # Disconnect all signals to prevent dangling connections
            try:
                self.main_window.render_worker.progress_updated.disconnect()
                self.main_window.render_worker.render_finished.disconnect()
                self.main_window.render_worker.cutlist_generated.disconnect()
            except TypeError:
                # Signals may already be disconnected
                pass

            # Wait for thread to finish
            self.main_window.render_worker.wait(5000)  # 5 second timeout

            # Schedule worker for deletion
            self.main_window.render_worker.deleteLater()
            self.main_window.render_worker = None

        # Update progress dialog before closing
        if self.main_window.render_progress_dialog:
            if success:
                # P3-FIX: Mark final phase as completed (using constant instead of magic number)
                self.main_window.render_progress_dialog.update_stage(
                    PHASE_RENDER, PhaseStatus.COMPLETED, progress=100
                )
                self.main_window.render_progress_dialog.finish(
                    success=True, message="Video erfolgreich erstellt!"
                )
            else:
                # P3-FIX: Mark as error (using constant instead of magic number)
                self.main_window.render_progress_dialog.update_stage(
                    PHASE_RENDER, PhaseStatus.ERROR, progress=0
                )
                self.main_window.render_progress_dialog.finish(
                    success=False, message="Fehler beim Rendering"
                )

            # Dialog closes automatically after finish()
            self.main_window.render_progress_dialog = None

        if self.main_window.process_status_widget:
            self.main_window.process_status_widget.finish_process(
                "render_video",
                success=success,
                message="Render complete" if success else "Render failed",
            )

        # Update UI
        self.main_window.progress_bar.setVisible(False)
        self.main_window.render_finished.emit()

        if success:
            logger.info(f"Render successful: {result}")
            self.main_window.update_status(f"Render complete: {Path(result).name}")

            # Load rendered video into preview
            if self.main_window.preview_widget:
                logger.info(f"Loading rendered video into preview: {result}")
                if self.main_window.preview_widget.load_video(result):
                    logger.info("Rendered video successfully loaded into preview")
                    self.main_window.update_status(
                        "Video ready for preview - use play button to watch"
                    )
                else:
                    logger.warning("Failed to load rendered video into preview widget")

            QMessageBox.information(
                self.main_window,
                "Render Complete",
                f"Video saved to:\n{result}\n\nYou can now preview it in the player above.",
            )
        else:
            logger.error(f"Render failed: {result}")
            self.main_window.update_status("Render failed")
            QMessageBox.critical(
                self.main_window, "Render Failed", f"Failed to render video:\n{result}"
            )

    def _on_cutlist_generated(self, cut_list):
        """
        Handle cut list generation from worker thread.

        Args:
            cut_list: List of CutListEntry objects
        """
        logger.info(f"Received cut list with {len(cut_list)} cuts - displaying on timeline")

        if self.main_window.timeline_widget:
            self.main_window.timeline_widget.display_cut_list(cut_list)
            logger.info("Cut list displayed on timeline")
        else:
            logger.warning("Timeline widget not available - cannot display cut list")

    def on_render_started(self):
        """Handle render started - disable UI during render."""
        self.main_window.play_button.setEnabled(False)

    def on_render_finished(self):
        """Handle render finished - re-enable UI."""
        self.main_window.play_button.setEnabled(True)

    def _on_render_cancelled(self):
        """Handle render cancellation from progress dialog."""
        logger.info("User cancelled render")

        # Stop worker if running
        if self.main_window.render_worker:
            self.main_window.render_worker.cancel()
            self.main_window.render_worker.wait()
            self.main_window.render_worker = None

        # Update UI
        self.main_window.progress_bar.setVisible(False)
        self.main_window.update_status("Render cancelled")
        self.main_window.render_finished.emit()

    def generate_preview(
        self,
        start_position: float | None = None,
        output_path: str | None = None,
        show_dialogs: bool = True,
    ):
        """
        Generate 90-second video preview with selectable start position.

        Args:
            start_position: Optional start position in seconds (for automated testing)
            output_path: Optional output file path (for automated testing)
            show_dialogs: Show interactive dialogs (default: True, False for automated tests)

        Shows dialog to select start position, then generates and displays preview.
        """
        from datetime import datetime

        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import (
            QDialog,
            QDialogButtonBox,
            QDoubleSpinBox,
            QLabel,
            QSlider,
            QVBoxLayout,
        )

        logger.info("Generating preview")

        # DEFENSIVE: Reject boolean parameters (common Qt signal bug)
        if isinstance(start_position, bool):
            logger.warning(f"Ignoring invalid start_position (boolean: {start_position})")
            start_position = None
        if isinstance(output_path, bool):
            logger.warning(f"Ignoring invalid output_path (boolean: {output_path})")
            output_path = None

        # Validate prerequisites
        if not self.main_window.timeline_widget or not self.main_window.timeline_widget.audio_path:
            if show_dialogs:
                QMessageBox.warning(
                    self.main_window, "No Audio", "Please load an audio file first."
                )
            else:
                logger.error("No audio file loaded")
            return

        if (
            not self.main_window.clip_library_widget
            or len(self.main_window.clip_library_widget.clips) == 0
        ):
            if show_dialogs:
                QMessageBox.warning(
                    self.main_window, "No Clips", "Please import video clips first."
                )
            else:
                logger.error("No video clips available")
            return

        # Get audio duration
        audio_path = self.main_window.timeline_widget.audio_path
        audio_duration = self.main_window.timeline_widget.duration or 0.0
        clips = self.main_window.clip_library_widget.clips

        if audio_duration < 90.0:
            if show_dialogs:
                QMessageBox.warning(
                    self.main_window,
                    "Audio Too Short",
                    f"Audio is only {audio_duration:.1f}s. Need at least 90s for preview.",
                )
            else:
                logger.error(f"Audio too short: {audio_duration:.1f}s < 90s")
            return

        # Headless mode (automated testing) - skip dialogs
        if not show_dialogs:
            if start_position is None:
                # Default: Random position in middle third of audio
                start_position = audio_duration / 3.0
                logger.info(f"Using default start position: {start_position:.2f}s")
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"preview_{timestamp}.mp4"
                logger.info(f"Using default output path: {output_path}")

            # Skip to preview generation
            self._start_preview_generation(
                audio_path, clips, start_position, output_path, show_dialogs=False
            )
            return

        # Interactive mode - show position selection dialog
        dialog = QDialog(self.main_window)
        dialog.setWindowTitle("Preview Start Marker (Marker auf der Timeline)")
        dialog.setMinimumWidth(500)

        layout = QVBoxLayout()

        # Info label
        info_label = QLabel(
            f"<b>Generiere 90-Sekunden Preview</b><br><br>"
            f"<b>Audio-Dauer:</b> {audio_duration:.1f} Sekunden<br>"
            f"<b>Aufgabe:</b> Setze einen Marker auf der Timeline wo der Preview starten soll.<br>"
            f"Der Marker zeigt wo die 90-Sekunden-Preview BEGINNT.<br><br>"
            f"<small><i>Hinweis: In Zukunft kannst du direkt auf der Timeline klicken um den Marker zu setzen.</i></small>"
        )
        layout.addWidget(info_label)

        # Start position selector with slider
        position_label = QLabel("Preview Start Position (Marker Platzierung):")
        layout.addWidget(position_label)

        # Horizontal slider for visual feedback
        position_slider = QSlider(Qt.Orientation.Horizontal)
        position_slider.setMinimum(0)
        position_slider.setMaximum(int(max(0.0, audio_duration - 90.0) * 10))
        position_slider.setValue(0)
        position_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        position_slider.setTickInterval(max(1, int((audio_duration - 90.0) / 10)))
        layout.addWidget(position_slider)

        # Numeric input for precise positioning
        position_spin = QDoubleSpinBox()
        position_spin.setMinimum(0.0)
        position_spin.setMaximum(max(0.0, audio_duration - 90.0))
        position_spin.setValue(0.0)
        position_spin.setDecimals(2)
        position_spin.setSuffix(" s")
        position_spin.setSingleStep(1.0)

        # Connect slider and spinbox
        position_slider.valueChanged.connect(lambda v: position_spin.setValue(v / 10.0))
        position_spin.valueChanged.connect(lambda v: position_slider.setValue(int(v * 10)))

        layout.addWidget(position_spin)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        dialog.setLayout(layout)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            logger.info("Preview generation cancelled by user")
            return

        start_position = position_spin.value()
        logger.info(f"Generating preview starting at {start_position:.2f}s")

        # Get output path
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"preview_{timestamp}.mp4"

        output_path, _ = QFileDialog.getSaveFileName(
            self.main_window, "Save Preview", default_name, "Video Files (*.mp4)"
        )

        if not output_path:
            logger.info("Preview generation cancelled (no output path)")
            return

        # Start preview generation
        self._start_preview_generation(
            audio_path, clips, start_position, output_path, show_dialogs=True
        )

    def _start_preview_generation(
        self,
        audio_path: str,
        clips: list,
        start_position: float,
        output_path: str,
        show_dialogs: bool = True,
    ):
        """Internal method to start preview generation with given parameters."""
        from PyQt6.QtCore import QThread, pyqtSignal

        from ..progress_dialog import ProgressDialog

        # Create progress dialog (only if show_dialogs=True)
        if show_dialogs:
            self.main_window.render_progress_dialog = ProgressDialog(
                title="Generating Preview",
                message=f"Creating 90s preview from {start_position:.1f}s...",
                parent=self.main_window,
                indeterminate=False,
                show_cancel=False,
            )
            self.main_window.render_progress_dialog.show()
        else:
            logger.info(f"Generating preview (headless): {start_position:.1f}s -> {output_path}")

        self.main_window.update_status("Generating preview...")

        # Define PreviewWorker as nested class
        class PreviewWorker(QThread):
            """Background worker for preview generation."""

            progress_updated = pyqtSignal(int, str)
            preview_finished = pyqtSignal(bool, str)

            def __init__(self, main_window, audio_path, clips, start_pos, output_path):
                super().__init__()
                self.main_window = main_window
                self.audio_path = audio_path
                self.clips = clips
                self.start_position = start_pos
                self.output_path = output_path

            def run(self):
                try:
                    from pb_studio.video.preview_renderer import PreviewRenderer

                    renderer = PreviewRenderer()

                    def progress_callback(progress: float):
                        percent = int(progress * 100)
                        self.progress_updated.emit(percent, f"Rendering preview... {percent}%")

                    # Check if stems are available from main window
                    use_stems = getattr(self.main_window, "stems_available", False)

                    result = renderer.generate_preview(
                        audio_path=self.audio_path,
                        clips=self.clips,
                        start_position=self.start_position,
                        duration=90.0,
                        output_path=self.output_path,
                        progress_callback=progress_callback,
                        use_stems=use_stems,
                    )

                    if result:
                        self.preview_finished.emit(True, str(result))
                    else:
                        self.preview_finished.emit(False, "Preview generation failed")

                except Exception as e:
                    logger.error(f"Preview generation error: {e}", exc_info=True)
                    self.preview_finished.emit(False, str(e))

        # Create and start worker
        self.main_window.preview_worker = PreviewWorker(
            self.main_window,
            str(self.main_window.timeline_widget.audio_path),
            self.main_window.clip_library_widget.clips,
            start_position,
            output_path,
        )

        if self.main_window.process_status_widget:
            self.main_window.process_status_widget.start_process(
                "preview_generation",
                message="Generating preview...",
                determinate=True,
            )

        self.main_window.preview_worker.progress_updated.connect(self.on_preview_progress)
        self.main_window.preview_worker.preview_finished.connect(self.on_preview_complete)
        self.main_window.preview_worker.start()

    def on_preview_progress(self, progress_percent: int, message: str):
        """Handle preview generation progress updates."""
        if self.main_window.render_progress_dialog:
            self.main_window.render_progress_dialog.update_progress(progress_percent, message)
        if self.main_window.process_status_widget:
            self.main_window.process_status_widget.update_process(
                "preview_generation",
                percent=progress_percent,
                message=message,
            )

    def on_preview_complete(self, success: bool, result: str):
        """Handle preview generation completion."""
        # Clean up worker
        if hasattr(self.main_window, "preview_worker") and self.main_window.preview_worker:
            self.main_window.preview_worker.wait()
            self.main_window.preview_worker = None

        if self.main_window.process_status_widget:
            self.main_window.process_status_widget.finish_process(
                "preview_generation",
                success=success,
                message="Preview ready" if success else "Preview failed",
            )

        # Close progress dialog
        if self.main_window.render_progress_dialog:
            self.main_window.render_progress_dialog.close()
            self.main_window.render_progress_dialog = None

        # Emit render_finished signal for automated tests
        self.main_window.render_finished.emit()

        if success:
            logger.info(f"Preview generated: {result}")
            self.main_window.update_status("Preview ready")

            # Load preview into PreviewWidget
            if self.main_window.preview_widget:
                if self.main_window.preview_widget.load_video(result):
                    logger.info("Preview loaded into widget")
                else:
                    logger.warning("Failed to load preview into widget")

            # Show success message
            QMessageBox.information(
                self.main_window,
                "Preview Ready",
                f"Preview saved to:\n{result}\n\nPreview is now loaded in the player above.",
                QMessageBox.StandardButton.Ok,
            )
        else:
            logger.error(f"Preview generation failed: {result}")
            self.main_window.update_status("Preview failed")
            QMessageBox.critical(
                self.main_window, "Preview Failed", f"Failed to generate preview:\n{result}"
            )
