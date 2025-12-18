"""
Dear PyGui Bridge für PB_studio

Integriert Dear PyGui Timeline und Preview parallel zur PyQt6 GUI.

Dear PyGui läuft in separaten Windows die neben dem PyQt6 MainWindow geöffnet werden.
Die Kommunikation erfolgt über Callbacks und shared state.

Features:
- Timeline Window (Dear PyGui) mit Waveform und Beat-Grid
- Preview Window (Dear PyGui) mit Echtzeit-Video-Preview
- Bi-direktionale Kommunikation mit PyQt6 MainWindow
- Shared Playback State

Usage:
    from pb_studio.gui.dearpygui_bridge import DearPyGuiBridge

    bridge = DearPyGuiBridge()
    bridge.show_timeline()
    bridge.show_preview()

    # In PyQt6 MainWindow:
    bridge.load_audio("track.mp3")
    bridge.load_cut_list(cut_list, video_clips)
    bridge.set_position(10.5)  # Sync playback position

Author: PB_studio Development Team
Task: A3 - Dear PyGui Integration
"""

import logging
from collections.abc import Callable
from pathlib import Path
from threading import Lock, Thread

import dearpygui.dearpygui as dpg

from ..gui.preview_logic import PlaybackState, PreviewLogic
from ..gui.widgets.timeline_widget import TimelineWidget as DPGTimelineWidget
from ..pacing import CutListEntry

logger = logging.getLogger(__name__)


class DearPyGuiBridge:
    """
    Bridge zwischen PyQt6 MainWindow und Dear PyGui Timeline/Preview.

    Startet Dear PyGui in separatem Thread und ermöglicht Kommunikation
    mit PyQt6 über Callbacks.
    """

    def __init__(
        self,
        on_position_changed: Callable[[float], None] | None = None,
        on_clip_selected: Callable[[int], None] | None = None,
        on_close: Callable[[], None] | None = None,
    ):
        """
        Initialize Dear PyGui Bridge.

        Args:
            on_position_changed: Callback wenn Timeline-Position ändert (seconds)
            on_clip_selected: Callback wenn Clip ausgewählt wird (clip_id)
            on_close: Callback wenn DearPyGui Window geschlossen wird
        """
        self.on_position_changed = on_position_changed
        self.on_clip_selected = on_clip_selected
        self.on_close = on_close

        # Dear PyGui state
        self.dpg_initialized = False
        self.dpg_thread: Thread | None = None
        self._is_shutting_down = False
        self._lock = Lock()

        # Widgets
        self.timeline: DPGTimelineWidget | None = None
        self.preview: PreviewLogic | None = None

        # Shared state
        self.current_position: float = 0.0
        self.is_playing: bool = False
        self.audio_path: Path | None = None
        self.cut_list: list[CutListEntry] = []
        self.video_clips: dict[str, Path] = {}
        self._timeline_markers: list[int] = []  # Tracking DPG timeline marker IDs

        logger.info("DearPyGuiBridge initialized")

    def initialize(self) -> bool:
        """
        Initialize Dear PyGui context.

        Returns:
            True if successful
        """
        if self.dpg_initialized:
            logger.warning("Dear PyGui already initialized")
            return True

        try:
            dpg.create_context()
            self.dpg_initialized = True
            logger.info("Dear PyGui context created")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Dear PyGui: {e}", exc_info=True)
            return False

    def show_timeline(self, width: int = 1200, height: int = 400) -> bool:
        """
        Show Dear PyGui Timeline Window.

        Args:
            width: Window width
            height: Window height

        Returns:
            True if successful
        """
        if not self.dpg_initialized:
            if not self.initialize():
                return False

        try:
            # Create timeline widget
            self.timeline = DPGTimelineWidget(
                width=width,
                height=height,
                on_clip_selected=self._on_timeline_clip_selected,
                on_timeline_changed=self._on_timeline_changed,
            )

            # Create timeline window
            self.timeline.create_window()

            logger.info("Dear PyGui Timeline window created")

            # Start Dear PyGui render loop in separate thread
            if not self.dpg_thread or not self.dpg_thread.is_alive():
                with self._lock:
                    if not self.dpg_thread or not self.dpg_thread.is_alive():
                        self._start_render_loop()

            return True

        except Exception as e:
            logger.error(f"Failed to show timeline: {e}", exc_info=True)
            return False

    def show_preview(self, width: int = 1920, height: int = 1080) -> bool:
        """
        Show Dear PyGui Preview Window.

        Args:
            width: Preview width
            height: Preview height

        Returns:
            True if successful
        """
        if not self.dpg_initialized:
            if not self.initialize():
                return False

        try:
            # Create preview logic
            self.preview = PreviewLogic(
                width=width,
                height=height,
                on_playback_state_changed=self._on_playback_state_changed,
                on_time_update=self._on_preview_time_update,
            )

            # Create preview window
            with dpg.window(
                label="Preview", width=width + 40, height=height + 100, pos=[1250, 20]
            ) as preview_window:
                # Toolbar
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="Play", callback=self._on_preview_play, tag="preview_play_btn"
                    )
                    dpg.add_button(
                        label="Pause", callback=self._on_preview_pause, tag="preview_pause_btn"
                    )
                    dpg.add_button(
                        label="Stop", callback=self._on_preview_stop, tag="preview_stop_btn"
                    )
                    dpg.add_text("00:00.000", tag="preview_time_label")

                # Texture for video preview
                self.preview.create_dpg_texture("preview_texture")
                dpg.add_image("preview_texture")

            logger.info("Dear PyGui Preview window created")

            # Start render loop if not running
            if not self.dpg_thread or not self.dpg_thread.is_alive():
                self._start_render_loop()

            return True

        except Exception as e:
            logger.error(f"Failed to show preview: {e}", exc_info=True)
            return False

    def _start_render_loop(self):
        """Start Dear PyGui render loop in separate thread."""

        def render_loop():
            try:
                dpg.create_viewport(title="PB_studio - Dear PyGui", width=1200, height=800)
                dpg.setup_dearpygui()
                dpg.show_viewport()

                # Register viewport close callback
                dpg.set_viewport_close_callback(self._on_window_close)

                # Main render loop
                while dpg.is_dearpygui_running() and not self._is_shutting_down:
                    # Update preview if playing
                    if self.preview and self.is_playing:
                        self.preview.update()

                    dpg.render_dearpygui_frame()

                # Cleanup nach render loop
                self._cleanup_resources()
                logger.info("Dear PyGui render loop ended")

            except Exception as e:
                logger.error(f"Dear PyGui render loop error: {e}", exc_info=True)
                self._cleanup_resources()

        self.dpg_thread = Thread(target=render_loop, daemon=True)
        self.dpg_thread.start()
        logger.info("Dear PyGui render loop started in separate thread")

    # === Public API for PyQt6 integration ===

    def load_audio(self, audio_path: str | Path, bpm: float | None = None) -> bool:
        """
        Load audio track into timeline.

        Args:
            audio_path: Path to audio file
            bpm: Optional manual BPM override

        Returns:
            True if successful
        """
        self.audio_path = Path(audio_path)

        with self._lock:
            if self.timeline:
                success = self.timeline.load_audio_track(audio_path, bpm=bpm)
                if success:
                    logger.info(f"Audio loaded into Dear PyGui Timeline: {self.audio_path.name}")
                return success
            else:
                logger.warning(
                    "Timeline not initialized - audio will be loaded when timeline is shown"
                )
                return False

    def load_cut_list(
        self, cut_list: list[CutListEntry], video_clips: dict[str, str | Path]
    ) -> bool:
        """
        Load cut list and video clips for preview.

        Args:
            cut_list: List of CutListEntry objects
            cut_list: Dict mapping clip_id to video file path

        Returns:
            True if successful
        """
        self.cut_list = cut_list
        self.video_clips = {k: Path(v) for k, v in video_clips.items()}

        # Load into timeline (as visualization)
        if self.timeline:
            self.timeline.set_pacing_blueprint(cut_list)

        # Load into preview (for playback)
        if self.preview:
            self.preview.load_cut_list(cut_list, self.video_clips)

            if self.audio_path:
                self.preview.load_audio_track(self.audio_path)

            logger.info(f"Cut list loaded: {len(cut_list)} cuts")
            return True
        else:
            logger.warning(
                "Preview not initialized - cut list will be loaded when preview is shown"
            )
            return False

    def set_position(self, position: float):
        """
        Set playback position (sync from PyQt6).

        Args:
            position: Time in seconds
        """
        self.current_position = position

        if self.preview:
            self.preview.seek(position)

    def play(self):
        """Start playback."""
        self.is_playing = True

        if self.preview:
            self.preview.play()

    def pause(self):
        """Pause playback."""
        self.is_playing = False

        if self.preview:
            self.preview.pause()

    def stop(self):
        """Stop playback."""
        self.is_playing = False
        self.current_position = 0.0

        if self.preview:
            self.preview.stop()

    def sync_cutlist_to_dpg(self, cutlist: list[dict]):
        """
        Synchronisiere Cut-List von PyQt zu DearPyGui.

        Args:
            cutlist: Liste mit Cut-Informationen (dicts mit start_time, id, color, etc.)
        """
        if not self.dpg_initialized or not self.timeline:
            logger.debug("Cannot sync cutlist - DearPyGui not ready")
            return

        try:
            # Entferne alte Marker
            for marker_id in self._timeline_markers:
                try:
                    if dpg.does_item_exist(marker_id):
                        dpg.delete_item(marker_id)
                except Exception as e:
                    logger.debug(f"Error removing marker {marker_id}: {e}")

            self._timeline_markers.clear()

            # Füge neue Marker hinzu (falls Timeline plotbar unterstützt)
            # Alternativ: Update Timeline-Widget direkt
            if hasattr(self.timeline, "update_cut_markers"):
                self.timeline.update_cut_markers(cutlist)
                logger.info(f"Synced {len(cutlist)} cuts to DearPyGui timeline")
            else:
                # Fallback: Store in internal state
                logger.debug(
                    f"Timeline widget doesn't support markers, storing {len(cutlist)} cuts"
                )

        except Exception as e:
            logger.warning(f"Error syncing cutlist to DearPyGui: {e}", exc_info=True)

    def on_cutlist_changed(self, cutlist: list[dict]):
        """
        Callback wenn Cut-List in PyQt geändert wurde.

        Args:
            cutlist: Aktualisierte Cut-List
        """
        logger.info(f"Cut-list changed: {len(cutlist)} cuts")
        self.sync_cutlist_to_dpg(cutlist)

    def register_clip_selection_callback(self, callback: Callable):
        """
        Registriere Callback für Clip-Selection.

        Args:
            callback: Funktion die bei Clip-Selection aufgerufen wird (clip_id)
        """
        self.on_clip_selected = callback
        logger.info("Clip selection callback registered")

    # === Internal Callbacks ===

    def _handle_clip_selection(self, sender, app_data, user_data):
        """
        Handle Clip-Selection im DearPyGui Timeline.

        Args:
            sender: DearPyGui sender widget
            app_data: DearPyGui app data
            user_data: Custom user data mit clip_id
        """
        if not user_data:
            logger.debug("Clip selection without user_data")
            return

        clip_id = user_data.get("clip_id")
        if clip_id is not None:
            logger.debug(f"Clip selected via handler: {clip_id}")

            # Sync to preview if available
            if self.preview and hasattr(self.preview, "select_clip"):
                self.preview.select_clip(clip_id)

            # Notify PyQt6
            if self.on_clip_selected:
                self.on_clip_selected(clip_id)
        else:
            logger.debug("No clip_id in user_data")

    def _on_timeline_clip_selected(self, clip_id: int | str):
        """Handle clip selection in timeline (legacy callback)."""
        logger.debug(f"Clip selected in timeline: {clip_id}")

        if self.on_clip_selected:
            self.on_clip_selected(clip_id)

    def _on_timeline_changed(self):
        """Handle timeline modifications."""
        logger.debug("Timeline changed")

    def _on_playback_state_changed(self, state: PlaybackState):
        """Handle playback state changes from preview."""
        logger.debug(f"Playback state changed: {state.value}")

        self.is_playing = state == PlaybackState.PLAYING

    def _on_preview_time_update(self, time: float):
        """Handle time updates from preview."""
        self.current_position = time

        # Update timeline playhead if exists
        # (Timeline doesn't have playhead in current DPG version, but could be added)

        # Notify PyQt6
        if self.on_position_changed:
            self.on_position_changed(time)

        # Update preview time label
        if dpg.does_item_exist("preview_time_label"):
            minutes = int(time // 60)
            seconds = time % 60
            time_str = f"{minutes:02d}:{seconds:06.3f}"
            dpg.set_value("preview_time_label", time_str)

    def _on_preview_play(self, sender, app_data, user_data):
        """Handle preview play button."""
        self.play()

    def _on_preview_pause(self, sender, app_data, user_data):
        """Handle preview pause button."""
        self.pause()

    def _on_preview_stop(self, sender, app_data, user_data):
        """Handle preview stop button."""
        self.stop()

    def _on_window_close(self):
        """Handle DearPyGui Window Close."""
        logger.info("DearPyGui window closing")

        # Signal shutdown to render loop
        self._is_shutting_down = True

        # Notify parent application
        if self.on_close:
            try:
                self.on_close()
            except Exception as e:
                logger.error(f"Error in on_close callback: {e}", exc_info=True)

    def _cleanup_resources(self):
        """Cleanup beim Schließen."""
        logger.info("Cleaning up DearPyGui resources")

        try:
            # Stop playback
            self.is_playing = False

            # Cleanup preview resources
            if self.preview:
                try:
                    self.preview.cleanup()
                    logger.debug("Preview resources cleaned")
                except Exception as e:
                    logger.warning(f"Error cleaning preview: {e}")

            # Remove timeline markers
            for marker_id in self._timeline_markers:
                try:
                    if dpg.does_item_exist(marker_id):
                        dpg.delete_item(marker_id)
                except Exception as e:
                    logger.debug(f"Error removing marker {marker_id}: {e}")

            self._timeline_markers.clear()

            # Destroy DearPyGui context
            with self._lock:
                if self.dpg_initialized:
                    try:
                        dpg.destroy_context()
                        self.dpg_initialized = False
                        logger.info("DearPyGui context destroyed")
                    except Exception as e:
                        logger.warning(f"Error destroying context: {e}")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)

    def shutdown(self):
        """
        Graceful shutdown der DearPyGui Bridge.

        Stoppt Playback, beendet Render-Loop und räumt alle Ressourcen auf.
        """
        logger.info("Shutting down DearPyGui Bridge")

        # Signal shutdown
        self._is_shutting_down = True

        try:
            # Stop render loop
            if dpg.is_dearpygui_running():
                dpg.stop_dearpygui()
                logger.debug("DearPyGui render loop stopped")

            # Wait for thread to finish (max 2 seconds)
            if self.dpg_thread and self.dpg_thread.is_alive():
                self.dpg_thread.join(timeout=2.0)
                if self.dpg_thread.is_alive():
                    logger.warning("DearPyGui thread did not finish in time")

        except Exception as e:
            logger.warning(f"DearPyGui shutdown error: {e}", exc_info=True)

        # Cleanup is called by render loop, but call again to be sure
        if self.dpg_initialized:
            self._cleanup_resources()

        logger.info("DearPyGui Bridge shutdown complete")

    def cleanup(self):
        """Clean up Dear PyGui resources (legacy method, calls shutdown)."""
        self.shutdown()
