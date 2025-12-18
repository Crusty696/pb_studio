"""
Interactive Timeline Widget for PB_studio

High-performance timeline visualization using Dear PyGui with beat-synced editing.

Features:
- Audio waveform visualization with beat markers
- Drag-and-drop clip placement from ClipLibraryWidget
- Cut point editing with snap-to-beat
- Pacing blueprint visualization
- Real-time preview scrubbing
- Energy curve overlay

Usage:
    from pb_studio.gui.widgets import TimelineWidget

    timeline = TimelineWidget()
    timeline.load_audio_track(audio_file, bpm=140.0)
    timeline.add_clip(clip_data, start_time=0.0, duration=2.0)
    timeline.set_pacing_blueprint(blueprint)

Dependencies:
- Dear PyGui (dearpygui)
- AudioAnalyzer (Tasks 18-21)
- PacingEngine (Tasks 25-29)
"""

import time
from collections.abc import Callable
from enum import Enum
from pathlib import Path

import dearpygui.dearpygui as dpg
import numpy as np

from ...audio import AudioAnalyzer
from ...pacing import BeatGridInfo, CutListEntry, PacingEngine
from ...utils.logger import get_logger
from ...video import VideoManager

logger = get_logger(__name__)


class PlaybackState(Enum):
    """Playback state machine."""

    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"


class TimelineClip:
    """Represents a video clip on the timeline."""

    def __init__(
        self,
        clip_id: int | str,
        file_path: str,
        start_time: float,
        duration: float,
        clip_data: dict | None = None,
    ):
        """
        Initialize timeline clip.

        Args:
            clip_id: Unique clip identifier
            file_path: Path to video file
            start_time: Start time on timeline (seconds)
            duration: Clip duration (seconds)
            clip_data: Optional full clip metadata
        """
        self.clip_id = clip_id
        self.file_path = file_path
        self.start_time = start_time
        self.duration = duration
        self.end_time = start_time + duration
        self.clip_data = clip_data or {}

        # Visual properties
        self.color = [0.3, 0.5, 0.8, 1.0]  # RGBA
        self.selected = False
        self.dragging = False


class TimelineWidget:
    """
    High-performance interactive timeline with Dear PyGui.

    Provides beat-synced video editing with waveform visualization,
    clip management, and cut point editing.
    """

    def __init__(
        self,
        width: int = 1200,
        height: int = 400,
        on_clip_selected: Callable | None = None,
        on_timeline_changed: Callable | None = None,
    ):
        """
        Initialize the TimelineWidget.

        Args:
            width: Timeline canvas width
            height: Timeline canvas height
            on_clip_selected: Callback when clip is selected (clip_id)
            on_timeline_changed: Callback when timeline is modified
        """
        self.width = width
        self.height = height
        self.on_clip_selected = on_clip_selected
        self.on_timeline_changed = on_timeline_changed

        # Timeline state
        self.clips: list[TimelineClip] = []
        self.cut_points: list[CutListEntry] = []
        self.current_time: float = 0.0
        self.total_duration: float = 60.0  # Default 60 seconds
        self.zoom_level: float = 1.0
        self.scroll_offset: float = 0.0

        # Audio analysis data
        self.audio_file: Path | None = None
        self.waveform: np.ndarray | None = None
        self.beat_times: list[float] = []
        self.onset_times: list[float] = []
        self.energy_curve: np.ndarray | None = None
        self.bpm: float | None = None

        # Pacing system
        self.pacing_engine: PacingEngine | None = None
        self.beatgrid_info: BeatGridInfo | None = None

        # Visual settings
        self.waveform_height = 100
        self.clip_track_height = 80
        self.beat_marker_height = 60
        self.energy_curve_height = 40

        # Interaction state
        self.selected_clip: TimelineClip | None = None
        self.dragging_clip: TimelineClip | None = None
        self.drag_offset: float = 0.0
        self.snap_to_beat: bool = True
        self.snap_threshold: float = 0.1  # seconds

        # Playback state (using State Machine)
        self.playback_state = PlaybackState.STOPPED
        self.playhead_position: float = 0.0
        self.playback_speed: float = 1.0
        self.loop_enabled: bool = True
        self._last_tick: float | None = None
        self._playback_timer: int | None = None

        # Dear PyGui elements
        self.dpg_window_tag: str | None = None
        self.dpg_canvas_tag: str | None = None

        # VideoManager for clip metadata
        self.video_manager = VideoManager()

    def create_window(self, parent: str | int | None = None) -> str:
        """
        Create Dear PyGui window and canvas.

        Args:
            parent: Optional parent window tag

        Returns:
            Window tag for reference
        """
        with dpg.window(
            label="Timeline Editor",
            width=self.width + 40,
            height=self.height + 100,
            pos=[20, 20],
            parent=parent,
        ) as window_id:
            self.dpg_window_tag = window_id

            # Toolbar
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Play/Pause", callback=self._on_play_pause, tag="timeline_play_btn"
                )
                dpg.add_button(label="Stop", callback=self._on_stop, tag="timeline_stop_btn")
                dpg.add_slider_float(
                    label="Zoom",
                    default_value=1.0,
                    min_value=0.1,
                    max_value=10.0,
                    callback=self._on_zoom_changed,
                    width=200,
                    tag="timeline_zoom_slider",
                )
                dpg.add_checkbox(
                    label="Snap to Beat",
                    default_value=True,
                    callback=self._on_snap_toggled,
                    tag="timeline_snap_checkbox",
                )
                dpg.add_text(
                    f"Time: 00:00.000 / {self._format_time(self.total_duration)}",
                    tag="timeline_time_label",
                )

            dpg.add_separator()

            # Timeline canvas
            with dpg.drawlist(
                width=self.width, height=self.height, tag="timeline_canvas"
            ) as canvas_id:
                self.dpg_canvas_tag = canvas_id

            # Playhead slider
            dpg.add_slider_float(
                label="",
                default_value=0.0,
                min_value=0.0,
                max_value=self.total_duration,
                callback=self._on_playhead_moved,
                width=self.width,
                tag="timeline_playhead_slider",
            )

        # Register mouse handlers
        with dpg.handler_registry():
            dpg.add_mouse_click_handler(callback=self._on_mouse_click)
            dpg.add_mouse_drag_handler(callback=self._on_mouse_drag)
            dpg.add_mouse_release_handler(callback=self._on_mouse_release)

        # Register keyboard shortcuts
        self._setup_keyboard_shortcuts()

        return window_id

    def load_audio_track(self, audio_file: str | Path, bpm: float | None = None) -> bool:
        """
        Load audio track and analyze for beat markers.

        Args:
            audio_file: Path to audio file
            bpm: Optional manual BPM override

        Returns:
            True if successful
        """
        try:
            audio_file = Path(audio_file)
            if not audio_file.exists():
                logger.warning(f"Audio file not found: {audio_file}")
                return False

            logger.info(f"Loading audio track: {audio_file.name}")
            self.audio_file = audio_file

            # Analyze audio
            analyzer = AudioAnalyzer()
            result = analyzer.analyze_full(audio_file)

            if not result:
                logger.error("Audio analysis failed")
                return False

            # Store analysis results
            self.bpm = bpm or result["bpm"]
            self.beat_times = result["beat_times"]
            self.onset_times = result["onset_times"]
            self.total_duration = result["duration"]

            # Create beatgrid info for pacing engine
            self.beatgrid_info = BeatGridInfo(
                bpm=self.bpm,
                offset=self.beat_times[0] if self.beat_times else 0.0,
                beat_times=self.beat_times,
            )

            # Initialize pacing engine
            self.pacing_engine = PacingEngine(self.beatgrid_info)

            # Update UI
            if self.dpg_window_tag:
                dpg.set_value("timeline_playhead_slider", {"max_value": self.total_duration})
                dpg.set_value(
                    "timeline_time_label",
                    f"Time: 00:00.000 / {self._format_time(self.total_duration)}",
                )

            # Load waveform for visualization
            self._load_waveform(audio_file)

            # Redraw timeline
            self._redraw()

            logger.info(f"Audio loaded: {self.bpm:.2f} BPM, {len(self.beat_times)} beats")
            return True

        except Exception as e:
            logger.error(f"Failed to load audio track: {e}")
            return False

    def _load_waveform(self, audio_file: Path) -> None:
        """Load audio waveform for visualization."""
        try:
            import librosa

            # Load audio
            y, sr = librosa.load(str(audio_file), sr=22050, mono=True)

            # Downsample for display (1 sample per pixel)
            target_samples = self.width
            if len(y) > target_samples:
                # Average pooling for downsampling
                chunk_size = len(y) // target_samples
                waveform = []
                for i in range(target_samples):
                    start = i * chunk_size
                    end = start + chunk_size
                    if end <= len(y):
                        waveform.append(np.abs(y[start:end]).mean())
                self.waveform = np.array(waveform)
            else:
                self.waveform = np.abs(y)

            # Normalize to 0-1 range
            if self.waveform.max() > 0:
                self.waveform = self.waveform / self.waveform.max()

        except Exception as e:
            logger.warning(f"Waveform loading failed: {e}")
            self.waveform = None

    def add_clip(
        self,
        clip_data: dict,
        start_time: float,
        duration: float | None = None,
        snap_to_beat: bool = True,
    ) -> bool:
        """
        Add video clip to timeline.

        Args:
            clip_data: Clip metadata from ClipLibraryWidget
            start_time: Start time on timeline
            duration: Optional duration override
            snap_to_beat: Whether to snap to nearest beat

        Returns:
            True if successful
        """
        try:
            # Snap to beat if enabled
            if snap_to_beat and self.snap_to_beat and self.beatgrid_info:
                start_time = self._snap_time_to_beat(start_time)

            # Get duration from clip data or use default
            if duration is None:
                duration = clip_data.get("duration", 2.0)

            # Create timeline clip
            clip = TimelineClip(
                clip_id=clip_data.get("id"),
                file_path=clip_data.get("file_path"),
                start_time=start_time,
                duration=duration,
                clip_data=clip_data,
            )

            # Check for overlaps
            if self._check_overlap(clip):
                logger.warning(f"Clip overlaps with existing clip at {start_time:.2f}s")
                # Could auto-adjust or reject

            self.clips.append(clip)
            self._redraw()

            if self.on_timeline_changed:
                self.on_timeline_changed()

            logger.info(f"Clip added: {Path(clip.file_path).name} at {start_time:.2f}s")
            return True

        except Exception as e:
            logger.error(f"Failed to add clip: {e}")
            return False

    def get_clip(self, clip_id: int | str) -> TimelineClip | None:
        """Return clip instance by id if present."""
        for clip in self.clips:
            if clip.clip_id == clip_id:
                return clip
        return None

    def remove_clip(self, clip_id: int | str) -> bool:
        """Remove clip from timeline."""
        for i, clip in enumerate(self.clips):
            if clip.clip_id == clip_id:
                del self.clips[i]
                self._redraw()
                if self.on_timeline_changed:
                    self.on_timeline_changed()
                return True
        return False

    def set_pacing_blueprint(self, cut_list: list[CutListEntry]) -> None:
        """
        Set pacing blueprint cut points.

        Args:
            cut_list: List of CutListEntry from PacingEngine
        """
        self.cut_points = cut_list
        self._redraw()

    def clear_timeline(self) -> None:
        """Clear all clips from timeline."""
        self.clips.clear()
        self.cut_points.clear()
        self.current_time = 0.0
        self._redraw()

        if self.on_timeline_changed:
            self.on_timeline_changed()

    def _snap_time_to_beat(self, time: float) -> float:
        """Snap time to nearest beat marker."""
        if not self.beat_times:
            return time

        # Find nearest beat
        beat_array = np.array(self.beat_times)
        idx = np.argmin(np.abs(beat_array - time))
        nearest_beat = self.beat_times[idx]

        # Only snap if within threshold
        if abs(nearest_beat - time) <= self.snap_threshold:
            return nearest_beat

        return time

    def _check_overlap(self, new_clip: TimelineClip) -> bool:
        """Check if clip overlaps with existing clips."""
        for clip in self.clips:
            if new_clip.start_time < clip.end_time and new_clip.end_time > clip.start_time:
                return True
        return False

    def _time_to_x(self, time: float) -> float:
        """Convert time to canvas x coordinate."""
        return (time - self.scroll_offset) * self.zoom_level * (self.width / self.total_duration)

    def _x_to_time(self, x: float) -> float:
        """Convert canvas x coordinate to time."""
        return x / (self.width / self.total_duration) / self.zoom_level + self.scroll_offset

    def _redraw(self) -> None:
        """Redraw entire timeline canvas."""
        if not self.dpg_canvas_tag:
            return

        # Clear canvas
        dpg.delete_item(self.dpg_canvas_tag, children_only=True)

        # Background
        dpg.draw_rectangle(
            [0, 0],
            [self.width, self.height],
            color=[30, 30, 30, 255],
            fill=[30, 30, 30, 255],
            parent=self.dpg_canvas_tag,
        )

        y_offset = 10

        # Draw waveform
        if self.waveform is not None:
            self._draw_waveform(y_offset)
        y_offset += self.waveform_height + 10

        # Draw beat markers
        self._draw_beat_markers(y_offset)
        y_offset += self.beat_marker_height + 10

        # Draw clips
        self._draw_clips(y_offset)
        y_offset += self.clip_track_height + 10

        # Draw cut points
        self._draw_cut_points()

        # Draw playhead
        self._draw_playhead()

    def _draw_waveform(self, y_offset: float) -> None:
        """Draw audio waveform."""
        if self.waveform is None:
            return

        # Draw waveform outline
        dpg.draw_rectangle(
            [0, y_offset],
            [self.width, y_offset + self.waveform_height],
            color=[60, 60, 60, 255],
            parent=self.dpg_canvas_tag,
        )

        # Draw waveform
        center_y = y_offset + self.waveform_height / 2
        samples = min(len(self.waveform), self.width)

        for i in range(samples):
            x = i * (self.width / samples)
            amplitude = self.waveform[i] * (self.waveform_height / 2)

            dpg.draw_line(
                [x, center_y - amplitude],
                [x, center_y + amplitude],
                color=[100, 180, 255, 200],
                thickness=1,
                parent=self.dpg_canvas_tag,
            )

    def _draw_beat_markers(self, y_offset: float) -> None:
        """Draw beat markers."""
        for beat_time in self.beat_times:
            x = self._time_to_x(beat_time)
            if 0 <= x <= self.width:
                dpg.draw_line(
                    [x, y_offset],
                    [x, y_offset + self.beat_marker_height],
                    color=[255, 200, 50, 150],
                    thickness=2,
                    parent=self.dpg_canvas_tag,
                )

    def _draw_clips(self, y_offset: float) -> None:
        """Draw video clips on timeline."""
        for clip in self.clips:
            x1 = self._time_to_x(clip.start_time)
            x2 = self._time_to_x(clip.end_time)

            # Only draw if visible
            if x2 < 0 or x1 > self.width:
                continue

            # Clip rectangle
            color = [int(c * 255) for c in clip.color]
            if clip.selected:
                color = [255, 255, 100, 255]  # Highlight selected

            dpg.draw_rectangle(
                [x1, y_offset],
                [x2, y_offset + self.clip_track_height],
                color=color,
                fill=[color[0] // 2, color[1] // 2, color[2] // 2, 200],
                thickness=2,
                parent=self.dpg_canvas_tag,
            )

            # Clip label
            clip_name = Path(clip.file_path).stem[:20]
            dpg.draw_text(
                [x1 + 5, y_offset + 10],
                clip_name,
                color=[255, 255, 255, 255],
                size=12,
                parent=self.dpg_canvas_tag,
            )

    def _draw_cut_points(self) -> None:
        """Draw cut point markers from pacing blueprint."""
        for cut in self.cut_points:
            x = self._time_to_x(cut.start_time)
            if 0 <= x <= self.width:
                dpg.draw_line(
                    [x, 0],
                    [x, self.height],
                    color=[255, 50, 50, 200],
                    thickness=2,
                    parent=self.dpg_canvas_tag,
                )

    def _draw_playhead(self) -> None:
        """Draw current time playhead."""
        x = self._time_to_x(self.current_time)
        if 0 <= x <= self.width:
            dpg.draw_line(
                [x, 0],
                [x, self.height],
                color=[255, 255, 255, 255],
                thickness=3,
                parent=self.dpg_canvas_tag,
            )

    def _format_time(self, seconds: float) -> str:
        """Format time as MM:SS.mmm"""
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:06.3f}"

    # Event handlers

    def _on_play_pause(self, sender=None, app_data=None, user_data=None):
        """Handle play/pause button."""
        logger.debug("Play/Pause clicked")
        if self.playback_state == PlaybackState.PLAYING:
            self._pause()
        else:
            self._play()

    def _on_stop(self, sender, app_data, user_data):
        """Handle stop button."""
        self._stop()

    def _on_zoom_changed(self, sender, app_data, user_data):
        """Handle zoom slider."""
        self.zoom_level = app_data
        self._redraw()

    def _on_snap_toggled(self, sender, app_data, user_data):
        """Handle snap to beat checkbox."""
        self.snap_to_beat = app_data

    def _on_playhead_moved(self, sender, app_data, user_data):
        """Handle playhead slider."""
        self.current_time = app_data
        dpg.set_value(
            "timeline_time_label",
            f"Time: {self._format_time(self.current_time)} / "
            f"{self._format_time(self.total_duration)}",
        )
        self._redraw()

    def _on_mouse_click(self, sender, app_data, user_data):
        """Handle mouse click on canvas."""
        try:
            # app_data: [button, x, y]
            if not app_data or len(app_data) < 3:
                return
            _, x, y = app_data
            if not self.dpg_canvas_tag:
                return
            rect_min = dpg.get_item_rect_min(self.dpg_canvas_tag)
            canvas_x = x - rect_min[0]
            if canvas_x < 0 or canvas_x > self.width:
                return
            new_time = np.clip(self._x_to_time(canvas_x), 0.0, self.total_duration)
            self.current_time = float(new_time)
            dpg.set_value("timeline_playhead_slider", self.current_time)
            dpg.set_value(
                "timeline_time_label",
                f"Time: {self._format_time(self.current_time)} / "
                f"{self._format_time(self.total_duration)}",
            )
            self._redraw()
        except Exception as exc:
            logger.debug(f"Mouse click handling failed: {exc}")

    def _on_mouse_drag(self, sender, app_data, user_data):
        """Handle mouse drag on canvas."""
        logger.info("Timeline mouse drag handling not implemented yet")

    def _on_mouse_release(self, sender, app_data, user_data):
        """Handle mouse release."""
        logger.info("Timeline mouse release handling not implemented yet")

    # Playback control helpers

    def _play(self):
        """Start playback."""
        if self.total_duration <= 0:
            logger.debug("Playback ignored: total_duration <= 0")
            return

        # Reset to start if at end
        if self.current_time >= self.total_duration:
            self.current_time = 0.0

        # Update state
        self.playback_state = PlaybackState.PLAYING
        self.playhead_position = self.current_time
        self._last_tick = time.perf_counter()

        # Update UI
        dpg.configure_item("timeline_play_btn", label="Pause")

        # Start playback timer
        self._schedule_tick()

        logger.debug(f"Playback started at {self.playhead_position:.2f}s")

    def _pause(self):
        """Pause playback."""
        self.playback_state = PlaybackState.PAUSED
        self._last_tick = None

        # Update UI
        dpg.configure_item("timeline_play_btn", label="Play")

        logger.debug(f"Playback paused at {self.current_time:.2f}s")

    def _stop(self):
        """Stop playback and reset to start."""
        self.playback_state = PlaybackState.STOPPED
        self.current_time = 0.0
        self.playhead_position = 0.0
        self._last_tick = None

        # Update UI
        dpg.configure_item("timeline_play_btn", label="Play")
        dpg.set_value("timeline_playhead_slider", 0.0)
        dpg.set_value(
            "timeline_time_label", f"Time: 00:00.000 / {self._format_time(self.total_duration)}"
        )

        self._redraw()
        logger.debug("Playback stopped")

    def _schedule_tick(self):
        """Registriert einen Frame-Callback fuer den naechsten Frame."""
        next_frame = dpg.get_frame_count() + 1
        dpg.set_frame_callback(next_frame, self._tick_playback)

    def _tick_playback(self, sender=None, app_data=None, user_data=None):
        """Update playhead position (called every frame during playback)."""
        if self.playback_state != PlaybackState.PLAYING:
            return

        now = time.perf_counter()
        if self._last_tick is None:
            self._last_tick = now

        # Calculate time delta
        dt = now - self._last_tick
        self._last_tick = now

        # Advance playhead position
        self.current_time += dt * self.playback_speed
        self.playhead_position = self.current_time

        # Handle end of timeline
        if self.current_time >= self.total_duration:
            if self.loop_enabled:
                self.current_time = 0.0
                self.playhead_position = 0.0
                logger.debug("Playback looped to start")
            else:
                self._stop()
                return

        # Update UI
        dpg.set_value("timeline_playhead_slider", self.current_time)
        dpg.set_value(
            "timeline_time_label",
            f"Time: {self._format_time(self.current_time)} / "
            f"{self._format_time(self.total_duration)}",
        )
        self._redraw()

        # Schedule next tick
        self._schedule_tick()

    def set_playback_speed(self, speed: float):
        """
        Set playback speed.

        Args:
            speed: Playback speed multiplier (0.25x - 4.0x)
        """
        self.playback_speed = max(0.25, min(4.0, float(speed)))
        logger.debug(f"Playback speed set to {self.playback_speed}x")

    def set_loop_enabled(self, enabled: bool):
        """
        Enable/disable loop mode.

        Args:
            enabled: True to enable looping at end of timeline
        """
        self.loop_enabled = bool(enabled)
        logger.debug(f"Loop mode: {'enabled' if enabled else 'disabled'}")

    def _seek(self, time: float):
        """
        Seek to specific time position.

        Args:
            time: Target time in seconds
        """
        self.current_time = max(0.0, min(time, self.total_duration))
        self.playhead_position = self.current_time

        # Update UI
        dpg.set_value("timeline_playhead_slider", self.current_time)
        dpg.set_value(
            "timeline_time_label",
            f"Time: {self._format_time(self.current_time)} / "
            f"{self._format_time(self.total_duration)}",
        )
        self._redraw()
        logger.debug(f"Seeked to {self.current_time:.2f}s")

    def _setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for playback control."""
        with dpg.handler_registry():
            # Space = Play/Pause
            dpg.add_key_press_handler(key=dpg.mvKey_Spacebar, callback=self._on_play_pause)

            # Home = Go to Start
            dpg.add_key_press_handler(key=dpg.mvKey_Home, callback=lambda: self._seek(0.0))

            # End = Go to End
            dpg.add_key_press_handler(
                key=dpg.mvKey_End, callback=lambda: self._seek(self.total_duration)
            )

        logger.debug("Keyboard shortcuts registered: Space (play/pause), Home (start), End (end)")
