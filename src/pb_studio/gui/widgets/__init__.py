"""
GUI Widgets for PB_studio

Custom PyQt6 and Dear PyGui widgets for audio/video editing interface.

Components:
- AudioTimelineWidget: High-performance audio waveform visualization with beat markers
- ClipLibraryWidget: Video clip library with thumbnails and drag-and-drop
- TimelineWidget: Interactive timeline with Dear PyGui (beat-synced editing, clip placement)

Note: ParameterDashboardWidget is in pb_studio.gui.parameter_dashboard_widget
      (not re-exported here to avoid circular imports)
"""

from .audio_timeline import AudioTimelineWidget
from .clip_library import ClipLibraryWidget
from .timeline_widget import TimelineWidget

__all__ = ["AudioTimelineWidget", "ClipLibraryWidget", "TimelineWidget"]
