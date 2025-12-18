"""
Clip Filter Widget - Erweiterte Filterung fuer Clip Library.

Ermoeglicht Filterung nach:
- Motion Type (STATIC, SLOW, MEDIUM, FAST, EXTREME)
- Mood (ENERGETIC, CALM, DARK, BRIGHT, etc.)
- Style (VINTAGE, FILMIC, NEON, etc.)
- Scene Type (PORTRAIT, LANDSCAPE, CLOSEUP, etc.)
- Brightness (DARK, MEDIUM, BRIGHT)
- Objects (erkannte Objekte via YOLO)
"""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..utils.logger import get_logger
from .theme_colors import DARK_COLORS, LIGHT_COLORS

logger = get_logger(__name__)


# Filter-Werte aus den Analyzern
MOTION_TYPES = ["All", "STATIC", "SLOW", "MEDIUM", "FAST", "EXTREME"]
MOOD_TYPES = [
    "All",
    "ENERGETIC",
    "CALM",
    "DARK",
    "BRIGHT",
    "MELANCHOLIC",
    "EUPHORIC",
    "AGGRESSIVE",
    "PEACEFUL",
    "MYSTERIOUS",
    "CHEERFUL",
    "TENSE",
    "DREAMY",
    "COOL",
    "WARM",
]
STYLE_TYPES = [
    "All",
    "VINTAGE",
    "FILMIC",
    "NEON",
    "MINIMALIST",
    "PSYCHEDELIC",
    "CINEMATIC",
    "DIGITAL",
    "DREAMY",
    "HIGH_CONTRAST",
    "LOW_KEY",
    "HIGH_KEY",
    "STANDARD",
]
SCENE_TYPES = [
    "All",
    "PORTRAIT",
    "LANDSCAPE",
    "CLOSEUP",
    "WIDE_SHOT",
    "ABSTRACT",
    "GEOMETRIC",
    "BUSY",
    "MINIMAL",
    "CROWD",
    "NATURE",
    "URBAN",
    "INDOOR",
]
BRIGHTNESS_TYPES = ["All", "DARK", "MEDIUM", "BRIGHT"]
CAMERA_MOTION_TYPES = [
    "All",
    "STATIC_CAM",
    "PAN_LEFT",
    "PAN_RIGHT",
    "TILT_UP",
    "TILT_DOWN",
    "ZOOM_IN",
    "ZOOM_OUT",
    "TRACKING",
    "HANDHELD",
]


class ClipFilterWidget(QFrame):
    """
    Widget fuer erweiterte Clip-Filterung.

    Signals:
        filters_changed: Emittiert wenn sich Filter aendern (filter_dict)
        analyze_requested: Emittiert wenn Analyse-Button geklickt wird
    """

    filters_changed = pyqtSignal(dict)
    analyze_requested = pyqtSignal()
    reanalyze_all_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)

        # Use default dark theme (can be changed via update_theme)
        self._theme = DARK_COLORS
        self._apply_theme()

        # Filter-Status
        self._current_filters: dict[str, str] = {
            "motion_type": "All",
            "mood": "All",
            "style": "All",
            "scene_type": "All",
            "brightness": "All",
            "camera_motion": "All",
        }

        self._expanded = False
        self._init_ui()

    def _apply_theme(self):
        """Apply theme colors to widget styling."""
        theme = self._theme
        self.setStyleSheet(
            f"""
            ClipFilterWidget {{
                background-color: {theme.background_alt};
                border: 1px solid {theme.border};
                border-radius: 5px;
            }}
            QLabel {{
                color: {theme.foreground};
                font-size: 11px;
            }}
            QComboBox {{
                background-color: {theme.background_input};
                color: {theme.foreground};
                border: 1px solid {theme.border};
                border-radius: 3px;
                padding: 3px 8px;
                min-width: 100px;
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid {theme.foreground_muted};
                margin-right: 5px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {theme.background_input};
                color: {theme.foreground};
                selection-background-color: {theme.primary};
            }}
            QPushButton {{
                background-color: {theme.background_input};
                color: {theme.foreground};
                border: 1px solid {theme.border};
                border-radius: 3px;
                padding: 5px 10px;
            }}
            QPushButton:hover {{
                background-color: {theme.hover};
                border: 1px solid {theme.primary};
            }}
            QPushButton:pressed {{
                background-color: {theme.primary};
            }}
            QGroupBox {{
                color: {theme.foreground};
                border: 1px solid {theme.border};
                border-radius: 3px;
                margin-top: 8px;
                padding-top: 8px;
                font-weight: bold;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }}
        """
        )

    def update_theme(self, use_dark: bool = True):
        """Update widget styling based on theme selection.

        Args:
            use_dark: True for dark theme, False for light theme
        """
        self._theme = DARK_COLORS if use_dark else LIGHT_COLORS
        self._apply_theme()

    def _init_ui(self):
        """Initialisiert die UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Header mit Expand/Collapse Button
        header = QHBoxLayout()

        self.filter_label = QLabel("🔍 Filter")
        self.filter_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        header.addWidget(self.filter_label)

        header.addStretch()

        # Quick-Filter Status anzeigen
        self.active_filter_label = QLabel("")
        self.active_filter_label.setStyleSheet(f"color: {self._theme.primary}; font-size: 10px;")
        header.addWidget(self.active_filter_label)

        # Reset Button
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setFixedWidth(60)
        self.reset_btn.clicked.connect(self._reset_filters)
        header.addWidget(self.reset_btn)

        # Expand/Collapse Button
        self.expand_btn = QPushButton("▼")
        self.expand_btn.setFixedWidth(30)
        self.expand_btn.clicked.connect(self._toggle_expand)
        header.addWidget(self.expand_btn)

        layout.addLayout(header)

        # Kompakte Filter-Zeile (immer sichtbar)
        compact_row = QHBoxLayout()

        # Motion Type
        compact_row.addWidget(QLabel("Motion:"))
        self.motion_combo = QComboBox()
        self.motion_combo.addItems(MOTION_TYPES)
        self.motion_combo.currentTextChanged.connect(
            lambda t: self._on_filter_changed("motion_type", t)
        )
        compact_row.addWidget(self.motion_combo)

        # Mood
        compact_row.addWidget(QLabel("Mood:"))
        self.mood_combo = QComboBox()
        self.mood_combo.addItems(MOOD_TYPES)
        self.mood_combo.currentTextChanged.connect(lambda t: self._on_filter_changed("mood", t))
        compact_row.addWidget(self.mood_combo)

        # Style
        compact_row.addWidget(QLabel("Style:"))
        self.style_combo = QComboBox()
        self.style_combo.addItems(STYLE_TYPES)
        self.style_combo.currentTextChanged.connect(lambda t: self._on_filter_changed("style", t))
        compact_row.addWidget(self.style_combo)

        compact_row.addStretch()
        layout.addLayout(compact_row)

        # Erweiterte Filter (collapsed by default)
        self.expanded_widget = QWidget()
        expanded_layout = QVBoxLayout(self.expanded_widget)
        expanded_layout.setContentsMargins(0, 8, 0, 0)

        # Zweite Filter-Zeile
        second_row = QHBoxLayout()

        # Scene Type
        second_row.addWidget(QLabel("Scene:"))
        self.scene_combo = QComboBox()
        self.scene_combo.addItems(SCENE_TYPES)
        self.scene_combo.currentTextChanged.connect(
            lambda t: self._on_filter_changed("scene_type", t)
        )
        second_row.addWidget(self.scene_combo)

        # Brightness
        second_row.addWidget(QLabel("Brightness:"))
        self.brightness_combo = QComboBox()
        self.brightness_combo.addItems(BRIGHTNESS_TYPES)
        self.brightness_combo.currentTextChanged.connect(
            lambda t: self._on_filter_changed("brightness", t)
        )
        second_row.addWidget(self.brightness_combo)

        # Camera Motion
        second_row.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(CAMERA_MOTION_TYPES)
        self.camera_combo.currentTextChanged.connect(
            lambda t: self._on_filter_changed("camera_motion", t)
        )
        second_row.addWidget(self.camera_combo)

        second_row.addStretch()
        expanded_layout.addLayout(second_row)

        # Analyse-Actions
        action_row = QHBoxLayout()

        self.analyze_btn = QPushButton("🔬 Analyze Unanalyzed")
        self.analyze_btn.setToolTip("Analyze all clips that haven't been analyzed yet")
        self.analyze_btn.clicked.connect(self.analyze_requested.emit)
        action_row.addWidget(self.analyze_btn)

        self.reanalyze_btn = QPushButton("🔄 Re-Analyze All")
        self.reanalyze_btn.setToolTip("Re-analyze all clips (overwrites existing data)")
        self.reanalyze_btn.clicked.connect(self.reanalyze_all_requested.emit)
        action_row.addWidget(self.reanalyze_btn)

        action_row.addStretch()

        # Status Label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {self._theme.foreground_muted}; font-size: 10px;")
        action_row.addWidget(self.status_label)

        expanded_layout.addLayout(action_row)

        layout.addWidget(self.expanded_widget)
        # Show expanded widget by default so Analyze buttons are visible
        self.expanded_widget.show()
        self._expanded = True
        self.expand_btn.setText("▲")

        self.setLayout(layout)

    def _toggle_expand(self):
        """Toggle erweiterte Filter an/aus."""
        self._expanded = not self._expanded

        if self._expanded:
            self.expanded_widget.show()
            self.expand_btn.setText("▲")
        else:
            self.expanded_widget.hide()
            self.expand_btn.setText("▼")

    def _on_filter_changed(self, filter_name: str, value: str):
        """Handler fuer Filter-Aenderungen."""
        self._current_filters[filter_name] = value
        self._update_active_filter_label()
        self.filters_changed.emit(self.get_filters())

    def _update_active_filter_label(self):
        """Aktualisiert die Anzeige aktiver Filter."""
        active = []
        for key, value in self._current_filters.items():
            if value != "All":
                active.append(f"{key.replace('_', ' ').title()}: {value}")

        if active:
            self.active_filter_label.setText(f"({len(active)} active)")
        else:
            self.active_filter_label.setText("")

    def _reset_filters(self):
        """Setzt alle Filter zurueck."""
        self.motion_combo.setCurrentText("All")
        self.mood_combo.setCurrentText("All")
        self.style_combo.setCurrentText("All")
        self.scene_combo.setCurrentText("All")
        self.brightness_combo.setCurrentText("All")
        self.camera_combo.setCurrentText("All")

        self._current_filters = {
            "motion_type": "All",
            "mood": "All",
            "style": "All",
            "scene_type": "All",
            "brightness": "All",
            "camera_motion": "All",
        }

        self._update_active_filter_label()
        self.filters_changed.emit(self.get_filters())

    def get_filters(self) -> dict[str, str]:
        """Gibt aktuelle Filter zurueck."""
        # Nur aktive Filter zurueckgeben (nicht 'All')
        return {k: v for k, v in self._current_filters.items() if v != "All"}

    def set_filter(self, filter_name: str, value: str):
        """Setzt einen Filter programmatisch."""
        if filter_name == "motion_type":
            self.motion_combo.setCurrentText(value)
        elif filter_name == "mood":
            self.mood_combo.setCurrentText(value)
        elif filter_name == "style":
            self.style_combo.setCurrentText(value)
        elif filter_name == "scene_type":
            self.scene_combo.setCurrentText(value)
        elif filter_name == "brightness":
            self.brightness_combo.setCurrentText(value)
        elif filter_name == "camera_motion":
            self.camera_combo.setCurrentText(value)

    def set_status(self, text: str):
        """Setzt Status-Text."""
        self.status_label.setText(text)

    def set_unanalyzed_count(self, count: int):
        """Setzt Anzahl unanalysierter Clips."""
        if count > 0:
            self.analyze_btn.setText(f"🔬 Analyze {count} Unanalyzed")
            self.analyze_btn.setEnabled(True)
        else:
            self.analyze_btn.setText("✓ All Analyzed")
            self.analyze_btn.setEnabled(False)


class QuickFilterBar(QWidget):
    """
    Kompakte Filter-Leiste fuer schnellen Zugriff.
    Zeigt Chips/Tags fuer haeufig verwendete Filter.
    """

    filter_clicked = pyqtSignal(str, str)  # filter_name, value

    def __init__(self, parent=None):
        super().__init__(parent)
        # Use default dark theme (can be changed via update_theme)
        self._theme = DARK_COLORS
        self._init_ui()

    def _init_ui(self):
        """Initialisiert die UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Quick-Filter Chips
        quick_filters = [
            ("Fast Motion", "motion_type", "FAST"),
            ("Energetic", "mood", "ENERGETIC"),
            ("Calm", "mood", "CALM"),
            ("Portrait", "scene_type", "PORTRAIT"),
            ("Cinematic", "style", "CINEMATIC"),
        ]

        for label, filter_name, value in quick_filters:
            chip = QPushButton(label)
            chip.setCheckable(True)
            chip.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {self._theme.background_input};
                    color: {self._theme.foreground_muted};
                    border: 1px solid {self._theme.border};
                    border-radius: 10px;
                    padding: 3px 10px;
                    font-size: 10px;
                }}
                QPushButton:hover {{
                    background-color: {self._theme.hover};
                    color: {self._theme.foreground};
                }}
                QPushButton:checked {{
                    background-color: {self._theme.primary};
                    color: {self._theme.primary_text};
                    border-color: {self._theme.primary};
                }}
            """
            )
            chip.clicked.connect(
                lambda checked, fn=filter_name, v=value: self.filter_clicked.emit(
                    fn, v if checked else "All"
                )
            )
            layout.addWidget(chip)

        layout.addStretch()

    def update_theme(self, use_dark: bool = True):
        """Update widget styling based on theme selection.

        Args:
            use_dark: True for dark theme, False for light theme
        """
        self._theme = DARK_COLORS if use_dark else LIGHT_COLORS
        # Re-initialize UI to apply new theme
        self._init_ui()


def apply_filters_to_clips(clips: list[dict], filters: dict[str, str]) -> list[dict]:
    """
    Wendet Filter auf eine Clip-Liste an.

    Args:
        clips: Liste von Clip-Dicts mit Analyse-Daten
        filters: Filter-Dict (z.B. {'motion_type': 'FAST', 'mood': 'ENERGETIC'})

    Returns:
        Gefilterte Liste
    """
    if not filters:
        return clips

    filtered = []

    for clip in clips:
        match = True

        for filter_name, filter_value in filters.items():
            if filter_value == "All":
                continue

            # Hole Analyse-Daten aus Clip
            analysis = clip.get("analysis", {})

            if filter_name == "motion_type":
                clip_value = analysis.get("motion", {}).get("motion_type", "")
                if clip_value != filter_value:
                    match = False
                    break

            elif filter_name == "mood":
                clip_moods = analysis.get("mood", {}).get("moods", [])
                if filter_value not in clip_moods:
                    match = False
                    break

            elif filter_name == "style":
                clip_styles = analysis.get("style", {}).get("styles", [])
                if filter_value not in clip_styles:
                    match = False
                    break

            elif filter_name == "scene_type":
                clip_scenes = analysis.get("scene", {}).get("scene_types", [])
                if filter_value not in clip_scenes:
                    match = False
                    break

            elif filter_name == "brightness":
                clip_brightness = analysis.get("color", {}).get("brightness", "")
                if clip_brightness != filter_value:
                    match = False
                    break

            elif filter_name == "camera_motion":
                clip_camera = analysis.get("motion", {}).get("camera_motion", "")
                if clip_camera != filter_value:
                    match = False
                    break

        if match:
            filtered.append(clip)

    return filtered
