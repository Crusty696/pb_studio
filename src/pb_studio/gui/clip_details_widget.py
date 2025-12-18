"""
Clip Details Widget - Zeigt detaillierte Clip-Informationen und Analyse-Daten.

Features:
- Thumbnail-Vorschau
- Video-Metadaten (Dauer, Aufloesung, FPS, etc.)
- Analyse-Ergebnisse (Farben, Motion, Mood, Style, Objects)
- Farbpalette Visualisierung
- Motion-Energie Kurve
"""

from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ColorPaletteWidget(QFrame):
    """Widget zur Anzeige der Farbpalette."""

    def __init__(self, colors: list[dict], parent=None):
        super().__init__(parent)
        self.colors = colors
        self.setFixedHeight(40)
        self.setStyleSheet("background-color: transparent;")

    def paintEvent(self, event):
        """Zeichnet die Farbpalette."""
        super().paintEvent(event)

        if not self.colors:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()
        x = 0

        for color_data in self.colors:
            rgb = color_data.get("rgb", [128, 128, 128])
            percentage = color_data.get("percentage", color_data.get("pct", 0))
            if isinstance(percentage, (int, float)) and 0.0 <= percentage <= 1.0:
                percentage = percentage * 100.0

            # Breite basierend auf Prozent
            color_width = int(width * float(percentage) / 100) if percentage else 0

            if color_width > 0:
                color = QColor(rgb[0], rgb[1], rgb[2])
                painter.fillRect(x, 0, color_width, height, color)
                x += color_width

        painter.end()


class MetricBar(QFrame):
    """Widget fuer eine einzelne Metrik mit Label und Fortschrittsbalken."""

    def __init__(
        self, label: str, value: float, max_value: float = 1.0, color: str = "#2a82da", parent=None
    ):
        super().__init__(parent)
        # Transparent background to blend with theme
        self.setStyleSheet("background-color: transparent;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        # Label
        label_widget = QLabel(label)
        label_widget.setFixedWidth(100)
        # Remove hardcoded color - theme handles it via "foreground_muted"
        label_widget.setObjectName("MetricLabel")
        layout.addWidget(label_widget)

        # Progress bar
        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setValue(int(value / max_value * 100))
        progress.setTextVisible(False)
        progress.setFixedHeight(12)
        # Keep functional color for chunk, but reset container
        progress.setStyleSheet(
            f"""
            QProgressBar {{
                background-color: rgba(0, 0, 0, 0.2);
                border: none;
                border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 2px;
            }}
        """
        )
        layout.addWidget(progress)

        # Value label
        if max_value == 1.0:
            value_text = f"{value:.2f}"
        elif max_value == 255:
            value_text = f"{value:.0f}"
        else:
            value_text = f"{value:.1f}"

        value_label = QLabel(value_text)
        value_label.setFixedWidth(50)
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(value_label)


class TagsWidget(QWidget):
    """Widget zur Anzeige von Tags/Labels."""

    def __init__(self, tags: list[str], colors: dict[str, str] | None = None, parent=None):
        super().__init__(parent)
        self.tags = tags
        self.colors = colors or {}

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        for tag in tags[:5]:  # Max 5 Tags
            color = self.colors.get(tag, "#555")
            tag_label = QLabel(tag)
            # Tags need specific background colors (functional), keep them
            tag_label.setStyleSheet(
                f"""
                QLabel {{
                    background-color: {color};
                    color: white;
                    border-radius: 3px;
                    padding: 2px 6px;
                    font-size: 10px;
                    font-weight: bold;
                }}
            """
            )
            layout.addWidget(tag_label)

        layout.addStretch()


class ClipDetailsWidget(QWidget):
    """
    Widget zur Anzeige detaillierter Clip-Informationen.
    """

    analyze_requested = pyqtSignal(int)
    find_similar_requested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._current_clip: dict[str, Any] | None = None
        self._init_ui()

    def _init_ui(self):
        """Initialisiert die UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Scroll Area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        self.content_layout = QVBoxLayout(content)
        self.content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Placeholder wenn kein Clip ausgewaehlt
        self.placeholder = QLabel("Select a clip to view details")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Use object name for styling if needed, removed hardcoded colors
        self.placeholder.setObjectName("PlaceholderLabel")
        self.content_layout.addWidget(self.placeholder)

        scroll.setWidget(content)
        layout.addWidget(scroll)

    def set_clip(self, clip_data: dict[str, Any]):
        """Setzt den anzuzeigenden Clip."""
        self._current_clip = clip_data
        self._update_display()

    def clear(self):
        """Leert die Anzeige."""
        self._current_clip = None
        self._clear_content()
        self.placeholder.show()

    def _clear_content(self):
        """Loescht allen Inhalt ausser Placeholder."""
        while self.content_layout.count() > 1:
            item = self.content_layout.takeAt(1)
            if item.widget():
                item.widget().deleteLater()

    def _update_display(self):
        """Aktualisiert die Anzeige."""
        if not self._current_clip:
            self.clear()
            return

        self._clear_content()
        self.placeholder.hide()

        clip = self._current_clip

        # Header mit Name
        header = QLabel(clip.get("name", "Unknown Clip"))
        # Use stylesheet for font size, let color be handled by theme
        header.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px 0;")
        header.setWordWrap(True)
        self.content_layout.addWidget(header)

        # Metadaten
        self._add_metadata_section(clip)

        # Analyse-Status
        analysis = clip.get("analysis", {})
        is_analyzed = clip.get("is_analyzed", False)

        if not is_analyzed or not analysis:
            self._add_not_analyzed_section(clip)
        else:
            # Analyse-Tabs
            self._add_analysis_tabs(analysis)

        # Action Buttons
        self._add_action_buttons(clip)

        self.content_layout.addStretch()

    def _add_metadata_section(self, clip: dict):
        """Fuegt Metadaten-Sektion hinzu."""
        group = QGroupBox("📊 Metadata")
        # Remove hardcoded colors/borders, rely on theme

        layout = QGridLayout(group)
        layout.setSpacing(8)

        # Dauer
        duration = clip.get("duration", 0)
        self._add_metadata_row(layout, 0, "Duration:", f"{duration:.2f}s")

        # Aufloesung
        width = clip.get("width", 0)
        height = clip.get("height", 0)
        self._add_metadata_row(layout, 1, "Resolution:", f"{width}x{height}")

        # FPS
        fps = clip.get("fps", 0)
        self._add_metadata_row(layout, 2, "Frame Rate:", f"{fps:.1f} fps")

        # Datei
        file_path = clip.get("file_path", "")
        if file_path:
            filename = Path(file_path).name
            self._add_metadata_row(layout, 3, "File:", filename[:40])

        # Fingerprint
        fingerprint = clip.get("content_fingerprint", "")
        if fingerprint:
            self._add_metadata_row(layout, 4, "Fingerprint:", fingerprint[:16] + "...")

        self.content_layout.addWidget(group)

    def _add_metadata_row(self, layout: QGridLayout, row: int, label: str, value: str):
        """Fuegt eine Metadaten-Zeile hinzu."""
        label_widget = QLabel(label)
        label_widget.setObjectName("MetaLabel")  # Style via theme (muted)
        layout.addWidget(label_widget, row, 0)

        value_widget = QLabel(value)
        value_widget.setObjectName("MetaValue")  # Style via theme (bright)
        layout.addWidget(value_widget, row, 1)

    def _add_not_analyzed_section(self, clip: dict):
        """Fuegt Hinweis fuer nicht analysierte Clips hinzu."""
        group = QFrame()
        # Remove hardcoded background/border, use object name
        group.setObjectName("InfoPanel")

        layout = QVBoxLayout(group)

        icon_label = QLabel("📝")
        icon_label.setStyleSheet("font-size: 32px;")
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon_label)

        text_label = QLabel("This clip has not been analyzed yet.")
        text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(text_label)

        analyze_btn = QPushButton("🔬 Analyze Now")
        # Keep functional color (blue) for primary action, but cleaner
        analyze_btn.setStyleSheet("background-color: #2a82da; color: white;")
        analyze_btn.clicked.connect(lambda: self.analyze_requested.emit(clip.get("id", 0)))
        layout.addWidget(analyze_btn)

        self.content_layout.addWidget(group)

    def _add_analysis_tabs(self, analysis: dict):
        """Fuegt Analyse-Tabs hinzu."""
        tabs = QTabWidget()
        # Remove hardcoded tab styling, use global theme

        # Color Tab
        if "color" in analysis:
            tabs.addTab(self._create_color_tab(analysis["color"]), "🎨 Color")

        # Motion Tab
        if "motion" in analysis:
            tabs.addTab(self._create_motion_tab(analysis["motion"]), "🏃 Motion")

        # Mood Tab
        if "mood" in analysis:
            tabs.addTab(self._create_mood_tab(analysis["mood"]), "😊 Mood")

        # Style Tab
        if "style" in analysis:
            tabs.addTab(self._create_style_tab(analysis["style"]), "🎬 Style")

        # Scene Tab
        if "scene" in analysis:
            tabs.addTab(self._create_scene_tab(analysis["scene"]), "📷 Scene")

        # Objects Tab
        if "objects" in analysis:
            tabs.addTab(self._create_objects_tab(analysis["objects"]), "🎯 Objects")

        self.content_layout.addWidget(tabs)

    def _create_color_tab(self, color_data: dict) -> QWidget:
        """Erstellt Color-Tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Farbpalette
        colors = color_data.get("dominant_colors", [])
        if colors:
            palette_label = QLabel("Dominant Colors:")
            palette_label.setStyleSheet("color: #fff; font-weight: bold;")
            layout.addWidget(palette_label)

            palette = ColorPaletteWidget(colors)
            layout.addWidget(palette)

        # Metriken
        brightness_value = color_data.get("brightness_value", 0.0)
        brightness_max = (
            255 if isinstance(brightness_value, (int, float)) and brightness_value > 2.0 else 1.0
        )
        layout.addWidget(
            MetricBar("Brightness", float(brightness_value or 0.0), brightness_max, "#f1c40f")
        )

        saturation_value = color_data.get("saturation_avg", color_data.get("saturation"))
        if saturation_value is not None:
            saturation_max = (
                255
                if isinstance(saturation_value, (int, float)) and saturation_value > 2.0
                else 1.0
            )
            layout.addWidget(
                MetricBar("Saturation", float(saturation_value), saturation_max, "#e74c3c")
            )

        # Tags
        temperature = color_data.get("temperature", "")
        brightness = color_data.get("brightness", "")
        tags = [t for t in [temperature, brightness] if t]
        if tags:
            layout.addWidget(
                TagsWidget(
                    tags,
                    {
                        "WARM": "#e74c3c",
                        "COOL": "#3498db",
                        "NEUTRAL": "#95a5a6",
                        "DARK": "#2c3e50",
                        "MEDIUM": "#7f8c8d",
                        "BRIGHT": "#f1c40f",
                    },
                )
            )

        color_moods = color_data.get("color_moods", [])
        if isinstance(color_moods, list) and color_moods:
            layout.addWidget(TagsWidget([str(m) for m in color_moods if m][:5]))

        layout.addStretch()
        return widget

    def _create_motion_tab(self, motion_data: dict) -> QWidget:
        """Erstellt Motion-Tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Motion Type Tag
        motion_type = motion_data.get("motion_type", "")
        if motion_type:
            layout.addWidget(
                TagsWidget(
                    [motion_type],
                    {
                        "STATIC": "#95a5a6",
                        "SLOW": "#3498db",
                        "MEDIUM": "#2ecc71",
                        "FAST": "#f39c12",
                        "EXTREME": "#e74c3c",
                    },
                )
            )

        # Metriken
        layout.addWidget(
            MetricBar("Motion Score", motion_data.get("motion_score", 0), 1.0, "#e74c3c")
        )

        layout.addWidget(
            MetricBar("Camera Mag.", motion_data.get("camera_magnitude", 0), 10.0, "#3498db")
        )

        # Camera Motion
        camera_motion = motion_data.get("camera_motion", "")
        if camera_motion:
            cam_label = QLabel(f"Camera: {camera_motion.replace('_', ' ').title()}")
            cam_label.setStyleSheet("color: #aaa; padding-top: 10px;")
            layout.addWidget(cam_label)

        # Rhythm
        rhythm = motion_data.get("motion_rhythm", "")
        if rhythm:
            rhythm_label = QLabel(f"Rhythm: {rhythm}")
            rhythm_label.setStyleSheet("color: #aaa;")
            layout.addWidget(rhythm_label)

        layout.addStretch()
        return widget

    def _create_mood_tab(self, mood_data: dict) -> QWidget:
        """Erstellt Mood-Tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Moods
        moods = mood_data.get("moods", [])
        if moods:
            layout.addWidget(
                TagsWidget(
                    moods,
                    {
                        "ENERGETIC": "#e74c3c",
                        "CALM": "#3498db",
                        "DARK": "#2c3e50",
                        "BRIGHT": "#f1c40f",
                        "MELANCHOLIC": "#9b59b6",
                        "EUPHORIC": "#e91e63",
                        "AGGRESSIVE": "#c0392b",
                        "PEACEFUL": "#1abc9c",
                        "MYSTERIOUS": "#8e44ad",
                        "CHEERFUL": "#f39c12",
                        "TENSE": "#e67e22",
                        "DREAMY": "#a29bfe",
                        "COOL": "#00bcd4",
                        "WARM": "#ff5722",
                    },
                )
            )

        # Metriken
        if mood_data.get("brightness") is not None:
            layout.addWidget(
                MetricBar("Brightness", mood_data.get("brightness", 0.0), 1.0, "#f1c40f")
            )
        if mood_data.get("saturation") is not None:
            layout.addWidget(
                MetricBar("Saturation", mood_data.get("saturation", 0.0), 1.0, "#e74c3c")
            )
        if mood_data.get("contrast") is not None:
            layout.addWidget(MetricBar("Contrast", mood_data.get("contrast", 0.0), 1.0, "#9b59b6"))

        layout.addWidget(MetricBar("Energy", mood_data.get("energy", 0.0), 1.0, "#e74c3c"))

        if mood_data.get("warm_ratio") is not None:
            layout.addWidget(
                MetricBar("Warm Ratio", mood_data.get("warm_ratio", 0.0), 1.0, "#ff5722")
            )
        if mood_data.get("cool_ratio") is not None:
            layout.addWidget(
                MetricBar("Cool Ratio", mood_data.get("cool_ratio", 0.0), 1.0, "#00bcd4")
            )

        layout.addStretch()
        return widget

    def _create_style_tab(self, style_data: dict) -> QWidget:
        """Erstellt Style-Tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Styles
        styles = style_data.get("styles", [])
        if styles:
            layout.addWidget(
                TagsWidget(
                    styles,
                    {
                        "VINTAGE": "#8e6d3e",
                        "FILMIC": "#5d6d7e",
                        "NEON": "#e91e63",
                        "MINIMALIST": "#bdc3c7",
                        "PSYCHEDELIC": "#9b59b6",
                        "CINEMATIC": "#2c3e50",
                        "DIGITAL": "#3498db",
                        "DREAMY": "#a29bfe",
                        "HIGH_CONTRAST": "#2c3e50",
                        "LOW_KEY": "#1c2833",
                        "HIGH_KEY": "#f7dc6f",
                        "STANDARD": "#7f8c8d",
                    },
                )
            )

        # Metriken
        layout.addWidget(
            MetricBar(
                "Sharpness", min(style_data.get("sharpness", 500) / 2000, 1.0), 1.0, "#3498db"
            )
        )

        layout.addWidget(MetricBar("Noise Level", style_data.get("noise_level", 0), 1.0, "#f39c12"))

        if style_data.get("dynamic_range") is not None:
            layout.addWidget(
                MetricBar("Dynamic Range", style_data.get("dynamic_range", 0.0), 1.0, "#9b59b6")
            )

        layout.addWidget(MetricBar("Vignette", style_data.get("vignette_score", 0), 1.0, "#2c3e50"))

        layout.addStretch()
        return widget

    def _create_scene_tab(self, scene_data: dict) -> QWidget:
        """Erstellt Scene-Tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Scene Types
        scene_types = scene_data.get("scene_types", [])
        if scene_types:
            layout.addWidget(
                TagsWidget(
                    scene_types,
                    {
                        "PORTRAIT": "#e91e63",
                        "LANDSCAPE": "#27ae60",
                        "CLOSEUP": "#e74c3c",
                        "WIDE_SHOT": "#3498db",
                        "ABSTRACT": "#9b59b6",
                        "GEOMETRIC": "#f39c12",
                        "BUSY": "#e67e22",
                        "MINIMAL": "#bdc3c7",
                        "CROWD": "#c0392b",
                        "NATURE": "#27ae60",
                        "URBAN": "#7f8c8d",
                        "INDOOR": "#8e44ad",
                    },
                )
            )

        # Face Info
        has_face = scene_data.get("has_face", False)
        face_count = scene_data.get("face_count", 0)

        if has_face:
            face_label = QLabel(f"👤 {face_count} face(s) detected")
            face_label.setStyleSheet("color: #2ecc71; padding: 10px 0;")
        else:
            face_label = QLabel("No faces detected")
            face_label.setStyleSheet("color: #888; padding: 10px 0;")
        layout.addWidget(face_label)

        # Metriken
        layout.addWidget(
            MetricBar("Edge Density", scene_data.get("edge_density", 0), 0.3, "#3498db")
        )

        if scene_data.get("texture_variance") is not None:
            layout.addWidget(
                MetricBar("Texture", scene_data.get("texture_variance", 0.0), 1.0, "#f39c12")
            )

        layout.addWidget(
            MetricBar("Depth of Field", scene_data.get("depth_of_field", 0.5), 1.0, "#9b59b6")
        )

        layout.addStretch()
        return widget

    def _create_objects_tab(self, objects_data: dict) -> QWidget:
        """Erstellt Objects-Tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # New schema (PB_studio): detected_objects/object_counts/content_tags
        detected_objects: list[str] = []
        raw_detected = objects_data.get("detected_objects")
        if isinstance(raw_detected, list):
            detected_objects = [str(o) for o in raw_detected if o]
        else:
            # Legacy fallback: objects=[{"name": ...}, ...] or objects=["person", ...]
            raw_objects = objects_data.get("objects", [])
            if isinstance(raw_objects, list) and raw_objects:
                if isinstance(raw_objects[0], dict):
                    detected_objects = [
                        str(o.get("name", "")) for o in raw_objects if o.get("name")
                    ]
                else:
                    detected_objects = [str(o) for o in raw_objects if o]

        object_counts = objects_data.get("object_counts")
        total_detections = None
        if isinstance(object_counts, dict) and object_counts:
            try:
                total_detections = int(sum(int(v) for v in object_counts.values()))
            except Exception:
                total_detections = None

        if total_detections is None:
            total_detections = len(detected_objects)

        # Detected objects
        if detected_objects:
            objects_label = QLabel("Detected objects:")
            objects_label.setStyleSheet("color: #fff; font-weight: bold; padding-top: 6px;")
            layout.addWidget(objects_label)
            layout.addWidget(TagsWidget(detected_objects[:5]))
        else:
            no_obj_label = QLabel("No objects detected")
            no_obj_label.setStyleSheet("color: #888; padding: 10px 0;")
            layout.addWidget(no_obj_label)

        # Detection count
        info_label = QLabel(f"Detections: {total_detections}")
        info_label.setStyleSheet("color: #aaa; padding: 6px 0;")
        layout.addWidget(info_label)

        # Counts (top classes)
        if isinstance(object_counts, dict) and object_counts:
            counts_label = QLabel("Top classes:")
            counts_label.setStyleSheet("color: #fff; font-weight: bold; padding-top: 6px;")
            layout.addWidget(counts_label)

            top = sorted(object_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
            count_tags = [f"{name}x{count}" for name, count in top]
            layout.addWidget(TagsWidget(count_tags))

        # Metrics (optional)
        if objects_data.get("green_ratio") is not None:
            layout.addWidget(
                MetricBar("Green Ratio", objects_data.get("green_ratio", 0.0), 1.0, "#27ae60")
            )
        if objects_data.get("sky_ratio") is not None:
            layout.addWidget(
                MetricBar("Sky Ratio", objects_data.get("sky_ratio", 0.0), 1.0, "#3498db")
            )
        if objects_data.get("symmetry") is not None:
            layout.addWidget(
                MetricBar("Symmetry", objects_data.get("symmetry", 0.0), 1.0, "#9b59b6")
            )

        # Content tags (includes semantic tags merged during analysis)
        content_tags = objects_data.get("content_tags") or objects_data.get("features") or []
        if isinstance(content_tags, list) and content_tags:
            tags_label = QLabel("Content tags:")
            tags_label.setStyleSheet("color: #fff; font-weight: bold; padding-top: 6px;")
            layout.addWidget(tags_label)
            layout.addWidget(TagsWidget([str(t) for t in content_tags if t][:5]))

        layout.addStretch()
        return widget

    def _add_action_buttons(self, clip: dict):
        """Fuegt Action-Buttons hinzu."""
        buttons = QHBoxLayout()

        # Find Similar
        similar_btn = QPushButton("🔍 Find Similar")
        similar_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #3a3a3a;
                color: #fff;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 8px 12px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border-color: #2a82da;
            }
        """
        )
        similar_btn.clicked.connect(lambda: self.find_similar_requested.emit(clip.get("id", 0)))
        buttons.addWidget(similar_btn)

        # Re-Analyze
        if clip.get("is_analyzed", False):
            reanalyze_btn = QPushButton("🔄 Re-Analyze")
            reanalyze_btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #3a3a3a;
                    color: #fff;
                    border: 1px solid #555;
                    border-radius: 4px;
                    padding: 8px 12px;
                }
                QPushButton:hover {
                    background-color: #4a4a4a;
                    border-color: #f39c12;
                }
            """
            )
            reanalyze_btn.clicked.connect(lambda: self.analyze_requested.emit(clip.get("id", 0)))
            buttons.addWidget(reanalyze_btn)

        buttons.addStretch()
        self.content_layout.addLayout(buttons)
