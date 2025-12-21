
"""
Keyframe Export Dialog
----------------------
Dialog for generating and exporting mathematical animation strings (Deforum, etc.)
based on audio analysis (beats).
"""

import pyperclip
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...database.connection import DatabaseManager
from ...database.models import AudioTrack
from ...pacing.keyframe_generator import KeyframeGenerator
from ...utils.logger import get_logger

logger = get_logger(__name__)


class KeyframeExportDialog(QDialog):
    """
    Dialog to configure and export keyframe strings.
    """

    def __init__(self, parent=None, project_id=None):
        super().__init__(parent)
        self.setWindowTitle("Export Keyframes")
        self.resize(500, 300)
        self.project_id = project_id
        
        # Database connection (short-lived)
        self.db_manager = DatabaseManager()
        # Note: In a real app, we might pass the session or manager from MainWindow,
        # but creating a new instance connecting to the same DB is also fine for a dialog.

        self._init_ui()
        self._load_audio_tracks()

    def _init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Form Layout for options
        form_layout = QFormLayout()
        
        # 1. Audio Track Selector
        self.track_combo = QComboBox()
        form_layout.addRow("Audio Source:", self.track_combo)
        
        # 2. Curve Type
        self.curve_combo = QComboBox()
        self.curve_combo.addItems(["Zoom", "Shake (Translation X)"])
        form_layout.addRow("Curve Type:", self.curve_combo)
        
        # 3. Intensity
        self.intensity_spin = QDoubleSpinBox()
        self.intensity_spin.setRange(0.1, 10.0)
        self.intensity_spin.setSingleStep(0.1)
        self.intensity_spin.setValue(1.0)
        self.intensity_spin.setToolTip("Multiplier for the effect strength")
        form_layout.addRow("Intensity:", self.intensity_spin)
        
        # 4. FPS
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(30)
        form_layout.addRow("Target FPS:", self.fps_spin)

        layout.addLayout(form_layout)
        
        # Preview / Info Label
        self.info_label = QLabel("Select a track with beat analysis.")
        self.info_label.setStyleSheet("color: gray; font-style: italic;")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        layout.addStretch()

        # Buttons
        btn_layout = QHBoxLayout()
        
        self.copy_btn = QPushButton("Copy to Clipboard")
        self.copy_btn.clicked.connect(self.generate_and_copy)
        self.copy_btn.setStyleSheet("font-weight: bold; padding: 5px;")
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.copy_btn)
        btn_layout.addWidget(self.close_btn)
        
        layout.addLayout(btn_layout)

    def _load_audio_tracks(self):
        """Loads audio tracks for the current project."""
        if not self.project_id:
            self.track_combo.addItem("No Project Active")
            self.track_combo.setEnabled(False)
            self.copy_btn.setEnabled(False)
            return

        try:
            session = self.db_manager.get_session()
            tracks = session.query(AudioTrack).filter_by(project_id=self.project_id).all()
            
            if not tracks:
                self.track_combo.addItem("No Audio Tracks Found")
                self.track_combo.setEnabled(False)
                self.copy_btn.setEnabled(False)
            else:
                for track in tracks:
                    # Store track ID in user data
                    self.track_combo.addItem(f"{track.name} (BPM: {track.bpm or '?'})", track.id)
            
            session.close()
            
        except Exception as e:
            logger.error(f"Failed to load audio tracks: {e}")
            self.track_combo.addItem("Error loading tracks")

    def generate_and_copy(self):
        """Generates the string and copies to clipboard."""
        track_id = self.track_combo.currentData()
        if not track_id:
            return

        try:
            session = self.db_manager.get_session()
            track = session.query(AudioTrack).get(track_id)
            
            if not track or not track.beatgrids:
                QMessageBox.warning(self, "No Analysis", "This track has no beatgrid data. Please analyze it first.")
                session.close()
                return
                
            # Get beat times from the first beatgrid
            beatgrid = track.beatgrids[0]
            beats = beatgrid.get_beat_times()
            
            session.close()
            
            if not beats:
                QMessageBox.warning(self, "No Beats", "Beatgrid is empty.")
                return
                
            # Generate String
            curve_type = self.curve_combo.currentText()
            intensity = self.intensity_spin.value()
            fps = self.fps_spin.value()
            
            result_string = ""
            
            if curve_type == "Zoom":
                result_string = KeyframeGenerator.generate_zoom_curve(beats, intensity, fps)
            elif "Shake" in curve_type:
                result_string = KeyframeGenerator.generate_shake_curve(beats, intensity, fps)
                
            # Copy
            pyperclip.copy(result_string)
            
            # Feedback
            self.info_label.setText(f"Copied {curve_type} string ({len(result_string)} chars) to clipboard!")
            self.info_label.setStyleSheet("color: green; font-weight: bold;")
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            QMessageBox.critical(self, "Error", str(e))
