"""
Dialog for creating and editing projects.
"""


from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from ...utils.logger import get_logger

logger = get_logger(__name__)


class ProjectDialog(QDialog):
    """Dialog for creating or editing a project."""

    def __init__(self, parent=None, project_to_edit=None):
        super().__init__(parent)
        self.project_to_edit = project_to_edit
        self.result_data = None

        self.setWindowTitle("Neues Projekt" if not project_to_edit else "Projekt bearbeiten")
        self.setMinimumWidth(500)
        self.setModal(True)

        self._init_ui()

        if project_to_edit:
            self._load_project_data(project_to_edit)

    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout()

        # Form
        form_layout = QFormLayout()

        # Name
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Mein neues Projekt")
        form_layout.addRow("Projektname:", self.name_input)

        # Path
        path_layout = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("C:/Projekte/...")
        self.path_input.setReadOnly(True)  # Path should be selected via browse
        browse_btn = QPushButton("...")
        browse_btn.setFixedWidth(40)
        browse_btn.clicked.connect(self._browse_path)

        path_layout.addWidget(self.path_input)
        path_layout.addWidget(browse_btn)
        form_layout.addRow("Speicherort:", path_layout)

        # Description
        self.desc_input = QLineEdit()
        self.desc_input.setPlaceholderText("Optionale Beschreibung")
        form_layout.addRow("Beschreibung:", self.desc_input)

        # Settings Group
        layout.addLayout(form_layout)

        # Tech Specs
        tech_layout = QHBoxLayout()

        # FPS
        fps_layout = QVBoxLayout()
        fps_layout.addWidget(QLabel("Target FPS:"))
        self.fps_combo = QComboBox()
        self.fps_combo.addItems(["24", "25", "30", "60"])
        self.fps_combo.setCurrentText("30")
        fps_layout.addWidget(self.fps_combo)
        tech_layout.addLayout(fps_layout)

        # Resolution
        res_layout = QVBoxLayout()
        res_layout.addWidget(QLabel("Auflösung:"))
        self.res_combo = QComboBox()
        self.res_combo.addItems(["1920x1080 (FHD)", "3840x2160 (4K)", "1280x720 (HD)"])
        res_layout.addWidget(self.res_combo)
        tech_layout.addLayout(res_layout)

        layout.addLayout(tech_layout)

        layout.addSpacing(20)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.cancel_btn = QPushButton("Abbrechen")
        self.cancel_btn.clicked.connect(self.reject)

        self.ok_btn = QPushButton("Projekt erstellen" if not self.project_to_edit else "Speichern")
        self.ok_btn.clicked.connect(self._validate_and_accept)
        self.ok_btn.setDefault(True)  # Enter key submits
        # Apply primary style to OK button (if theme manager supports it later)

        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.ok_btn)

        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def _browse_path(self):
        """Open directory browser."""
        directory = QFileDialog.getExistingDirectory(self, "Projektordner wählen")
        if directory:
            self.path_input.setText(directory)

    def _load_project_data(self, project):
        """Load data from existing project."""
        self.name_input.setText(project.name)
        self.path_input.setText(project.path)
        self.path_input.setEnabled(False)  # Path usually shouldn't change for existing project
        if project.description:
            self.desc_input.setText(project.description)

        # Set FPS
        idx = self.fps_combo.findText(str(project.target_fps))
        if idx >= 0:
            self.fps_combo.setCurrentIndex(idx)

        # Set Resolution
        res_str = f"{project.resolution_width}x{project.resolution_height}"
        # Matches partial string
        for i in range(self.res_combo.count()):
            if res_str in self.res_combo.itemText(i):
                self.res_combo.setCurrentIndex(i)
                break

    def _validate_and_accept(self):
        """Validate inputs and accept dialog."""
        name = self.name_input.text().strip()
        path = self.path_input.text().strip()

        if not name:
            QMessageBox.warning(self, "Fehler", "Bitte geben Sie einen Projektnamen ein.")
            self.name_input.setFocus()
            return

        if not path:
            QMessageBox.warning(self, "Fehler", "Bitte wählen Sie einen Speicherort.")
            return

        # Check for duplicates (only for new projects)
        if not self.project_to_edit:
            # Simple check against DB
            # Ideally this logic might reside in controller, but simplified here
            # Note: Path uniqueness is usually enforced by OS, logic duplicate by DB
            pass

        # Parse Resolution
        res_text = self.res_combo.currentText()  # e.g. "1920x1080 (FHD)"
        width, height = map(int, res_text.split(" ")[0].split("x"))

        self.result_data = {
            "name": name,
            "path": path,
            "description": self.desc_input.text().strip(),
            "target_fps": int(self.fps_combo.currentText()),
            "resolution_width": width,
            "resolution_height": height,
        }

        self.accept()

    def get_data(self):
        """Return the result data."""
        return self.result_data
