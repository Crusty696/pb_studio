"""
Dialog for selecting existing projects.
"""


from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from ...database.crud import delete_project, get_all_projects, get_project
from ...utils.logger import get_logger

logger = get_logger(__name__)


class ProjectSelector(QDialog):
    """Dialog to select a project from the database."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_project = None
        self.projects = []

        self.setWindowTitle("Projekt öffnen")
        self.setMinimumSize(600, 400)
        self.setModal(True)

        self._init_ui()
        self._load_projects()

    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout()

        # Header / Filter
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("Suche:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Projektname filtern...")
        self.search_input.textChanged.connect(self._filter_projects)
        header_layout.addWidget(self.search_input)
        layout.addLayout(header_layout)

        # Project List
        self.project_list = QListWidget()
        self.project_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.project_list.itemDoubleClicked.connect(self._on_double_click)
        self.project_list.itemSelectionChanged.connect(self._update_buttons)
        layout.addWidget(self.project_list)

        # Buttons
        btn_layout = QHBoxLayout()

        self.delete_btn = QPushButton("Löschen")
        self.delete_btn.setStyleSheet("color: #ff4444;")  # Simple warning color
        self.delete_btn.clicked.connect(self._delete_project)
        self.delete_btn.setEnabled(False)

        btn_layout.addWidget(self.delete_btn)
        btn_layout.addStretch()

        self.cancel_btn = QPushButton("Abbrechen")
        self.cancel_btn.clicked.connect(self.reject)

        self.open_btn = QPushButton("Öffnen")
        self.open_btn.clicked.connect(self._accept_selection)
        self.open_btn.setEnabled(False)
        self.open_btn.setDefault(True)

        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.open_btn)

        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def _load_projects(self):
        """Load projects from database."""
        try:
            self.projects = get_all_projects()
            self._update_list()
        except Exception as e:
            logger.error(f"Failed to load projects: {e}")
            QMessageBox.critical(self, "Fehler", f"Fehler beim Laden der Projekte: {e}")

    def _update_list(self):
        """Update list widget with current projects (filtered)."""
        self.project_list.clear()
        filter_text = self.search_input.text().lower()

        for project in self.projects:
            if filter_text and filter_text not in project.name.lower():
                continue

            last_modified = (
                project.modified_at.strftime("%Y-%m-%d %H:%M") if project.modified_at else "Nie"
            )

            item_text = (
                f"{project.name} \n   Path: {project.path}\n   Last Modified: {last_modified}"
            )
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, project.id)  # Store ID
            self.project_list.addItem(item)

        self._update_buttons()

    def _filter_projects(self):
        """Filter list based on search input."""
        self._update_list()

    def _update_buttons(self):
        """Update button states based on selection."""
        has_selection = len(self.project_list.selectedItems()) > 0
        self.open_btn.setEnabled(has_selection)
        self.delete_btn.setEnabled(has_selection)

    def _delete_project(self):
        """Delete selected project."""
        items = self.project_list.selectedItems()
        if not items:
            return

        project_id = items[0].data(Qt.ItemDataRole.UserRole)
        # Fetch project object for details (name) - optional, use cache
        project_name = items[0].text().split("\n")[0].strip()

        reply = QMessageBox.question(
            self,
            "Projekt löschen",
            f"Möchten Sie das Projekt '{project_name}' wirklich löschen?\n\n"
            "Dies löscht alle Datenbank-Einträge.\n"
            "Sollen auch die Dateien auf der Festplatte entfernt werden?",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )

        if reply == QMessageBox.StandardButton.Cancel:
            return

        delete_files = reply == QMessageBox.StandardButton.Yes

        if delete_project(project_id, delete_files=delete_files):
            # Refresh list
            self._load_projects()
        else:
            QMessageBox.critical(self, "Fehler", "Projekt konnte nicht gelöscht werden.")

    def _on_double_click(self, item):
        """Handle double click to open."""
        self._accept_selection()

    def _accept_selection(self):
        """Accept selected project."""
        items = self.project_list.selectedItems()
        if not items:
            return

        project_id = items[0].data(Qt.ItemDataRole.UserRole)
        self.selected_project = get_project(project_id)

        if self.selected_project:
            self.accept()
        else:
            QMessageBox.critical(
                self, "Fehler", "Projekt konnte nicht geladen werden (ID nicht gefunden)."
            )

    def get_selected_project(self):
        """Return the selected project object."""
        return self.selected_project
