"""
Keyboard Shortcuts Manager für PB_studio GUI

Zentralisierte Verwaltung aller Keyboard-Shortcuts mit konfigurierbaren Key-Bindings.

Unterstützte Shortcuts:
- Datei-Operationen: Neu, Öffnen, Speichern, Beenden
- Bearbeitung: Undo, Redo
- Rendering: Start, Stop, Preview
- Navigation: Hilfe
- Debug: Log-Fenster öffnen

Features:
- Konfigurierbare Key-Bindings
- Auto-Dokumentation (Hilfe-Dialog)
- PyQt6 QShortcut Integration
- Conflict-Detection
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import QMessageBox, QWidget

logger = logging.getLogger(__name__)


@dataclass
class ShortcutDefinition:
    """
    Definition eines Keyboard-Shortcuts.

    Attributes:
        key: Key-Sequenz (z.B. "Ctrl+N", "F1")
        description: Beschreibung der Funktion
        category: Kategorie für Hilfe-Dialog (z.B. "File", "Edit")
        callback: Callback-Funktion die ausgeführt wird
    """

    key: str
    description: str
    category: str
    callback: Callable | None = None


class KeyboardShortcutManager:
    """
    Verwaltet alle Keyboard-Shortcuts für die Anwendung.

    Ermöglicht:
    - Zentrale Shortcut-Registrierung
    - Auto-Generierung von Hilfe-Dialog
    - Conflict-Detection
    - Dynamisches An-/Abschalten von Shortcuts

    Example:
        >>> manager = KeyboardShortcutManager(main_window)
        >>> manager.register_shortcut("render_start", "F5", "Start Rendering", "Render", start_render_callback)
        >>> manager.install_all()
        >>> manager.show_help()
    """

    def __init__(self, parent: QWidget):
        """
        Initialisiert Shortcut-Manager.

        Args:
            parent: Parent-Widget (normalerweise MainWindow)
        """
        self.parent = parent
        self.shortcuts: dict[str, ShortcutDefinition] = {}
        self.active_shortcuts: dict[str, QShortcut] = {}

        # Standard-Shortcuts definieren
        self._define_default_shortcuts()

        logger.info("KeyboardShortcutManager initialisiert")

    def _define_default_shortcuts(self):
        """
        Definiert Standard-Shortcuts.

        Diese werden später mit Callbacks verknüpft via register_shortcut().
        """
        # Datei-Operationen
        self.shortcuts["new_project"] = ShortcutDefinition(
            key="Ctrl+N", description="Neues Projekt erstellen", category="File"
        )
        self.shortcuts["open_project"] = ShortcutDefinition(
            key="Ctrl+O", description="Projekt öffnen", category="File"
        )
        self.shortcuts["save_project"] = ShortcutDefinition(
            key="Ctrl+S", description="Projekt speichern", category="File"
        )
        self.shortcuts["save_as"] = ShortcutDefinition(
            key="Ctrl+Shift+S", description="Projekt speichern als...", category="File"
        )
        self.shortcuts["quit"] = ShortcutDefinition(
            key="Ctrl+Q", description="Anwendung beenden", category="File"
        )

        # Bearbeitung
        self.shortcuts["undo"] = ShortcutDefinition(
            key="Ctrl+Z", description="Rückgängig", category="Edit"
        )
        self.shortcuts["redo"] = ShortcutDefinition(
            key="Ctrl+Y", description="Wiederherstellen", category="Edit"
        )

        # Rendering
        self.shortcuts["render_start"] = ShortcutDefinition(
            key="F5", description="Rendering starten", category="Render"
        )
        self.shortcuts["render_preview"] = ShortcutDefinition(
            key="F6", description="Preview generieren", category="Render"
        )
        self.shortcuts["render_stop"] = ShortcutDefinition(
            key="Esc", description="Rendering abbrechen", category="Render"
        )

        # Navigation & Hilfe
        self.shortcuts["help"] = ShortcutDefinition(
            key="F1", description="Hilfe anzeigen", category="Help"
        )
        self.shortcuts["shortcuts_help"] = ShortcutDefinition(
            key="Ctrl+H", description="Keyboard-Shortcuts anzeigen", category="Help"
        )

        # Debug
        self.shortcuts["toggle_log"] = ShortcutDefinition(
            key="F12", description="Log-Fenster anzeigen/verstecken", category="Debug"
        )

    def register_shortcut(
        self, name: str, key: str, description: str, category: str, callback: Callable
    ):
        """
        Registriert einen neuen oder überschreibt bestehenden Shortcut.

        Args:
            name: Eindeutiger Name (z.B. "render_start")
            key: Key-Sequenz (z.B. "F5", "Ctrl+N")
            description: Beschreibung für Hilfe-Dialog
            category: Kategorie (z.B. "File", "Render")
            callback: Funktion die ausgeführt wird
        """
        # Conflict-Detection
        existing_key = self._find_shortcut_by_key(key)
        if existing_key and existing_key != name:
            logger.warning(
                f"Shortcut-Konflikt: '{key}' bereits verwendet von '{existing_key}'. "
                f"Überschreibe mit '{name}'."
            )

        self.shortcuts[name] = ShortcutDefinition(
            key=key, description=description, category=category, callback=callback
        )

        logger.debug(f"Shortcut registriert: {name} → {key}")

    def _find_shortcut_by_key(self, key: str) -> str | None:
        """
        Findet Shortcut-Name anhand der Key-Sequenz.

        Args:
            key: Key-Sequenz (z.B. "F5")

        Returns:
            Shortcut-Name oder None
        """
        for name, shortcut in self.shortcuts.items():
            if shortcut.key == key:
                return name
        return None

    def install_all(self):
        """
        Installiert alle registrierten Shortcuts.

        Erstellt QShortcut-Objekte und verbindet sie mit Callbacks.
        """
        for name, definition in self.shortcuts.items():
            self.install_shortcut(name)

        logger.info(f"{len(self.shortcuts)} Shortcuts installiert")

    def install_shortcut(self, name: str):
        """
        Installiert einen einzelnen Shortcut.

        Args:
            name: Shortcut-Name
        """
        if name not in self.shortcuts:
            logger.error(f"Shortcut nicht gefunden: {name}")
            return

        definition = self.shortcuts[name]

        # Callback muss gesetzt sein
        if not definition.callback:
            logger.debug(f"Kein Callback für Shortcut '{name}', überspringe")
            return

        # QShortcut erstellen
        shortcut = QShortcut(QKeySequence(definition.key), self.parent)
        shortcut.activated.connect(definition.callback)

        # Speichern für späteres Deaktivieren
        self.active_shortcuts[name] = shortcut

        logger.debug(f"Shortcut installiert: {name} ({definition.key})")

    def uninstall_shortcut(self, name: str):
        """
        Deinstalliert einen Shortcut.

        Args:
            name: Shortcut-Name
        """
        if name in self.active_shortcuts:
            shortcut = self.active_shortcuts[name]
            shortcut.setEnabled(False)
            del self.active_shortcuts[name]
            logger.debug(f"Shortcut deinstalliert: {name}")

    def set_enabled(self, name: str, enabled: bool):
        """
        Aktiviert/Deaktiviert einen Shortcut.

        Args:
            name: Shortcut-Name
            enabled: True = aktivieren, False = deaktivieren
        """
        if name in self.active_shortcuts:
            self.active_shortcuts[name].setEnabled(enabled)
            logger.debug(f"Shortcut '{name}' enabled={enabled}")

    def show_help(self):
        """
        Zeigt Hilfe-Dialog mit allen Shortcuts.

        Gruppiert nach Kategorien.
        """
        # Gruppiere Shortcuts nach Kategorie
        categories: dict[str, list] = {}
        for name, definition in self.shortcuts.items():
            if definition.category not in categories:
                categories[definition.category] = []
            categories[definition.category].append(definition)

        # HTML-Tabelle erstellen
        html = "<h2>Keyboard Shortcuts</h2>"
        html += "<table border='1' cellpadding='5' cellspacing='0'>"
        html += "<tr><th>Kategorie</th><th>Shortcut</th><th>Beschreibung</th></tr>"

        for category in sorted(categories.keys()):
            shortcuts_list = categories[category]

            # Erste Zeile mit Kategorie-Namen
            first = shortcuts_list[0]
            html += "<tr>"
            html += f"<td rowspan='{len(shortcuts_list)}'><b>{category}</b></td>"
            html += f"<td><code>{first.key}</code></td>"
            html += f"<td>{first.description}</td>"
            html += "</tr>"

            # Weitere Zeilen ohne Kategorie-Namen
            for shortcut in shortcuts_list[1:]:
                html += "<tr>"
                html += f"<td><code>{shortcut.key}</code></td>"
                html += f"<td>{shortcut.description}</td>"
                html += "</tr>"

        html += "</table>"

        # Dialog anzeigen
        msg_box = QMessageBox(self.parent)
        msg_box.setWindowTitle("Keyboard Shortcuts")
        msg_box.setTextFormat(Qt.TextFormat.RichText)
        msg_box.setText(html)
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.exec()

        logger.debug("Shortcuts-Hilfe angezeigt")

    def get_shortcut_key(self, name: str) -> str | None:
        """
        Gibt Key-Sequenz für einen Shortcut zurück.

        Args:
            name: Shortcut-Name

        Returns:
            Key-Sequenz (z.B. "F5") oder None
        """
        if name in self.shortcuts:
            return self.shortcuts[name].key
        return None

    def list_shortcuts(self) -> dict[str, str]:
        """
        Gibt alle Shortcuts als Dictionary zurück.

        Returns:
            Dict[name, key]
        """
        return {name: s.key for name, s in self.shortcuts.items()}
