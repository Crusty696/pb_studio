"""
Undo/Redo System mit Command Pattern

Implementiert vollständiges Undo/Redo-System für PB_studio GUI-Operationen.

Command Pattern:
- Jede Operation ist ein Command-Objekt
- execute() führt Operation aus
- undo() macht Operation rückgängig
- redo() wiederholt rückgängig gemachte Operation

Features:
- Unbegrenzte Undo/Redo-Historie (konfigurierbar)
- Command-Gruppen für atomare Operationen
- Macro-Commands für Batch-Operations
- State-Tracking für Clean/Dirty-Status

Verwendung:
    >>> undo_manager = UndoManager()
    >>> command = AddClipCommand(clip_id=1, position=0.5)
    >>> undo_manager.execute(command)
    >>> undo_manager.undo()
    >>> undo_manager.redo()

Author: PB_studio Development Team
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class Command(ABC):
    """
    Abstract Base Class für Commands.

    Jede Operation (z.B. Clip hinzufügen, Parameter ändern) ist ein Command.
    """

    @abstractmethod
    def execute(self):
        """Führt Command aus."""
        pass

    @abstractmethod
    def undo(self):
        """Macht Command rückgängig."""
        pass

    def redo(self):
        """
        Wiederholt Command (nach Undo).

        Default-Implementierung ruft execute() auf.
        Override für optimierte Implementierung.
        """
        self.execute()

    def get_description(self) -> str:
        """
        Gibt Beschreibung des Commands zurück.

        Returns:
            Human-readable Beschreibung (z.B. "Clip hinzugefügt")
        """
        return self.__class__.__name__


class MacroCommand(Command):
    """
    Gruppiert mehrere Commands zu einem Macro.

    Ermöglicht atomare Ausführung von mehreren Operations.

    Example:
        >>> macro = MacroCommand([
        ...     AddClipCommand(clip_id=1),
        ...     SetParameterCommand("intensity", 0.8)
        ... ])
        >>> undo_manager.execute(macro)  # Führt beide aus
        >>> undo_manager.undo()  # Macht beide rückgängig
    """

    def __init__(self, commands: list[Command], description: str = "Macro"):
        """
        Initialisiert Macro-Command.

        Args:
            commands: Liste von Commands
            description: Beschreibung des Macros
        """
        self.commands = commands
        self.description = description

    def execute(self):
        """Führt alle Commands aus."""
        for command in self.commands:
            command.execute()

    def undo(self):
        """Macht alle Commands rückgängig (in umgekehrter Reihenfolge)."""
        for command in reversed(self.commands):
            command.undo()

    def redo(self):
        """Wiederholt alle Commands."""
        for command in self.commands:
            command.redo()

    def get_description(self) -> str:
        """Gibt Macro-Beschreibung zurück."""
        return f"{self.description} ({len(self.commands)} operations)"


class UndoManager:
    """
    Verwaltet Undo/Redo-Historie für Commands.

    Features:
    - Unbegrenzte Historie (oder limit setzen)
    - Clean/Dirty State-Tracking
    - Command-Beschreibungen für UI
    - Clear-Funktion für Historie-Reset

    Example:
        >>> manager = UndoManager(max_history=100)
        >>> manager.execute(command)
        >>> manager.can_undo()  # True
        >>> manager.undo()
        >>> manager.can_redo()  # True
        >>> manager.redo()
    """

    def __init__(self, max_history: int = 100):
        """
        Initialisiert UndoManager.

        Args:
            max_history: Maximale Anzahl Commands in Historie (0 = unbegrenzt)
        """
        self.max_history = max_history
        self.undo_stack: list[Command] = []
        self.redo_stack: list[Command] = []
        self.clean_state_index: int | None = 0  # Index für "saved" state

        logger.info(f"UndoManager initialisiert: max_history={max_history}")

    def execute(self, command: Command):
        """
        Führt Command aus und fügt es zur Undo-Historie hinzu.

        Args:
            command: Command zum Ausführen
        """
        try:
            command.execute()

            # Füge zu Undo-Stack hinzu
            self.undo_stack.append(command)

            # Clear Redo-Stack (neue Operation invalidiert Redo)
            self.redo_stack.clear()

            # Limit Historie wenn nötig
            if self.max_history > 0 and len(self.undo_stack) > self.max_history:
                removed = self.undo_stack.pop(0)
                logger.debug(
                    f"Undo-Historie voll, ältestes Command entfernt: {removed.get_description()}"
                )

                # Update clean_state_index
                if self.clean_state_index is not None:
                    self.clean_state_index -= 1
                    if self.clean_state_index < 0:
                        self.clean_state_index = None  # Clean state verloren

            logger.debug(f"Command ausgeführt: {command.get_description()}")

        except Exception as e:
            logger.error(f"Fehler beim Ausführen von Command: {e}", exc_info=True)
            raise

    def undo(self) -> bool:
        """
        Macht letztes Command rückgängig.

        Returns:
            True wenn erfolgreich, False wenn keine Undo-Operation möglich
        """
        if not self.can_undo():
            logger.debug("Undo nicht möglich: Stack leer")
            return False

        try:
            command = self.undo_stack.pop()
            command.undo()

            # Füge zu Redo-Stack hinzu
            self.redo_stack.append(command)

            logger.info(f"Undo ausgeführt: {command.get_description()}")
            return True

        except Exception as e:
            logger.error(f"Fehler beim Undo: {e}", exc_info=True)
            # Command zurück auf Stack wenn Fehler
            self.undo_stack.append(command)
            return False

    def redo(self) -> bool:
        """
        Wiederholt zuletzt rückgängig gemachtes Command.

        Returns:
            True wenn erfolgreich, False wenn keine Redo-Operation möglich
        """
        if not self.can_redo():
            logger.debug("Redo nicht möglich: Stack leer")
            return False

        try:
            command = self.redo_stack.pop()
            command.redo()

            # Füge zurück zu Undo-Stack
            self.undo_stack.append(command)

            logger.info(f"Redo ausgeführt: {command.get_description()}")
            return True

        except Exception as e:
            logger.error(f"Fehler beim Redo: {e}", exc_info=True)
            # Command zurück auf Redo-Stack wenn Fehler
            self.redo_stack.append(command)
            return False

    def can_undo(self) -> bool:
        """
        Prüft ob Undo möglich ist.

        Returns:
            True wenn Undo-Stack nicht leer
        """
        return len(self.undo_stack) > 0

    def can_redo(self) -> bool:
        """
        Prüft ob Redo möglich ist.

        Returns:
            True wenn Redo-Stack nicht leer
        """
        return len(self.redo_stack) > 0

    def get_undo_description(self) -> str | None:
        """
        Gibt Beschreibung des nächsten Undo-Commands zurück.

        Returns:
            Beschreibung oder None
        """
        if self.can_undo():
            return self.undo_stack[-1].get_description()
        return None

    def get_redo_description(self) -> str | None:
        """
        Gibt Beschreibung des nächsten Redo-Commands zurück.

        Returns:
            Beschreibung oder None
        """
        if self.can_redo():
            return self.redo_stack[-1].get_description()
        return None

    def clear(self):
        """Leert komplette Undo/Redo-Historie."""
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.clean_state_index = 0
        logger.info("Undo/Redo-Historie geleert")

    def mark_clean(self):
        """
        Markiert aktuellen State als "clean" (gespeichert).

        Verwendet für Dirty-Flag in GUI (zeigt * bei unsaved changes).
        """
        self.clean_state_index = len(self.undo_stack)
        logger.debug(f"Clean state markiert bei Index {self.clean_state_index}")

    def is_clean(self) -> bool:
        """
        Prüft ob aktueller State "clean" ist (keine unsaved changes).

        Returns:
            True wenn State = Clean State
        """
        if self.clean_state_index is None:
            return False
        return len(self.undo_stack) == self.clean_state_index

    def get_history_info(self) -> dict:
        """
        Gibt Info über aktuelle Historie zurück.

        Returns:
            Dict mit undo_count, redo_count, is_clean
        """
        return {
            "undo_count": len(self.undo_stack),
            "redo_count": len(self.redo_stack),
            "is_clean": self.is_clean(),
            "undo_descriptions": [cmd.get_description() for cmd in self.undo_stack[-5:]],  # Last 5
            "redo_descriptions": [cmd.get_description() for cmd in self.redo_stack[-5:]],  # Last 5
        }


# ================================================================================
# Example Commands fuer PB_studio
# ================================================================================


class TimelineInterface(Protocol):
    """Minimale Schnittstelle des TimelineWidgets fuer Undo/Redo."""

    def add_clip(
        self,
        clip_data: dict,
        start_time: float,
        duration: float | None = None,
        snap_to_beat: bool = True,
    ) -> bool: ...

    def remove_clip(self, clip_id: int | str) -> bool: ...

    def get_clip(self, clip_id: int | str) -> Any: ...


class ParameterDashboardInterface(Protocol):
    """Minimalinterface fuer das ParameterDashboard."""

    def set_parameter_value(self, name: str, value: object, emit: bool = True) -> bool: ...

    def get_parameter_value(self, name: str) -> object | None: ...


@dataclass
class AddClipCommand(Command):
    """Command zum Hinzufuegen eines Clips auf der Timeline."""

    timeline: TimelineInterface
    clip_id: int | str
    position: float
    clip_data: dict
    duration: float | None = None
    snap_to_beat: bool = True

    _executed: bool = False

    def execute(self):
        if not self.timeline:
            logger.warning("AddClipCommand: Kein Timeline-Widget gesetzt.")
            return

        success = self.timeline.add_clip(
            self.clip_data,
            start_time=self.position,
            duration=self.duration,
            snap_to_beat=self.snap_to_beat,
        )
        self._executed = bool(success)
        if not success:
            logger.warning(f"AddClipCommand: Hinzufuegen von Clip {self.clip_id} fehlgeschlagen.")

    def undo(self):
        if not self.timeline or not self._executed:
            return
        removed = self.timeline.remove_clip(self.clip_id)
        if not removed:
            logger.warning(
                f"AddClipCommand Undo: Entfernen von Clip {self.clip_id} fehlgeschlagen."
            )

    def get_description(self) -> str:
        return f"Clip {self.clip_id} hinzugefuegt"


@dataclass
class RemoveClipCommand(Command):
    """Command zum Entfernen eines Clips von der Timeline."""

    timeline: TimelineInterface
    clip_id: int | str
    position: float  # Nur fuer Logging/Referenz

    _backup_clip_data: dict | None = None
    _backup_duration: float | None = None
    _backup_start: float | None = None

    def execute(self):
        if not self.timeline:
            logger.warning("RemoveClipCommand: Kein Timeline-Widget gesetzt.")
            return

        clip_obj = None
        getter = getattr(self.timeline, "get_clip", None)
        if callable(getter):
            clip_obj = getter(self.clip_id)

        if clip_obj:
            self._backup_start = getattr(clip_obj, "start_time", self.position)
            self._backup_duration = getattr(clip_obj, "duration", None)
            base_data = getattr(clip_obj, "clip_data", {}) or {}
            if not base_data.get("id"):
                base_data = {**base_data, "id": getattr(clip_obj, "clip_id", self.clip_id)}
            if "file_path" not in base_data and hasattr(clip_obj, "file_path"):
                base_data = {**base_data, "file_path": clip_obj.file_path}
            if "duration" not in base_data and self._backup_duration is not None:
                base_data = {**base_data, "duration": self._backup_duration}
            self._backup_clip_data = base_data

        success = self.timeline.remove_clip(self.clip_id)
        if not success:
            logger.warning(f"RemoveClipCommand: Entfernen von Clip {self.clip_id} fehlgeschlagen.")

    def undo(self):
        if not self.timeline:
            return
        if not self._backup_clip_data:
            logger.warning(
                f"RemoveClipCommand Undo: Keine gesicherten Daten fuer Clip {self.clip_id}."
            )
            return
        self.timeline.add_clip(
            self._backup_clip_data,
            start_time=self._backup_start or self.position,
            duration=self._backup_clip_data.get("duration", self._backup_duration),
            snap_to_beat=False,  # exakt an gleicher Stelle wiederherstellen
        )

    def get_description(self) -> str:
        return f"Clip {self.clip_id} entfernt"


@dataclass
class SetParameterCommand(Command):
    """
    Command zum Aendern eines Parameters (Dashboard-integriert).

    Verhindert Signal-Loops durch emit=False bei Undo/Redo.
    UI wird automatisch durch set_parameter_value() aktualisiert.
    """

    dashboard: ParameterDashboardInterface
    parameter_name: str
    new_value: object
    old_value: object | None = None

    def execute(self):
        """
        Fuehrt Parameteraenderung aus.

        Speichert automatisch den alten Wert beim ersten Ausfuehren.
        Verwendet emit=True fuer normale UI-Interaktion.
        """
        if not self.dashboard:
            logger.warning("SetParameterCommand: Kein ParameterDashboard gesetzt.")
            return

        # Speichere alten Wert beim ersten Ausfuehren
        if self.old_value is None:
            self.old_value = self.dashboard.get_parameter_value(self.parameter_name)
            logger.debug(
                f"SetParameterCommand: Gespeichert {self.parameter_name}: {self.old_value} -> {self.new_value}"
            )

        # Setze neuen Wert mit emit=True (normale UI-Interaktion)
        updated = self.dashboard.set_parameter_value(self.parameter_name, self.new_value, emit=True)
        if not updated:
            logger.warning(f"SetParameterCommand: Setzen von {self.parameter_name} fehlgeschlagen.")

    def undo(self):
        """
        Macht Parameteraenderung rueckgaengig.

        Verwendet emit=False um Signal-Loops zu verhindern.
        Dashboard blockiert intern Signals via blockSignals().
        """
        if not self.dashboard:
            return
        if self.old_value is None:
            logger.warning(
                f"SetParameterCommand Undo: Kein alter Wert fuer {self.parameter_name} vorhanden."
            )
            return

        # Setze alten Wert OHNE Signal-Emission (verhindert Loops)
        reverted = self.dashboard.set_parameter_value(
            self.parameter_name, self.old_value, emit=False
        )
        if not reverted:
            logger.warning(
                f"SetParameterCommand Undo: Ruecksetzen von {self.parameter_name} fehlgeschlagen."
            )
        else:
            logger.debug(f"SetParameterCommand Undo: {self.parameter_name} -> {self.old_value}")

    def redo(self):
        """
        Wiederholt Parameteraenderung.

        Verwendet emit=False um Signal-Loops zu verhindern.
        """
        if not self.dashboard:
            return

        # Setze neuen Wert OHNE Signal-Emission (verhindert Loops)
        updated = self.dashboard.set_parameter_value(
            self.parameter_name, self.new_value, emit=False
        )
        if not updated:
            logger.warning(
                f"SetParameterCommand Redo: Setzen von {self.parameter_name} fehlgeschlagen."
            )
        else:
            logger.debug(f"SetParameterCommand Redo: {self.parameter_name} -> {self.new_value}")

    def get_description(self) -> str:
        """Gibt Beschreibung des Commands zurueck."""
        return f"{self.parameter_name} geaendert"
