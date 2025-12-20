"""
Analysis Progress Dialog - Zeigt Fortschritt der Video-Analyse.

Zeigt:
- Gesamtfortschritt
- Aktuell analysierter Clip
- Einzelne Analyse-Schritte
- Abbruch-Moeglichkeit
"""

from PyQt6.QtCore import QObject, QThread
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtCore import pyqtSlot as Slot
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from ..analysis import VideoAnalyzer
from ..utils.logger import get_logger

logger = get_logger()


class AnalysisWorker(QObject):
    """Worker fuer Hintergrund-Analyse."""

    # Signals
    progress = Signal(int, int, str)  # current, total, current_file
    step_progress = Signal(str)  # analysis_step
    finished = Signal(dict)  # stats
    error = Signal(str)

    def __init__(self, clip_ids: list[int], analyzer: VideoAnalyzer):
        super().__init__()
        self.clip_ids = clip_ids
        self.analyzer = analyzer
        self._stop_requested = False

    def stop(self):
        """Stoppt die Analyse."""
        self._stop_requested = True

    @Slot()
    def run(self):
        """Fuehrt die Analyse durch."""
        try:
            stats = {"total": len(self.clip_ids), "analyzed": 0, "failed": 0, "skipped": 0}

            for i, clip_id in enumerate(self.clip_ids):
                if self._stop_requested:
                    logger.info("Analyse abgebrochen")
                    break

                try:
                    # Clip-Info fuer Progress-Anzeige
                    clip_name = f"Clip {clip_id}"

                    self.progress.emit(i + 1, len(self.clip_ids), clip_name)

                    # Analyse durchfuehren
                    result = self.analyzer.analyze_clip(clip_id, save_to_db=True)

                    if result:
                        stats["analyzed"] += 1
                    else:
                        stats["skipped"] += 1

                except Exception as e:
                    logger.error(f"Analyse-Fehler bei Clip {clip_id}: {e}")
                    stats["failed"] += 1

            self.finished.emit(stats)

        except Exception as e:
            logger.error(f"Analyse-Worker Fehler: {e}")
            self.error.emit(str(e))


class AnalysisProgressDialog(QDialog):
    """Dialog fuer Analyse-Fortschritt."""

    def __init__(self, clip_ids: list[int], parent=None):
        """
        Args:
            clip_ids: Liste der zu analysierenden Clip-IDs
            parent: Parent Widget
        """
        super().__init__(parent)
        self.clip_ids = clip_ids
        self.analyzer = VideoAnalyzer()

        self._worker = None
        self._thread = None
        self._is_running = False

        self._setup_ui()

    def _setup_ui(self):
        """Erstellt die UI."""
        self.setWindowTitle("Video-Analyse")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        self.setModal(True)

        layout = QVBoxLayout(self)

        # Header
        header = QLabel(f"Analysiere {len(self.clip_ids)} Clips")
        font = QFont("Arial", 12)
        font.setBold(True)
        header.setFont(font)
        layout.addWidget(header)

        # Gesamtfortschritt
        progress_group = QGroupBox("Fortschritt")
        progress_layout = QVBoxLayout(progress_group)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(len(self.clip_ids))
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        # Status-Labels
        status_layout = QHBoxLayout()
        self.current_label = QLabel("Warte auf Start...")
        self.count_label = QLabel("0 / " + str(len(self.clip_ids)))
        status_layout.addWidget(self.current_label)
        status_layout.addStretch()
        status_layout.addWidget(self.count_label)
        progress_layout.addLayout(status_layout)

        layout.addWidget(progress_group)

        # Analyse-Schritte
        steps_group = QGroupBox("Analyse-Schritte")
        steps_layout = QVBoxLayout(steps_group)

        self.step_labels = {}
        steps = ["Farben", "Bewegung", "Szene", "Stimmung", "Objekte", "Style", "Features"]
        for step in steps:
            step_frame = QFrame()
            step_frame.setFrameShape(QFrame.Shape.NoFrame)
            step_hl = QHBoxLayout(step_frame)
            step_hl.setContentsMargins(0, 2, 0, 2)

            label = QLabel(step)
            status = QLabel("⏳")  # Wartend
            step_hl.addWidget(label)
            step_hl.addStretch()
            step_hl.addWidget(status)

            self.step_labels[step] = status
            steps_layout.addWidget(step_frame)

        layout.addWidget(steps_group)

        # Log-Ausgabe
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.start_button = QPushButton("Starten")
        self.start_button.clicked.connect(self._start_analysis)
        button_layout.addWidget(self.start_button)

        self.cancel_button = QPushButton("Abbrechen")
        self.cancel_button.clicked.connect(self._cancel_analysis)
        self.cancel_button.setEnabled(False)
        button_layout.addWidget(self.cancel_button)

        self.close_button = QPushButton("Schliessen")
        self.close_button.clicked.connect(self.close)
        self.close_button.setVisible(False)
        button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)

    def _start_analysis(self):
        """Startet die Analyse."""
        self._is_running = True
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)

        self._log("Starte Analyse...")

        # Worker und Thread erstellen
        self._thread = QThread()
        self._worker = AnalysisWorker(self.clip_ids, self.analyzer)
        self._worker.moveToThread(self._thread)

        # Signals verbinden
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.step_progress.connect(self._on_step_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self._thread.start()

    def _cancel_analysis(self):
        """Bricht die Analyse ab."""
        if self._worker:
            self._worker.stop()
            self._log("Abbruch angefordert...")
            self.cancel_button.setEnabled(False)

    @Slot(int, int, str)
    def _on_progress(self, current: int, total: int, current_file: str):
        """Aktualisiert den Fortschritt."""
        self.progress_bar.setValue(current)
        self.count_label.setText(f"{current} / {total}")
        self.current_label.setText(current_file)

        # Schritte zuruecksetzen
        for label in self.step_labels.values():
            label.setText("⏳")

        self._log(f"Analysiere: {current_file}")

    @Slot(str)
    def _on_step_progress(self, step: str):
        """Aktualisiert Analyse-Schritt."""
        if step in self.step_labels:
            self.step_labels[step].setText("✅")

    @Slot(dict)
    def _on_finished(self, stats: dict):
        """Analyse abgeschlossen."""
        self._is_running = False
        self._cleanup_thread()

        # UI aktualisieren
        self.cancel_button.setVisible(False)
        self.close_button.setVisible(True)

        # Statistiken anzeigen
        self._log("")
        self._log("=" * 40)
        self._log("Analyse abgeschlossen!")
        self._log(f"Analysiert: {stats.get('analyzed', 0)}")
        self._log(f"Uebersprungen: {stats.get('skipped', 0)}")
        self._log(f"Fehler: {stats.get('failed', 0)}")

        self.current_label.setText("Abgeschlossen")

        # Alle Schritte als erledigt markieren
        for label in self.step_labels.values():
            label.setText("✅")

    @Slot(str)
    def _on_error(self, error: str):
        """Fehler aufgetreten."""
        self._is_running = False
        self._cleanup_thread()

        self._log(f"FEHLER: {error}")
        self.cancel_button.setEnabled(False)
        self.close_button.setVisible(True)

    def _cleanup_thread(self):
        """Raeumt Thread auf."""
        if self._thread:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._worker = None

    def _log(self, message: str):
        """Fuegt Log-Nachricht hinzu."""
        self.log_text.append(message)

    def closeEvent(self, event):
        """Wird beim Schliessen aufgerufen."""
        if self._is_running:
            self._cancel_analysis()
            # Warten bis Worker fertig
            if self._thread:
                self._thread.quit()
                self._thread.wait(5000)  # Max 5s warten

        super().closeEvent(event)


class QuickAnalysisDialog(QDialog):
    """Schneller Dialog fuer Einzelclip-Analyse."""

    def __init__(self, clip_id: int, clip_name: str = "", parent=None):
        super().__init__(parent)
        self.clip_id = clip_id
        self.clip_name = clip_name or f"Clip {clip_id}"

        self._setup_ui()
        self._run_analysis()

    def _setup_ui(self):
        """Erstellt die UI."""
        self.setWindowTitle("Schnell-Analyse")
        self.setMinimumWidth(300)
        self.setModal(True)

        layout = QVBoxLayout(self)

        self.status_label = QLabel(f"Analysiere {self.clip_name}...")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        layout.addWidget(self.progress_bar)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setVisible(False)
        layout.addWidget(self.result_text)

        self.close_button = QPushButton("Schliessen")
        self.close_button.clicked.connect(self.close)
        self.close_button.setVisible(False)
        layout.addWidget(self.close_button)

    def _run_analysis(self):
        """Fuehrt Analyse im Hintergrund durch."""
        self._thread = QThread()
        self._worker = QuickAnalysisWorker(self.clip_id)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self._thread.start()

    @Slot(dict)
    def _on_finished(self, result: dict):
        """Analyse abgeschlossen."""
        self._thread.quit()
        self._thread.wait()

        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.status_label.setText("Analyse abgeschlossen!")

        # Ergebnis anzeigen
        self.result_text.setVisible(True)
        self.close_button.setVisible(True)

        text = []
        if "colors" in result:
            text.append(
                f"Farben: {result['colors'].get('temperature', 'N/A')}, "
                f"Helligkeit: {result['colors'].get('brightness', 'N/A')}"
            )
        if "motion" in result:
            text.append(f"Bewegung: {result['motion'].get('motion_type', 'N/A')}")
        if "scene" in result:
            text.append(f"Szene: {', '.join(result['scene'].get('scene_types', []))}")
        if "mood" in result:
            text.append(f"Stimmung: {', '.join(result['mood'].get('moods', []))}")
        if "style" in result:
            text.append(f"Style: {', '.join(result['style'].get('styles', []))}")

        self.result_text.setText("\n".join(text) if text else "Keine Ergebnisse")

    @Slot(str)
    def _on_error(self, error: str):
        """Fehler aufgetreten."""
        self._thread.quit()
        self._thread.wait()

        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Fehler: {error}")
        self.close_button.setVisible(True)


class QuickAnalysisWorker(QObject):
    """Worker fuer Schnell-Analyse."""

    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, clip_id: int):
        super().__init__()
        self.clip_id = clip_id

    @Slot()
    def run(self):
        """Fuehrt Analyse durch."""
        try:
            analyzer = VideoAnalyzer()
            result = analyzer.analyze_clip(self.clip_id, save_to_db=True)
            self.finished.emit(result or {})
        except Exception as e:
            self.error.emit(str(e))


# =============================================================================
# STEM SEPARATION DIALOG (2025-12-08)
# Progress-Dialog fuer Stem-basierte Audio-Analyse
# =============================================================================


class StemSeparationWorker(QObject):
    """Worker fuer Stem-Separation im Hintergrund."""

    # Signals
    stage_changed = Signal(str, float)  # stage_name, progress (0-1)
    finished = Signal(dict)  # result dict
    error = Signal(str)

    def __init__(self, audio_path: str, expected_bpm: float = None):
        super().__init__()
        self.audio_path = audio_path
        self.expected_bpm = expected_bpm
        self._stop_requested = False

    def stop(self):
        """Stoppt die Analyse."""
        self._stop_requested = True

    @Slot()
    def run(self):
        """Fuehrt Stem-Analyse durch."""
        try:
            from ..pacing.trigger_system import TriggerSystem

            # TriggerSystem mit Stems initialisieren
            trigger_system = TriggerSystem(use_stems=True)

            if not trigger_system.is_stem_analysis_available():
                self.error.emit(
                    "Stem-Analyse nicht verfuegbar. Installiere: pip install audio-separator[cpu]"
                )
                return

            def progress_callback(stage: str, progress: float):
                if self._stop_requested:
                    return
                self.stage_changed.emit(stage, progress)

            # Stem-basierte Analyse
            result = trigger_system.analyze_triggers_with_stems(
                self.audio_path, expected_bpm=self.expected_bpm, progress_callback=progress_callback
            )

            if self._stop_requested:
                self.error.emit("Abgebrochen")
                return

            # Konvertiere zu dict fuer Signal
            result_dict = {
                "bpm": result.bpm,
                "duration": result.duration,
                "beat_count": len(result.beat_times),
                "kick_count": len(result.kick_times),
                "snare_count": len(result.snare_times),
                "hihat_count": len(result.hihat_times),
                "energy_count": len(result.energy_times),
            }

            self.finished.emit(result_dict)

        except Exception as e:
            logger.error(f"Stem-Analyse Fehler: {e}", exc_info=True)
            self.error.emit(str(e))


class StemSeparationDialog(QDialog):
    """
    Dialog fuer Stem-basierte Audio-Analyse.

    Zeigt Fortschritt der einzelnen Phasen mit individuellen Balken pro Stem:
    - Stem-Separation (je ein Balken fuer Drums, Bass, Melody)
    - Stem-Analyse (je ein Balken fuer Drums, Bass, Melody)
    - Trigger-Merge
    """

    # Signal wenn Analyse fertig
    analysis_complete = Signal(dict)
    stage_progress = Signal(str, float)
    analysis_failed = Signal(str)

    def __init__(self, audio_path: str, expected_bpm: float = None, parent=None):
        """
        Args:
            audio_path: Pfad zur Audio-Datei
            expected_bpm: Erwartete BPM (optional)
            parent: Parent Widget
        """
        super().__init__(parent)
        self.audio_path = audio_path
        self.expected_bpm = expected_bpm

        self._worker = None
        self._thread = None
        self._is_running = False

        self._setup_ui()

    def _setup_ui(self):
        """Erstellt die UI."""
        self.setWindowTitle("Stem-basierte Audio-Analyse")
        self.setMinimumWidth(550)
        self.setMinimumHeight(550)
        self.setModal(True)

        layout = QVBoxLayout(self)

        # Header
        header = QLabel("Stem-basierte Trigger-Analyse")
        font = QFont("Arial", 12)
        font.setBold(True)
        header.setFont(font)
        layout.addWidget(header)

        # Info
        info_label = QLabel(
            "Trennt Audio in Drums/Bass/Melody für präzisere Trigger-Erkennung.\n"
            "Ideal für EDM/DJ-Mixes. Erste Analyse kann 20-30 Min dauern (wird gecacht)."
        )
        info_label.setStyleSheet("color: #888888; font-size: 10px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        self.phase_bars = {}
        self.phase_labels = {}

        # ========================================
        # STEM SEPARATION GROUP (individuelle Balken)
        # ========================================
        sep_group = QGroupBox("1. Stem-Separation")
        sep_layout = QVBoxLayout(sep_group)

        stem_phases = [
            ("drums_sep", "🥁 Drums-Stem", "Extrahiert Drums (Kick, Snare, HiHat)"),
            ("bass_sep", "🎸 Bass-Stem", "Extrahiert Bass-Frequenzen"),
            ("other_sep", "🎹 Melody-Stem", "Extrahiert Synths, Melodien, Hooks"),
        ]

        for phase_id, phase_name, phase_desc in stem_phases:
            self._add_phase_bar(sep_layout, phase_id, phase_name, phase_desc)

        layout.addWidget(sep_group)

        # ========================================
        # STEM ANALYSE GROUP
        # ========================================
        analysis_group = QGroupBox("2. Stem-Analyse")
        analysis_layout = QVBoxLayout(analysis_group)

        analysis_phases = [
            ("drums", "🥁 Drums-Analyse", "Erkennt Kick, Snare, HiHat Events"),
            ("bass", "🎸 Bass-Analyse", "Erkennt Bass-Drops und Sub-Bass"),
            ("melody", "🎹 Melody-Analyse", "Erkennt Synth-Stabs und Hooks"),
        ]

        for phase_id, phase_name, phase_desc in analysis_phases:
            self._add_phase_bar(analysis_layout, phase_id, phase_name, phase_desc)

        layout.addWidget(analysis_group)

        # ========================================
        # MERGE GROUP
        # ========================================
        merge_group = QGroupBox("3. Zusammenführung")
        merge_layout = QVBoxLayout(merge_group)
        self._add_phase_bar(merge_layout, "merge", "🔀 Trigger-Merge", "Kombiniert alle Trigger")

        layout.addWidget(merge_group)

        # Ergebnis und Buttons
        self._setup_result_and_buttons(layout)

    def _add_phase_bar(self, layout, phase_id: str, phase_name: str, phase_desc: str):
        """Fuegt einen Phase-Fortschrittsbalken hinzu."""
        phase_frame = QFrame()
        phase_hl = QVBoxLayout(phase_frame)
        phase_hl.setContentsMargins(0, 3, 0, 3)

        label_row = QHBoxLayout()
        label = QLabel(phase_name)
        label.setToolTip(phase_desc)
        status = QLabel("⏳")
        label_row.addWidget(label)
        label_row.addStretch()
        label_row.addWidget(status)
        phase_hl.addLayout(label_row)

        # Progress Bar
        progress_bar = QProgressBar()
        progress_bar.setMaximum(100)
        progress_bar.setValue(0)
        progress_bar.setMaximumHeight(12)
        progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 1px solid #444;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 2px;
            }
        """
        )
        phase_hl.addWidget(progress_bar)

        self.phase_bars[phase_id] = progress_bar
        self.phase_labels[phase_id] = status

        layout.addWidget(phase_frame)

    def _setup_result_and_buttons(self, layout):
        """Erstellt Ergebnis-Anzeige und Buttons."""
        # Ergebnis-Anzeige
        result_group = QGroupBox("Ergebnis")
        result_layout = QVBoxLayout(result_group)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(100)
        self.result_text.setPlaceholderText("Warte auf Analyse...")
        result_layout.addWidget(self.result_text)

        layout.addWidget(result_group)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        # Start-Button versteckt - Analyse startet automatisch
        self.start_button = QPushButton("Analyse starten")
        self.start_button.clicked.connect(self._start_analysis)
        self.start_button.setVisible(False)  # Versteckt da automatischer Start
        button_layout.addWidget(self.start_button)

        self.cancel_button = QPushButton("Abbrechen")
        self.cancel_button.clicked.connect(self._cancel_analysis)
        self.cancel_button.setEnabled(True)  # Sofort aktiv für Auto-Start
        button_layout.addWidget(self.cancel_button)

        self.close_button = QPushButton("Schliessen")
        self.close_button.clicked.connect(self.close)
        self.close_button.setVisible(False)
        button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)

    def _start_analysis(self):
        """Startet die Stem-Analyse."""
        self._is_running = True
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)

        self.result_text.setText("Starte Analyse...\n")

        # Reset Progress
        for bar in self.phase_bars.values():
            bar.setValue(0)
        for label in self.phase_labels.values():
            label.setText("⏳")

        # Worker und Thread erstellen
        self._thread = QThread()
        self._worker = StemSeparationWorker(self.audio_path, self.expected_bpm)
        self._worker.moveToThread(self._thread)

        # Signals verbinden
        self._thread.started.connect(self._worker.run)
        self._worker.stage_changed.connect(self._on_stage_changed)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self._thread.start()

    def _cancel_analysis(self):
        """Bricht die Analyse ab."""
        if self._worker:
            self._worker.stop()
            self.result_text.append("Abbruch angefordert...")
            self.cancel_button.setEnabled(False)

    @Slot(str, float)
    def _on_stage_changed(self, stage: str, progress: float):
        """Aktualisiert Phase-Fortschritt."""
        if stage in self.phase_bars:
            self.phase_bars[stage].setValue(int(progress * 100))

            # Status-Icon aktualisieren
            if progress >= 1.0:
                self.phase_labels[stage].setText("✅")
            elif progress > 0:
                self.phase_labels[stage].setText("🔄")
            self.stage_progress.emit(stage, progress)

    @Slot(dict)
    def _on_finished(self, result: dict):
        """Analyse abgeschlossen."""
        self._is_running = False
        self._cleanup_thread()

        # UI aktualisieren
        self.cancel_button.setVisible(False)
        self.close_button.setVisible(True)

        # Alle Phasen als fertig markieren
        for label in self.phase_labels.values():
            label.setText("✅")
        for bar in self.phase_bars.values():
            bar.setValue(100)

        # Ergebnis anzeigen
        self.result_text.clear()
        self.result_text.append("✅ Analyse abgeschlossen!\n")
        self.result_text.append(f"BPM: {result.get('bpm', 0):.1f}")
        self.result_text.append(f"Dauer: {result.get('duration', 0):.1f}s")
        self.result_text.append("\nTrigger erkannt:")
        self.result_text.append(f"  • Beats: {result.get('beat_count', 0)}")
        self.result_text.append(f"  • Kicks (Stem): {result.get('kick_count', 0)}")
        self.result_text.append(f"  • Snares (Stem): {result.get('snare_count', 0)}")
        self.result_text.append(f"  • HiHats (Stem): {result.get('hihat_count', 0)}")
        self.result_text.append(f"  • Bass-Drops: {result.get('energy_count', 0)}")

        # Signal emittieren
        self.analysis_complete.emit(result)

    @Slot(str)
    def _on_error(self, error: str):
        """Fehler aufgetreten."""
        self._is_running = False
        self._cleanup_thread()

        self.result_text.append(f"\n❌ FEHLER: {error}")
        self.cancel_button.setEnabled(False)
        self.close_button.setVisible(True)
        self.analysis_failed.emit(error)

    def _cleanup_thread(self):
        """Raeumt Thread auf."""
        if self._thread:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._worker = None

    def showEvent(self, event):
        """Startet Analyse automatisch beim Anzeigen des Dialogs."""
        super().showEvent(event)
        # Starte Analyse automatisch nach kurzem Delay (für UI-Rendering)
        from PyQt6.QtCore import QTimer

        QTimer.singleShot(100, self._start_analysis)

    def closeEvent(self, event):
        """Wird beim Schliessen aufgerufen."""
        if self._is_running:
            self._cancel_analysis()
            if self._thread:
                self._thread.quit()
                self._thread.wait(5000)

        super().closeEvent(event)
