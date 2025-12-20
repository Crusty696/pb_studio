\# MASTERPLAN\_TASKS.md

\# --------------------------------------------------------------------------------

\# CODE-AUDIT FIXING PLAN: PB\_STUDIO

\# Datum: 2025-12-20

\# Anleitung: Kopiere jeweils einen "TASK BLOCK" vollständig und sende ihn an den AI-Agenten.

\# --------------------------------------------------------------------------------



\## 🔴 PRIORITÄT 1: KRITISCHE FEHLER (Sofort beheben)



\### TASK BLOCK 1: Audio Analyzer (Windows Crash)

\*\*ZIEL-DATEI:\*\* `src/pb\_studio/audio/audio\_analyzer.py`

\*\*FEHLER-DETAILS:\*\*

1\. Missing Import: `os` wird verwendet (Zeile 191), aber nur bedingt importiert (Zeile 201). Führt zum Crash auf Windows.



Wir bearbeiten nun den nächsten Task aus dem Masterplan.



Du arbeitest ab jetzt nach dem strengen \*\*"Surgical Fix Protocol"\*\*. Dein Ziel ist es, die dokumentierten Fehler zu beheben, OHNE die bestehende Struktur, den Stil oder funktionierende Teile des Codes unnötig zu verändern.



Führe für diese Datei exakt dieses 5-Schritte-Schema durch:



\### SCHRITT 1: Scope \& Isolation (Die Quarantäne)

\* Identifiziere exakt die Zeilen, die laut Fehlerbericht (aus Schritt 1) defekt sind.

\* \*\*REGEL:\*\* Ändere NICHTS, was nicht explizit als Fehler gemeldet wurde oder zwingend für den Fix notwendig ist. Kein "Schöner-Machen", kein unnötiges Refactoring.

\* Bestätige kurz: "Ich fokussiere mich nur auf \[Liste der Fehler]."



\### SCHRITT 2: Recherche \& Validierung

\* Falls externe Libraries (FFmpeg, Qt, NumPy) betroffen sind: Prüfe kurz intern, ob deine geplante Lösung mit der aktuellen Version kompatibel ist.

\* Falls du dir unsicher bist, recherchiere oder simuliere den Aufruf.



\### SCHRITT 3: Kontext-Check (Side-Effects)

\* Prüfe: Wenn ich diesen Fehler hier behebe, bricht das etwas in einer anderen Datei (z.B. Importe, Funktionsaufrufe)?

\* Passe die Lösung so an, dass sie sich nahtlos in den Rest der App einfügt.



\### SCHRITT 4: Die "Dry-Run" Simulation (Qualitäts-Schleife)

\* Simuliere gedanklich die Ausführung des neuen Codes.

\* \*Szenario:\* Wenn der Fehler behoben ist, läuft der Rest des Codes weiter?

\* Falls du ein Problem findest: Gehe zurück zu Schritt 2 und korrigiere deinen Ansatz, BEVOR du Code ausgibst.



\### SCHRITT 5: Finale Ausgabe

Erstelle den vollständigen Code der Datei.

\* Kommentiere NUR an den geänderten Stellen: `# FIX: \[Was wurde warum geändert]`.

\* Lasse den restlichen Code exakt so, wie er war (Copy-Paste-Ready).



\*\*START:\*\*

Wende dieses Schema jetzt auf die genannte Datei an. Beginne mit Schritt 1.



---



\### TASK BLOCK 2: Video Renderer (Timeouts \& Threading)

\*\*ZIEL-DATEI:\*\* `src/pb\_studio/video/video\_renderer.py`

\*\*FEHLER-DETAILS:\*\*

1\. FFmpeg Timeouts: Konstanten definiert, aber nicht in `ffmpeg.run` verwendet (Zeile 97-100, 1443).

2\. ThreadPoolExecutor: Wird ohne Context Manager verwendet, kein Shutdown bei Exceptions (Zeile 800-900).



Wir bearbeiten nun den nächsten Task aus dem Masterplan.



Du arbeitest ab jetzt nach dem strengen \*\*"Surgical Fix Protocol"\*\*. Dein Ziel ist es, die dokumentierten Fehler zu beheben, OHNE die bestehende Struktur, den Stil oder funktionierende Teile des Codes unnötig zu verändern.



Führe für diese Datei exakt dieses 5-Schritte-Schema durch:



\### SCHRITT 1: Scope \& Isolation (Die Quarantäne)

\* Identifiziere exakt die Zeilen, die laut Fehlerbericht (aus Schritt 1) defekt sind.

\* \*\*REGEL:\*\* Ändere NICHTS, was nicht explizit als Fehler gemeldet wurde oder zwingend für den Fix notwendig ist. Kein "Schöner-Machen", kein unnötiges Refactoring.

\* Bestätige kurz: "Ich fokussiere mich nur auf \[Liste der Fehler]."



\### SCHRITT 2: Recherche \& Validierung

\* Falls externe Libraries (FFmpeg, Qt, NumPy) betroffen sind: Prüfe kurz intern, ob deine geplante Lösung mit der aktuellen Version kompatibel ist.

\* Falls du dir unsicher bist, recherchiere oder simuliere den Aufruf.



\### SCHRITT 3: Kontext-Check (Side-Effects)

\* Prüfe: Wenn ich diesen Fehler hier behebe, bricht das etwas in einer anderen Datei (z.B. Importe, Funktionsaufrufe)?

\* Passe die Lösung so an, dass sie sich nahtlos in den Rest der App einfügt.



\### SCHRITT 4: Die "Dry-Run" Simulation (Qualitäts-Schleife)

\* Simuliere gedanklich die Ausführung des neuen Codes.

\* \*Szenario:\* Wenn der Fehler behoben ist, läuft der Rest des Codes weiter?

\* Falls du ein Problem findest: Gehe zurück zu Schritt 2 und korrigiere deinen Ansatz, BEVOR du Code ausgibst.



\### SCHRITT 5: Finale Ausgabe

Erstelle den vollständigen Code der Datei.

\* Kommentiere NUR an den geänderten Stellen: `# FIX: \[Was wurde warum geändert]`.

\* Lasse den restlichen Code exakt so, wie er war (Copy-Paste-Ready).



\*\*START:\*\*

Wende dieses Schema jetzt auf die genannte Datei an. Beginne mit Schritt 1.



---



\### TASK BLOCK 3: Video Manager (Zombie Prozesse \& Security)

\*\*ZIEL-DATEI:\*\* `src/pb\_studio/video/video\_manager.py`

\*\*FEHLER-DETAILS:\*\*

1\. Zombie Prozesse: `ffprobe` wird bei Timeout nicht gekillt (Zeile 75-93).

2\. Security: Path Injection möglich, Validierung fehlt vor subprocess (Zeile 50-70).



Wir bearbeiten nun den nächsten Task aus dem Masterplan.



Du arbeitest ab jetzt nach dem strengen \*\*"Surgical Fix Protocol"\*\*. Dein Ziel ist es, die dokumentierten Fehler zu beheben, OHNE die bestehende Struktur, den Stil oder funktionierende Teile des Codes unnötig zu verändern.



Führe für diese Datei exakt dieses 5-Schritte-Schema durch:



\### SCHRITT 1: Scope \& Isolation (Die Quarantäne)

\* Identifiziere exakt die Zeilen, die laut Fehlerbericht (aus Schritt 1) defekt sind.

\* \*\*REGEL:\*\* Ändere NICHTS, was nicht explizit als Fehler gemeldet wurde oder zwingend für den Fix notwendig ist. Kein "Schöner-Machen", kein unnötiges Refactoring.

\* Bestätige kurz: "Ich fokussiere mich nur auf \[Liste der Fehler]."



\### SCHRITT 2: Recherche \& Validierung

\* Falls externe Libraries (FFmpeg, Qt, NumPy) betroffen sind: Prüfe kurz intern, ob deine geplante Lösung mit der aktuellen Version kompatibel ist.

\* Falls du dir unsicher bist, recherchiere oder simuliere den Aufruf.



\### SCHRITT 3: Kontext-Check (Side-Effects)

\* Prüfe: Wenn ich diesen Fehler hier behebe, bricht das etwas in einer anderen Datei (z.B. Importe, Funktionsaufrufe)?

\* Passe die Lösung so an, dass sie sich nahtlos in den Rest der App einfügt.



\### SCHRITT 4: Die "Dry-Run" Simulation (Qualitäts-Schleife)

\* Simuliere gedanklich die Ausführung des neuen Codes.

\* \*Szenario:\* Wenn der Fehler behoben ist, läuft der Rest des Codes weiter?

\* Falls du ein Problem findest: Gehe zurück zu Schritt 2 und korrigiere deinen Ansatz, BEVOR du Code ausgibst.



\### SCHRITT 5: Finale Ausgabe

Erstelle den vollständigen Code der Datei.

\* Kommentiere NUR an den geänderten Stellen: `# FIX: \[Was wurde warum geändert]`.

\* Lasse den restlichen Code exakt so, wie er war (Copy-Paste-Ready).



\*\*START:\*\*

Wende dieses Schema jetzt auf die genannte Datei an. Beginne mit Schritt 1.



---



\### TASK BLOCK 4: Pacing Engine (Math Errors)

\*\*ZIEL-DATEI:\*\* `src/pb\_studio/pacing/pacing\_engine.py`

\*\*FEHLER-DETAILS:\*\*

1\. Division by Zero: `beat\_duration` Property crasht bei BPM=0 (Zeile 68).

2\. Float Vergleich: `cut.start\_time == start\_time` ist unzuverlässig (Zeile 303).



Wir bearbeiten nun den nächsten Task aus dem Masterplan.



Du arbeitest ab jetzt nach dem strengen \*\*"Surgical Fix Protocol"\*\*. Dein Ziel ist es, die dokumentierten Fehler zu beheben, OHNE die bestehende Struktur, den Stil oder funktionierende Teile des Codes unnötig zu verändern.



Führe für diese Datei exakt dieses 5-Schritte-Schema durch:



\### SCHRITT 1: Scope \& Isolation (Die Quarantäne)

\* Identifiziere exakt die Zeilen, die laut Fehlerbericht (aus Schritt 1) defekt sind.

\* \*\*REGEL:\*\* Ändere NICHTS, was nicht explizit als Fehler gemeldet wurde oder zwingend für den Fix notwendig ist. Kein "Schöner-Machen", kein unnötiges Refactoring.

\* Bestätige kurz: "Ich fokussiere mich nur auf \[Liste der Fehler]."



\### SCHRITT 2: Recherche \& Validierung

\* Falls externe Libraries (FFmpeg, Qt, NumPy) betroffen sind: Prüfe kurz intern, ob deine geplante Lösung mit der aktuellen Version kompatibel ist.

\* Falls du dir unsicher bist, recherchiere oder simuliere den Aufruf.



\### SCHRITT 3: Kontext-Check (Side-Effects)

\* Prüfe: Wenn ich diesen Fehler hier behebe, bricht das etwas in einer anderen Datei (z.B. Importe, Funktionsaufrufe)?

\* Passe die Lösung so an, dass sie sich nahtlos in den Rest der App einfügt.



\### SCHRITT 4: Die "Dry-Run" Simulation (Qualitäts-Schleife)

\* Simuliere gedanklich die Ausführung des neuen Codes.

\* \*Szenario:\* Wenn der Fehler behoben ist, läuft der Rest des Codes weiter?

\* Falls du ein Problem findest: Gehe zurück zu Schritt 2 und korrigiere deinen Ansatz, BEVOR du Code ausgibst.



\### SCHRITT 5: Finale Ausgabe

Erstelle den vollständigen Code der Datei.

\* Kommentiere NUR an den geänderten Stellen: `# FIX: \[Was wurde warum geändert]`.

\* Lasse den restlichen Code exakt so, wie er war (Copy-Paste-Ready).



\*\*START:\*\*

Wende dieses Schema jetzt auf die genannte Datei an. Beginne mit Schritt 1.



---



\### TASK BLOCK 5: Main Window (Threading Safety \& Memory)

\*\*ZIEL-DATEI:\*\* `src/pb\_studio/gui/main\_window.py`

\*\*FEHLER-DETAILS:\*\*

1\. Race Condition: `\_is\_cancelled` Flag ist nicht thread-safe (Zeile 155, 263).

2\. Memory Leak: `RenderWorker` hält starke Referenz zu Main, kein `deleteLater` (Zeile 150-270).



Wir bearbeiten nun den nächsten Task aus dem Masterplan.



Du arbeitest ab jetzt nach dem strengen \*\*"Surgical Fix Protocol"\*\*. Dein Ziel ist es, die dokumentierten Fehler zu beheben, OHNE die bestehende Struktur, den Stil oder funktionierende Teile des Codes unnötig zu verändern.



Führe für diese Datei exakt dieses 5-Schritte-Schema durch:



\### SCHRITT 1: Scope \& Isolation (Die Quarantäne)

\* Identifiziere exakt die Zeilen, die laut Fehlerbericht (aus Schritt 1) defekt sind.

\* \*\*REGEL:\*\* Ändere NICHTS, was nicht explizit als Fehler gemeldet wurde oder zwingend für den Fix notwendig ist. Kein "Schöner-Machen", kein unnötiges Refactoring.

\* Bestätige kurz: "Ich fokussiere mich nur auf \[Liste der Fehler]."



\### SCHRITT 2: Recherche \& Validierung

\* Falls externe Libraries (FFmpeg, Qt, NumPy) betroffen sind: Prüfe kurz intern, ob deine geplante Lösung mit der aktuellen Version kompatibel ist.

\* Falls du dir unsicher bist, recherchiere oder simuliere den Aufruf.



\### SCHRITT 3: Kontext-Check (Side-Effects)

\* Prüfe: Wenn ich diesen Fehler hier behebe, bricht das etwas in einer anderen Datei (z.B. Importe, Funktionsaufrufe)?

\* Passe die Lösung so an, dass sie sich nahtlos in den Rest der App einfügt.



\### SCHRITT 4: Die "Dry-Run" Simulation (Qualitäts-Schleife)

\* Simuliere gedanklich die Ausführung des neuen Codes.

\* \*Szenario:\* Wenn der Fehler behoben ist, läuft der Rest des Codes weiter?

\* Falls du ein Problem findest: Gehe zurück zu Schritt 2 und korrigiere deinen Ansatz, BEVOR du Code ausgibst.



\### SCHRITT 5: Finale Ausgabe

Erstelle den vollständigen Code der Datei.

\* Kommentiere NUR an den geänderten Stellen: `# FIX: \[Was wurde warum geändert]`.

\* Lasse den restlichen Code exakt so, wie er war (Copy-Paste-Ready).



\*\*START:\*\*

Wende dieses Schema jetzt auf die genannte Datei an. Beginne mit Schritt 1.



---



\### TASK BLOCK 6: Emotion \& Energy Curves (Algorithmus Fehler)

\*\*ZIEL-DATEI:\*\* `src/pb\_studio/pacing/emotion\_curve.py`

\*\*ZIEL-DATEI 2:\*\* `src/pb\_studio/pacing/energy\_curve.py`

\*\*FEHLER-DETAILS:\*\*

1\. `emotion\_curve.py`: Division by Zero bei doppelten Keyframes (Zeile 90).

2\. `energy\_curve.py`: Endlosschleife/Logikfehler in `\_find\_peak` (Zeile 120-140).



Wir bearbeiten nun den nächsten Task aus dem Masterplan.



Du arbeitest ab jetzt nach dem strengen \*\*"Surgical Fix Protocol"\*\*. Dein Ziel ist es, die dokumentierten Fehler zu beheben, OHNE die bestehende Struktur, den Stil oder funktionierende Teile des Codes unnötig zu verändern.



Führe für diese Datei exakt dieses 5-Schritte-Schema durch:



\### SCHRITT 1: Scope \& Isolation (Die Quarantäne)

\* Identifiziere exakt die Zeilen, die laut Fehlerbericht (aus Schritt 1) defekt sind.

\* \*\*REGEL:\*\* Ändere NICHTS, was nicht explizit als Fehler gemeldet wurde oder zwingend für den Fix notwendig ist. Kein "Schöner-Machen", kein unnötiges Refactoring.

\* Bestätige kurz: "Ich fokussiere mich nur auf \[Liste der Fehler]."



\### SCHRITT 2: Recherche \& Validierung

\* Falls externe Libraries (FFmpeg, Qt, NumPy) betroffen sind: Prüfe kurz intern, ob deine geplante Lösung mit der aktuellen Version kompatibel ist.

\* Falls du dir unsicher bist, recherchiere oder simuliere den Aufruf.



\### SCHRITT 3: Kontext-Check (Side-Effects)

\* Prüfe: Wenn ich diesen Fehler hier behebe, bricht das etwas in einer anderen Datei (z.B. Importe, Funktionsaufrufe)?

\* Passe die Lösung so an, dass sie sich nahtlos in den Rest der App einfügt.



\### SCHRITT 4: Die "Dry-Run" Simulation (Qualitäts-Schleife)

\* Simuliere gedanklich die Ausführung des neuen Codes.

\* \*Szenario:\* Wenn der Fehler behoben ist, läuft der Rest des Codes weiter?

\* Falls du ein Problem findest: Gehe zurück zu Schritt 2 und korrigiere deinen Ansatz, BEVOR du Code ausgibst.



\### SCHRITT 5: Finale Ausgabe

Erstelle den vollständigen Code der Datei.

\* Kommentiere NUR an den geänderten Stellen: `# FIX: \[Was wurde warum geändert]`.

\* Lasse den restlichen Code exakt so, wie er war (Copy-Paste-Ready).



\*\*START:\*\*

Wende dieses Schema jetzt auf die genannte Datei an. Beginne mit Schritt 1.



---



\## 🟠 PRIORITÄT 2: HOHE PRIORITÄT (Wichtige Fixes)



\### TASK BLOCK 7: Utils \& Threading (Singletons)

\*\*ZIEL-DATEI:\*\* `src/pb\_studio/utils/embedding\_cache.py`

\*\*FEHLER-DETAILS:\*\*

1\. Race Condition: Singleton-Implementierung ist nicht thread-safe (Zeile 266-278).



Wir bearbeiten nun den nächsten Task aus dem Masterplan.



Du arbeitest ab jetzt nach dem strengen \*\*"Surgical Fix Protocol"\*\*. Dein Ziel ist es, die dokumentierten Fehler zu beheben, OHNE die bestehende Struktur, den Stil oder funktionierende Teile des Codes unnötig zu verändern.



Führe für diese Datei exakt dieses 5-Schritte-Schema durch:



\### SCHRITT 1: Scope \& Isolation (Die Quarantäne)

\* Identifiziere exakt die Zeilen, die laut Fehlerbericht (aus Schritt 1) defekt sind.

\* \*\*REGEL:\*\* Ändere NICHTS, was nicht explizit als Fehler gemeldet wurde oder zwingend für den Fix notwendig ist. Kein "Schöner-Machen", kein unnötiges Refactoring.

\* Bestätige kurz: "Ich fokussiere mich nur auf \[Liste der Fehler]."



\### SCHRITT 2: Recherche \& Validierung

\* Falls externe Libraries (FFmpeg, Qt, NumPy) betroffen sind: Prüfe kurz intern, ob deine geplante Lösung mit der aktuellen Version kompatibel ist.

\* Falls du dir unsicher bist, recherchiere oder simuliere den Aufruf.



\### SCHRITT 3: Kontext-Check (Side-Effects)

\* Prüfe: Wenn ich diesen Fehler hier behebe, bricht das etwas in einer anderen Datei (z.B. Importe, Funktionsaufrufe)?

\* Passe die Lösung so an, dass sie sich nahtlos in den Rest der App einfügt.



\### SCHRITT 4: Die "Dry-Run" Simulation (Qualitäts-Schleife)

\* Simuliere gedanklich die Ausführung des neuen Codes.

\* \*Szenario:\* Wenn der Fehler behoben ist, läuft der Rest des Codes weiter?

\* Falls du ein Problem findest: Gehe zurück zu Schritt 2 und korrigiere deinen Ansatz, BEVOR du Code ausgibst.



\### SCHRITT 5: Finale Ausgabe

Erstelle den vollständigen Code der Datei.

\* Kommentiere NUR an den geänderten Stellen: `# FIX: \[Was wurde warum geändert]`.

\* Lasse den restlichen Code exakt so, wie er war (Copy-Paste-Ready).



\*\*START:\*\*

Wende dieses Schema jetzt auf die genannte Datei an. Beginne mit Schritt 1.



---



\### TASK BLOCK 8: GUI Controllers (Memory Leaks)

\*\*ZIEL-DATEI:\*\* `src/pb\_studio/gui/controllers/clip\_browser\_controller.py`

\*\*FEHLER-DETAILS:\*\*

1\. Memory Leak: `ThumbnailWorker` fehlt `deleteLater()` nach Beendigung (Zeile 45-80).



Wir bearbeiten nun den nächsten Task aus dem Masterplan.



Du arbeitest ab jetzt nach dem strengen \*\*"Surgical Fix Protocol"\*\*. Dein Ziel ist es, die dokumentierten Fehler zu beheben, OHNE die bestehende Struktur, den Stil oder funktionierende Teile des Codes unnötig zu verändern.



Führe für diese Datei exakt dieses 5-Schritte-Schema durch:



\### SCHRITT 1: Scope \& Isolation (Die Quarantäne)

\* Identifiziere exakt die Zeilen, die laut Fehlerbericht (aus Schritt 1) defekt sind.

\* \*\*REGEL:\*\* Ändere NICHTS, was nicht explizit als Fehler gemeldet wurde oder zwingend für den Fix notwendig ist. Kein "Schöner-Machen", kein unnötiges Refactoring.

\* Bestätige kurz: "Ich fokussiere mich nur auf \[Liste der Fehler]."



\### SCHRITT 2: Recherche \& Validierung

\* Falls externe Libraries (FFmpeg, Qt, NumPy) betroffen sind: Prüfe kurz intern, ob deine geplante Lösung mit der aktuellen Version kompatibel ist.

\* Falls du dir unsicher bist, recherchiere oder simuliere den Aufruf.



\### SCHRITT 3: Kontext-Check (Side-Effects)

\* Prüfe: Wenn ich diesen Fehler hier behebe, bricht das etwas in einer anderen Datei (z.B. Importe, Funktionsaufrufe)?

\* Passe die Lösung so an, dass sie sich nahtlos in den Rest der App einfügt.



\### SCHRITT 4: Die "Dry-Run" Simulation (Qualitäts-Schleife)

\* Simuliere gedanklich die Ausführung des neuen Codes.

\* \*Szenario:\* Wenn der Fehler behoben ist, läuft der Rest des Codes weiter?

\* Falls du ein Problem findest: Gehe zurück zu Schritt 2 und korrigiere deinen Ansatz, BEVOR du Code ausgibst.



\### SCHRITT 5: Finale Ausgabe

Erstelle den vollständigen Code der Datei.

\* Kommentiere NUR an den geänderten Stellen: `# FIX: \[Was wurde warum geändert]`.

\* Lasse den restlichen Code exakt so, wie er war (Copy-Paste-Ready).



\*\*START:\*\*

Wende dieses Schema jetzt auf die genannte Datei an. Beginne mit Schritt 1.



---



\### TASK BLOCK 9: Database Connection (Error Handling)

\*\*ZIEL-DATEI:\*\* `src/pb\_studio/database/connection.py`

\*\*FEHLER-DETAILS:\*\*

1\. Unhandled Exception: Wenn DuckDB gesperrt/korrupt ist, stürzt App beim Verbindungsaufbau ab (Zeile 246-251).



Wir bearbeiten nun den nächsten Task aus dem Masterplan.



Du arbeitest ab jetzt nach dem strengen \*\*"Surgical Fix Protocol"\*\*. Dein Ziel ist es, die dokumentierten Fehler zu beheben, OHNE die bestehende Struktur, den Stil oder funktionierende Teile des Codes unnötig zu verändern.



Führe für diese Datei exakt dieses 5-Schritte-Schema durch:



\### SCHRITT 1: Scope \& Isolation (Die Quarantäne)

\* Identifiziere exakt die Zeilen, die laut Fehlerbericht (aus Schritt 1) defekt sind.

\* \*\*REGEL:\*\* Ändere NICHTS, was nicht explizit als Fehler gemeldet wurde oder zwingend für den Fix notwendig ist. Kein "Schöner-Machen", kein unnötiges Refactoring.

\* Bestätige kurz: "Ich fokussiere mich nur auf \[Liste der Fehler]."



\### SCHRITT 2: Recherche \& Validierung

\* Falls externe Libraries (FFmpeg, Qt, NumPy) betroffen sind: Prüfe kurz intern, ob deine geplante Lösung mit der aktuellen Version kompatibel ist.

\* Falls du dir unsicher bist, recherchiere oder simuliere den Aufruf.



\### SCHRITT 3: Kontext-Check (Side-Effects)

\* Prüfe: Wenn ich diesen Fehler hier behebe, bricht das etwas in einer anderen Datei (z.B. Importe, Funktionsaufrufe)?

\* Passe die Lösung so an, dass sie sich nahtlos in den Rest der App einfügt.



\### SCHRITT 4: Die "Dry-Run" Simulation (Qualitäts-Schleife)

\* Simuliere gedanklich die Ausführung des neuen Codes.

\* \*Szenario:\* Wenn der Fehler behoben ist, läuft der Rest des Codes weiter?

\* Falls du ein Problem findest: Gehe zurück zu Schritt 2 und korrigiere deinen Ansatz, BEVOR du Code ausgibst.



\### SCHRITT 5: Finale Ausgabe

Erstelle den vollständigen Code der Datei.

\* Kommentiere NUR an den geänderten Stellen: `# FIX: \[Was wurde warum geändert]`.

\* Lasse den restlichen Code exakt so, wie er war (Copy-Paste-Ready).



\*\*START:\*\*

Wende dieses Schema jetzt auf die genannte Datei an. Beginne mit Schritt 1.



---



\### TASK BLOCK 10: Semantic Analysis (ML Config \& GPU)

\*\*ZIEL-DATEI:\*\* `src/pb\_studio/analysis/semantic\_analyzer.py`

\*\*FEHLER-DETAILS:\*\*

1\. Mode: CLIP-Modell nicht im `eval()` Modus (Zeile 155-157).

2\. GPU Memory: `torch.cuda.empty\_cache()` fehlt nach Batch-Processing (Zeile 200-250).



Wir bearbeiten nun den nächsten Task aus dem Masterplan.



Du arbeitest ab jetzt nach dem strengen \*\*"Surgical Fix Protocol"\*\*. Dein Ziel ist es, die dokumentierten Fehler zu beheben, OHNE die bestehende Struktur, den Stil oder funktionierende Teile des Codes unnötig zu verändern.



Führe für diese Datei exakt dieses 5-Schritte-Schema durch:



\### SCHRITT 1: Scope \& Isolation (Die Quarantäne)

\* Identifiziere exakt die Zeilen, die laut Fehlerbericht (aus Schritt 1) defekt sind.

\* \*\*REGEL:\*\* Ändere NICHTS, was nicht explizit als Fehler gemeldet wurde oder zwingend für den Fix notwendig ist. Kein "Schöner-Machen", kein unnötiges Refactoring.

\* Bestätige kurz: "Ich fokussiere mich nur auf \[Liste der Fehler]."



\### SCHRITT 2: Recherche \& Validierung

\* Falls externe Libraries (FFmpeg, Qt, NumPy) betroffen sind: Prüfe kurz intern, ob deine geplante Lösung mit der aktuellen Version kompatibel ist.

\* Falls du dir unsicher bist, recherchiere oder simuliere den Aufruf.



\### SCHRITT 3: Kontext-Check (Side-Effects)

\* Prüfe: Wenn ich diesen Fehler hier behebe, bricht das etwas in einer anderen Datei (z.B. Importe, Funktionsaufrufe)?

\* Passe die Lösung so an, dass sie sich nahtlos in den Rest der App einfügt.



\### SCHRITT 4: Die "Dry-Run" Simulation (Qualitäts-Schleife)

\* Simuliere gedanklich die Ausführung des neuen Codes.

\* \*Szenario:\* Wenn der Fehler behoben ist, läuft der Rest des Codes weiter?

\* Falls du ein Problem findest: Gehe zurück zu Schritt 2 und korrigiere deinen Ansatz, BEVOR du Code ausgibst.



\### SCHRITT 5: Finale Ausgabe

Erstelle den vollständigen Code der Datei.

\* Kommentiere NUR an den geänderten Stellen: `# FIX: \[Was wurde warum geändert]`.

\* Lasse den restlichen Code exakt so, wie er war (Copy-Paste-Ready).



\*\*START:\*\*

Wende dieses Schema jetzt auf die genannte Datei an. Beginne mit Schritt 1.

