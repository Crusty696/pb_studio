# Dear PyGui Integration in PB_studio

## Überblick

PB_studio nutzt **zwei parallele GUI-Frameworks**:

1. **PyQt6** - Haupt-GUI (Main Window, Docks, Dialoge)
2. **Dear PyGui** - Timeline und Preview (separate Windows)

Die Integration erfolgt über die `DearPyGuiBridge` Klasse, die bidirektionale Kommunikation zwischen beiden Frameworks ermöglicht.

## Architektur

```
┌─────────────────────────────────────────┐
│          PyQt6 Main Window              │
│  - Menu Bar, Toolbar, Status Bar        │
│  - Clip Library, Parameter Dashboard    │
│  - PyQt6 Timeline (QGraphicsView)       │
└──────────────┬──────────────────────────┘
               │
               │ DearPyGuiBridge
               │
       ┌───────┴────────┬──────────────┐
       │                │              │
┌──────▼──────┐  ┌──────▼──────┐ ┌────▼─────┐
│ DPG Timeline│  │ DPG Preview │ │ Callbacks│
│  Window     │  │   Window    │ │  & Sync  │
└─────────────┘  └─────────────┘ └──────────┘
```

## Warum zwei Frameworks?

Dear PyGui bietet spezielle Vorteile für:

1. **Timeline**:
   - Hardware-beschleunigte Waveform-Rendering
   - Echtzeit-Scrubbing ohne Lag
   - Native Beat-Grid Visualisierung
   - Drag & Drop mit Pixel-Präzision

2. **Preview**:
   - GPU-Textures für Video-Frames
   - Minimale Latenz bei Frame-Updates
   - Effizientes Frame-Caching
   - Bessere Performance bei HD/4K

## Verwendung

### Dear PyGui Timeline öffnen

**Via Menu:**
```
View → Show Dear PyGui Timeline (Ctrl+Shift+T)
```

**Programmatisch:**
```python
main_window.show_dpg_timeline()
```

### Dear PyGui Preview öffnen

**Via Menu:**
```
View → Show Dear PyGui Preview (Ctrl+Shift+P)
```

**Programmatisch:**
```python
main_window.show_dpg_preview()
```

## DearPyGuiBridge API

### Initialisierung

```python
from pb_studio.gui.dearpygui_bridge import DearPyGuiBridge

bridge = DearPyGuiBridge(
    on_position_changed=callback_func,  # Position-Sync PyQt6 ← DPG
    on_clip_selected=callback_func       # Clip-Auswahl Event
)
```

### Audio laden

```python
bridge.load_audio("track.mp3", bpm=140.0)
```

### Cut List laden

```python
bridge.load_cut_list(
    cut_list=[CutListEntry(...)],
    video_clips={"clip1": "/path/to/video1.mp4", ...}
)
```

### Playback-Steuerung

```python
bridge.play()       # Start playback
bridge.pause()      # Pause playback
bridge.stop()       # Stop and reset
bridge.set_position(10.5)  # Seek to 10.5 seconds
```

## Threading & Performance

- **Dear PyGui läuft in separatem Thread** - UI bleibt responsive
- **Callbacks sind thread-safe** - Synchronisation via PyQt Signals
- **Frame-Cache** - 500MB Cache für Video-Frames
- **GPU-Textures** - Hardware-beschleunigtes Rendering

## Implementierte Features

### Timeline (DPG)
- ✅ Waveform-Visualisierung
- ✅ Beat-Grid Overlay
- ✅ Clip-Platzierung mit Drag & Drop
- ✅ Snap-to-Beat Funktion
- ✅ Zoom & Scroll
- ✅ Playhead mit Sync
- ✅ Cut-Point Marker

### Preview (DPG)
- ✅ Video-Frame-Rendering
- ✅ OpenCV Frame-Extraktion
- ✅ Frame-Cache (FIFO)
- ✅ Audio-Video-Sync
- ✅ Playback Controls
- ✅ Time Display

## Bekannte Limitierungen

1. **Separate Windows**: Dear PyGui kann nicht in PyQt6 Widgets embedded werden
2. **Threading**: DPG muss in eigenem Thread laufen
3. **Cleanup**: DPG Context muss manuell zerstört werden beim Schließen

## Code-Beispiel: Vollständige Integration

```python
from pb_studio.gui.dearpygui_bridge import DearPyGuiBridge

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # DPG Bridge initialisieren
        self.dpg_bridge = DearPyGuiBridge(
            on_position_changed=self._sync_position,
            on_clip_selected=self._on_clip_selected
        )

    def show_dpg_timeline(self):
        """Timeline öffnen."""
        self.dpg_bridge.show_timeline()

        # Audio laden wenn verfügbar
        if self.audio_path:
            self.dpg_bridge.load_audio(self.audio_path, bpm=self.bpm)

    def show_dpg_preview(self):
        """Preview öffnen."""
        self.dpg_bridge.show_preview()

        # Cut list laden wenn verfügbar
        if self.cut_list:
            self.dpg_bridge.load_cut_list(self.cut_list, self.video_clips)

    def _sync_position(self, position: float):
        """Sync position von DPG zu PyQt6."""
        self.timeline_widget.set_position(position)

    def _on_clip_selected(self, clip_id: int):
        """Handle clip selection."""
        print(f"Clip {clip_id} selected")

    def closeEvent(self, event):
        """Cleanup beim Schließen."""
        if self.dpg_bridge:
            self.dpg_bridge.cleanup()
        event.accept()
```

## Debugging

**Dear PyGui Logs aktivieren:**
```python
import dearpygui.dearpygui as dpg
dpg.configure_app(manual_callback_management=False)
```

**Check DPG Status:**
```python
if dpg.is_dearpygui_running():
    print("DPG is running")
```

**Memory Debugging:**
```python
print(f"Frame cache size: {bridge.preview.frame_cache.current_size / 1024 / 1024:.2f} MB")
```

## Dependencies

Required packages:
```
dearpygui >= 1.10.1
opencv-python >= 4.8.0
numpy >= 1.24.0
```

Install:
```bash
poetry install --with dev
```

## Weiterführende Dokumentation

- [Dear PyGui Docs](https://dearpygui.readthedocs.io/)
- [TimelineWidget API](widgets/timeline_widget.py)
- [PreviewLogic API](preview_logic.py)
- [DearPyGuiBridge Source](dearpygui_bridge.py)

---

**Author:** PB_studio Development Team
**Task:** A3 - Dear PyGui Integration
**Date:** 2025-11-29
