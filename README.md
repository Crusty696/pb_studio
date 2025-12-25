# PB Studio - Precision Beat Video Studio

Musiksynchrone Videobearbeitung mit harten Schnitten.

## Features

- Audio-Analyse mit Librosa
- Pacing-Engine fuer Beat-Synchronisation
- Video-Verarbeitung mit FFmpeg
- PyQt6 GUI
- Vector Search (FAISS/Qdrant)
- KI-Module (YOLO, CLIP) mit DirectML GPU-Beschleunigung

## Quick Start

### Windows (empfohlen)

**Doppelklick auf:**
- `start_pb_studio.bat` (Batch)
- `Start-PBStudio.ps1` (PowerShell)

Die Skripte erkennen automatisch:
1. `.venv` - Virtuelle Umgebung (bevorzugt)
2. Poetry - Falls installiert
3. System-Python - Fallback

### Manuell

```bash
# Mit Poetry
poetry install
poetry run python start_app.py

# Mit venv
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python start_app.py
```

## Installation

```bash
# Poetry (empfohlen)
poetry install

# Pip
pip install -r requirements.txt
```

### GPU-Profile & KI-Features installieren

**Wichtig:** Installiere zuerst das passende PyTorch-Paket für deine Hardware, dann Poetry-Extras. Beispiele:

- NVIDIA/CUDA (Python 3.10/3.11, CUDA 12.1):
  ```bash
  pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121
  poetry install -E ai-video -E clip-video
  ```
- AMD/Intel DirectML (Windows):
  ```bash
  pip install torch-directml
  poetry install -E ai-amd -E clip-video
  ```
- CPU-only:
  ```bash
  poetry install -E ai-cpu -E clip-video
  ```

Abhängigkeiten:
- FFmpeg muss im `PATH` liegen (für ffprobe/Rendering/Thumbnails).
- PyAV (`av`) benötigt FFmpeg-Runtime; die Wheels bringen sie unter Windows meist mit.
- `clip` (OpenAI CLIP) wird über das Extra `clip-video` installiert.

## Testing

```bash
poetry run pytest
```

## Dependencies

- Python 3.10+
- PyQt6
- Librosa
- FFmpeg
- SQLAlchemy
- FAISS (CPU) or Qdrant (GPU alternative)
- ONNX Runtime DirectML (AMD GPU)
- torch-directml (AMD GPU)

## AI-Modelle (Optional)

PB Studio nutzt optionale KI-Modelle für erweiterte Funktionen:

| Modell | Funktion | Download |
|--------|----------|----------|
| Moondream2 | Bild-Beschreibung | `python download_models.py` |
| Phi-3 Mini | Story-Generation | `python download_models.py` |
| YOLOv8n | Objekt-Erkennung | ✅ Vorinstalliert |

👉 **Details:** [docs/AI_MODELS.md](docs/AI_MODELS.md)

