# AI-Modelle in PB Studio

PB Studio nutzt lokale KI-Modelle für erweiterte Funktionen. Alle Modelle laufen **offline** auf deiner Hardware.

## Verfügbare Modelle

### 1. Moondream2 (Vision-Analyse)
- **Funktion:** Beschreibt Bildinhalte, erkennt Szenen und Objekte
- **Format:** ONNX (Split-Encoder/Decoder)
- **Größe:** ~1.5 GB
- **Quelle:** [Xenova/moondream2](https://huggingface.co/Xenova/moondream2)

### 2. Phi-3 Mini (Story-Generation)
- **Funktion:** Generiert Story-Beschreibungen und Text-Analysen
- **Format:** ONNX (DirectML-optimiert für AMD/Intel)
- **Größe:** ~2.5 GB
- **Quelle:** [microsoft/Phi-3-mini-4k-instruct-onnx](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx)

### 3. YOLOv8n (Objekt-Erkennung)
- **Funktion:** Erkennt Objekte und Personen in Videos
- **Format:** ONNX
- **Größe:** ~12 MB
- **Status:** ✅ Vorinstalliert

---

## Modelle herunterladen

### Automatisch (empfohlen)

```powershell
# Aktiviere venv, dann:
python download_models.py
```

Das Skript lädt Moondream2 und Phi-3 automatisch herunter.

### Manuell

Die Modelle werden in `data/ai_models/` erwartet:
```
data/ai_models/
├── moondream2/              # Moondream2 ONNX-Dateien
├── phi-3-mini-4k-directml/  # Phi-3 DirectML-Version
└── yolov8n.onnx             # YOLO (vorinstalliert)
```

---

## GPU-Kompatibilität

| Hardware | Moondream2 | Phi-3 | YOLO |
|----------|------------|-------|------|
| **NVIDIA (CUDA)** | ✅ | ⚠️ GenAI nötig | ✅ |
| **AMD (DirectML)** | ✅ | ✅ | ✅ |
| **Intel (DirectML)** | ✅ | ✅ | ✅ |
| **CPU** | ✅ (langsam) | ✅ (langsam) | ✅ |

### Phi-3 auf AMD/Intel

Phi-3 nutzt `onnxruntime-genai` für DirectML. Falls du diese Warnung siehst:
```
WARNING - DirectML detected, but onnxruntime-genai is missing. Phi-3 might fail.
```

Installiere zusätzlich:
```powershell
pip install onnxruntime-genai-directml
```

> **Hinweis:** `onnxruntime-genai` ist aktuell nur für Windows verfügbar.

---

## Fehlerbehebung

### "Model files missing"
→ Führe `python download_models.py` aus

### "onnxruntime_genai not installed"
→ `pip install onnxruntime-genai-directml` (AMD/Intel)
→ `pip install onnxruntime-genai` (NVIDIA)

### Modelle werden nicht erkannt
→ Prüfe, ob die Ordnerstruktur in `data/ai_models/` korrekt ist
