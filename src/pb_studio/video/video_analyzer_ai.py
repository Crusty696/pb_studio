"""
KI-basierte Video-Analyse mit X-CLIP.

Dieses Modul bietet Deep-Learning-basierte Video-Embeddings (512D) als Alternative
zu den manuellen 27D Features. Es nutzt Microsoft X-CLIP für semantische Video-Analyse.

WICHTIG: Vollständig optional! Wenn Modelle fehlen, nutze Fallback zu manuellen Features.

Verwendung:
    from pb_studio.video.video_analyzer_ai import AIVideoAnalyzer

    analyzer = AIVideoAnalyzer()
    if analyzer.available:
        embedding = analyzer.extract_video_embedding("video.mp4")  # Shape: (512,)
"""

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Konstanten
AI_EMBEDDING_DIM = 512  # X-CLIP projection dimension
MODEL_NAME = "microsoft/xclip-base-patch32"
NUM_FRAMES = 8  # X-CLIP erwartet 8 Frames
FRAME_SIZE = 224  # 224x224 Pixel

# Cache-Verzeichnis für heruntergeladene Modelle
DEFAULT_MODEL_CACHE = Path("data/ai_models")
DEFAULT_MODEL_DIR = DEFAULT_MODEL_CACHE / "xclip"


class AIVideoAnalyzer:
    """
    KI-basierte Video-Analyse mit X-CLIP.

    WICHTIG: Vollständig optional! Wenn Modelle fehlen, nutze Fallback.

    Attributes:
        available (bool): True wenn Modell geladen und nutzbar
        model: Das X-CLIP Modell
        processor: Der X-CLIP Processor für Video-Preprocessing
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton-Pattern für Memory-Effizienz."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: str | None = None, use_gpu: bool = True):
        """
        Initialisiert den AIVideoAnalyzer.

        Args:
            model_path: Pfad zu heruntergeladenem Modell (None = auto-download)
            use_gpu: GPU verwenden falls verfügbar
        """
        if self._initialized:
            return

        self.available = False
        self.model = None
        self.processor = None
        self.device = "cpu"
        self._inference_lock = threading.Lock()

        try:
            resolved_path = self._resolve_model_path(model_path)
            self._load_model(resolved_path, use_gpu)
            self._initialized = True
        except Exception as e:
            logger.warning(f"KI-Modelle nicht verfügbar: {e}")
            logger.info("Fallback: Nutze bestehende manuelle Feature-Extraktion (27D)")
            self._initialized = True

    def _resolve_model_path(self, model_path: str | None) -> Path | None:
        """
        Wählt den Modellpfad:
        1. Expliziter Pfad, falls angegeben
        2. Lokaler Cache: data/ai_models/xclip (wenn vorhanden)
        3. None -> HuggingFace Hub (Online)
        """
        if model_path:
            return Path(model_path)

        if DEFAULT_MODEL_DIR.exists():
            return DEFAULT_MODEL_DIR

        return None

    def _load_model(self, model_path: Path | None, use_gpu: bool) -> None:
        """Lädt das X-CLIP Modell."""
        try:
            import torch
            from transformers import AutoModel, AutoProcessor

            # Device bestimmen
            self._directml_device = None  # Speichere DirectML Device-Objekt separat
            if use_gpu and torch.cuda.is_available():
                self.device = "cuda"
                logger.info("AIVideoAnalyzer: CUDA GPU erkannt")
            elif use_gpu:
                # DirectML für AMD prüfen (Windows)
                try:
                    import torch_directml

                    self._directml_device = torch_directml.device()
                    self.device = "directml"  # String für Konsistenz
                    logger.info("AIVideoAnalyzer: DirectML (AMD GPU) erkannt")
                except ImportError:
                    self.device = "cpu"
                    logger.info("AIVideoAnalyzer: CPU-Modus (kein CUDA/DirectML)")
            else:
                self.device = "cpu"
                logger.info("AIVideoAnalyzer: CPU-Modus (explizit angefordert)")

            # Modell laden (lokal bevorzugt)
            if model_path is not None:
                local_processor = model_path / "processor"
                local_model = model_path / "model"

                processor_source: str | Path = (
                    local_processor if local_processor.exists() else model_path
                )
                model_source: str | Path = local_model if local_model.exists() else model_path
                logger.info(f"Lade X-CLIP Modell lokal: {model_source}")
            else:
                processor_source = MODEL_NAME
                model_source = MODEL_NAME
                logger.info(f"Lade X-CLIP Modell aus HuggingFace Hub: {MODEL_NAME}")

            self.processor = AutoProcessor.from_pretrained(processor_source)
            self.model = AutoModel.from_pretrained(model_source)

            # Auf Device verschieben
            if self.device == "cuda":
                self.model = self.model.to("cuda")
            elif self.device == "directml" and self._directml_device is not None:
                self.model = self.model.to(self._directml_device)

            # Eval-Modus für Inferenz
            self.model.eval()

            self.available = True
            logger.info(
                f"AIVideoAnalyzer initialisiert: device={self.device}, embedding_dim={AI_EMBEDDING_DIM}"
            )

        except ImportError as e:
            raise ImportError(
                f"Benötigte Pakete fehlen: {e}. Installiere mit: poetry install -E ai-video"
            )
        except Exception as e:
            raise RuntimeError(f"Modell konnte nicht geladen werden: {e}")

    def extract_video_embedding(self, video_path: str) -> np.ndarray | None:
        """
        Extrahiert 512D Embedding aus Video.

        Args:
            video_path: Pfad zur Video-Datei

        Returns:
            np.ndarray shape (512,) dtype float32, oder None bei Fehler

        Fallback:
            Wenn Modell nicht verfügbar, gibt None zurück (Caller nutzt manuelle Features)
        """
        if not self.available:
            logger.debug("AIVideoAnalyzer nicht verfügbar, Fallback zu manuellen Features")
            return None

        try:
            # Frames extrahieren
            frames = self._extract_frames(video_path, NUM_FRAMES)
            if frames is None or len(frames) == 0:
                logger.warning(f"Keine Frames extrahiert aus: {video_path}")
                return None

            # Thread-Safety für Modell-Inferenz
            with self._inference_lock:
                return self._compute_embedding(frames)

        except Exception as e:
            logger.error(f"Video-Embedding-Extraktion fehlgeschlagen: {e}")
            return None

    def _extract_frames(self, video_path: str, num_frames: int) -> np.ndarray | None:
        """
        Extrahiert gleichmäßig verteilte Frames aus Video.

        Args:
            video_path: Pfad zur Video-Datei
            num_frames: Anzahl zu extrahierender Frames

        Returns:
            np.ndarray shape (num_frames, height, width, 3) oder None
        """
        try:
            import av

            container = av.open(video_path)
            stream = container.streams.video[0]
            total_frames = stream.frames

            if total_frames == 0:
                # Fallback: Frames zählen durch Iteration
                # Container neu öffnen nach Iteration (seek kann unzuverlässig sein)
                for _ in container.decode(video=0):
                    total_frames += 1
                container.close()
                container = av.open(video_path)

            # Gleichmäßig verteilte Frame-Indizes (als Set für O(1) lookup)
            if total_frames <= num_frames:
                indices = set(range(total_frames))
            else:
                indices = set(np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist())

            # Frames dekodieren
            frames = []
            for i, frame in enumerate(container.decode(video=0)):
                if i in indices:
                    # Zu RGB24 numpy array konvertieren
                    img = frame.to_ndarray(format="rgb24")
                    frames.append(img)
                if len(frames) >= num_frames:
                    break

            container.close()

            if len(frames) < num_frames:
                # Padding mit letztem Frame wenn nötig
                while len(frames) < num_frames:
                    frames.append(
                        frames[-1]
                        if frames
                        else np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
                    )

            return np.stack(frames)

        except ImportError:
            logger.error("PyAV nicht installiert. Installiere mit: poetry install -E ai-video")
            return None
        except Exception as e:
            logger.error(f"Frame-Extraktion fehlgeschlagen: {e}")
            return None

    def _compute_embedding(self, frames: np.ndarray) -> np.ndarray:
        """
        Berechnet Video-Embedding aus Frames.

        Args:
            frames: np.ndarray shape (num_frames, height, width, 3)

        Returns:
            np.ndarray shape (512,) dtype float32
        """
        import torch

        # Frames als Liste für Processor
        frame_list = [frames[i] for i in range(frames.shape[0])]

        # Preprocessing
        inputs = self.processor(videos=frame_list, return_tensors="pt")

        # Auf Device verschieben
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        elif self.device == "directml" and self._directml_device is not None:
            inputs = {k: v.to(self._directml_device) for k, v in inputs.items()}

        # Inferenz
        with torch.no_grad():
            video_features = self.model.get_video_features(**inputs)

        # Zu numpy konvertieren
        embedding = video_features.cpu().numpy().squeeze()

        # Normalisieren (für Cosine-Similarity)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        else:
            logger.warning("Embedding hat Norm 0, Normalisierung übersprungen")

        return embedding.astype(np.float32)

    def analyze_batch(
        self, video_paths: list[str], progress_callback=None
    ) -> dict[str, np.ndarray]:
        """
        Analysiert mehrere Videos in einem Batch.

        Args:
            video_paths: Liste von Video-Pfaden
            progress_callback: Optional callback(current, total)

        Returns:
            Dict[video_path, embedding] - Nur erfolgreiche Extraktionen
        """
        results = {}
        total = len(video_paths)

        for i, path in enumerate(video_paths):
            if progress_callback:
                progress_callback(i, total)

            embedding = self.extract_video_embedding(path)
            if embedding is not None:
                results[path] = embedding

        if progress_callback:
            progress_callback(total, total)

        logger.info(f"Batch-Analyse abgeschlossen: {len(results)}/{total} Videos erfolgreich")
        return results

    @staticmethod
    def get_embedding_dimension() -> int:
        """Gibt die Embedding-Dimension zurück (512 für X-CLIP)."""
        return AI_EMBEDDING_DIM

    @staticmethod
    def is_model_available() -> bool:
        """Prüft ob X-CLIP Modell verfügbar/installiert ist."""
        try:
            from transformers import AutoModel, AutoProcessor

            return True
        except ImportError:
            return False

    def get_status(self) -> dict[str, Any]:
        """Gibt Status-Informationen zurück."""
        return {
            "available": self.available,
            "device": str(self.device),
            "model_name": MODEL_NAME,
            "embedding_dim": AI_EMBEDDING_DIM,
            "num_frames": NUM_FRAMES,
        }


# =============================================================================
# Convenience Functions (für einfache Nutzung)
# =============================================================================


def extract_ai_video_embedding(video_path: str) -> np.ndarray | None:
    """
    Convenience-Funktion: Extrahiert KI-Embedding aus Video.

    Args:
        video_path: Pfad zur Video-Datei

    Returns:
        np.ndarray shape (512,) oder None wenn KI nicht verfügbar
    """
    try:
        analyzer = AIVideoAnalyzer()
        if not analyzer.available:
            return None
        return analyzer.extract_video_embedding(video_path)
    except Exception as e:
        logger.debug(f"AI-Embedding fehlgeschlagen: {e}")
        return None


def get_ai_embedding_dimension() -> int:
    """Gibt die KI-Embedding-Dimension zurück (512)."""
    return AI_EMBEDDING_DIM


def is_ai_video_available() -> bool:
    """Prüft ob KI-Video-Analyse verfügbar ist."""
    try:
        analyzer = AIVideoAnalyzer()
        return analyzer.available
    except Exception:
        return False
