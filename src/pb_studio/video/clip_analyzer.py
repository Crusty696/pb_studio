"""
CLIP-basierte semantische Videoanalyse für PB_studio

Kategorisiert Video-Clips nach:
- Szenentyp (Crowd, Artist, Close-up, Wide-shot, etc.)
- Stimmung (energetic, calm, dark, bright, etc.)
- Inhalt (people, nature, abstract, etc.)

Performance Features:
- Batch-Processing für CLIP-Inference (analyze_batch)
- GPU-Memory-basierte Batch-Size-Schätzung
- Progress-Callbacks für UI-Integration
- Parallele Multi-Video-Verarbeitung (analyze_videos_parallel)

Security Fixes:
- K-02: Path Traversal Protection via validate_file_path

Benötigt: torch, clip-by-openai, pillow
"""

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..core.hardware import get_device
from ..utils.gpu_memory import torch_device_context
from ..utils.path_utils import validate_file_path

logger = logging.getLogger(__name__)

# K-02: Erlaubte Video-Erweiterungen
ALLOWED_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv", ".flv"]

# CLIP-Verfügbarkeit prüfen
try:
    import clip
    import torch
    from PIL import Image

    CLIP_AVAILABLE = True
    logger.info("CLIP dependencies available")
except ImportError as e:
    CLIP_AVAILABLE = False
    logger.warning(f"CLIP nicht verfügbar: {e}. Semantische Analyse deaktiviert.")


class CLIPVideoAnalyzer:
    """Analysiert Video-Clips mit CLIP für semantische Kategorisierung."""

    # Vordefinierte Kategorien
    SCENE_TYPES = [
        "crowd shot",
        "artist performance",
        "close-up face",
        "wide shot",
        "aerial view",
        "backstage",
        "abstract visuals",
    ]

    MOODS = ["high energy", "calm", "dark", "bright", "nostalgic", "futuristic", "natural"]

    CONTENT_TYPES = [
        "people dancing",
        "musical instruments",
        "nature landscape",
        "urban city",
        "abstract patterns",
        "lights and effects",
    ]

    def __init__(self, model_name: str = "ViT-B/32"):
        """
        Initialisiert CLIP-Analyzer mit Hardware-Detection.

        Args:
            model_name: CLIP-Modell (ViT-B/32, ViT-L/14, etc.)
        """
        if not CLIP_AVAILABLE:
            logger.error(
                "CLIP nicht installiert - pip install git+https://github.com/openai/CLIP.git"
            )
            self.model = None
            self.preprocess = None
            self.device = "cpu"
            return

        try:
            # Hardware Detection
            self.device = get_device()
            if self.device == "dml":
                try:
                    import torch_directml

                    self.device = torch_directml.device()
                except ImportError:
                    logger.warning(
                        "DirectML angefordert, aber torch_directml nicht installiert – Fallback auf CPU"
                    )
                    self.device = "cpu"

            # Load CLIP model with GPU memory context
            with torch_device_context(self.device):
                self.model, self.preprocess = clip.load(model_name, device=self.device)

                # Text-Embeddings vorberechnen
                self._precompute_text_embeddings()

            # Batch-Size für GPU-Memory schätzen
            self.batch_size = self._estimate_batch_size()

            logger.info(
                f"CLIPVideoAnalyzer initialisiert: {model_name} auf {self.device}, Batch-Size: {self.batch_size}"
            )

        except Exception as e:
            logger.error(f"CLIP-Initialisierung fehlgeschlagen: {e}")
            self.model = None
            self.preprocess = None
            self.device = "cpu"

    def _precompute_text_embeddings(self):
        """Berechnet Text-Embeddings für alle Kategorien vor."""
        if not self.model:
            return

        all_labels = self.SCENE_TYPES + self.MOODS + self.CONTENT_TYPES

        text_tokens = clip.tokenize(all_labels).to(self.device)

        with torch.no_grad():
            self.text_embeddings = self.model.encode_text(text_tokens)
            self.text_embeddings /= self.text_embeddings.norm(dim=-1, keepdim=True)

        self.all_labels = all_labels
        logger.debug(f"Text-Embeddings vorberechnet: {len(all_labels)} Labels")

    def _estimate_batch_size(self) -> int:
        """
        Schätzt maximale Batch-Size basierend auf GPU-Memory.

        Returns:
            Empfohlene Batch-Size (4-32)
        """
        try:
            from ..utils.gpu_memory import get_gpu_memory_info

            info = get_gpu_memory_info()

            if not info:
                # CPU-Modus
                return 4

            vram_gb = info.get("total_gb", 4)

            # CLIP ViT-B/32: ~200MB pro Bild
            # Konservative Schätzung: 50% des VRAMs nutzen
            estimated = min(32, max(4, int((vram_gb * 0.5) / 0.2)))

            logger.debug(f"Batch-Size geschätzt: {estimated} (VRAM: {vram_gb:.1f} GB)")
            return estimated

        except Exception as e:
            logger.warning(f"Batch-Size-Schätzung fehlgeschlagen: {e}, nutze Default=8")
            return 8

    def analyze_frame(self, frame_path: str | Path) -> dict[str, float]:
        """
        Analysiert ein einzelnes Frame mit GPU Memory Management.

        Args:
            frame_path: Pfad zum Frame-Bild

        Returns:
            Dict mit Label -> Confidence-Score
        """
        if not CLIP_AVAILABLE or not self.model:
            return {}

        try:
            # Use GPU memory context for automatic cleanup
            with torch_device_context(self.device):
                image = Image.open(frame_path)
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    image_embedding = self.model.encode_image(image_input)
                    image_embedding /= image_embedding.norm(dim=-1, keepdim=True)

                    # Cosine Similarity
                    similarities = (image_embedding @ self.text_embeddings.T).squeeze(0)

                results = {}
                for label, score in zip(self.all_labels, similarities.cpu().numpy()):
                    results[label] = float(score)

            # GPU memory is automatically cleaned up here
            return results

        except Exception as e:
            logger.error(f"Frame-Analyse fehlgeschlagen: {e}")
            return {}

    def analyze_batch(
        self, frames: list[np.ndarray], labels: list[str] | None = None
    ) -> list[dict[str, float]]:
        """
        Batch-Verarbeitung mehrerer Frames für höhere Effizienz.

        Args:
            frames: Liste von Frames (numpy arrays in BGR/RGB)
            labels: Optional: spezifische Labels, sonst alle vordefinierten

        Returns:
            Liste von Dicts mit Label -> Confidence-Score für jedes Frame
        """
        if not CLIP_AVAILABLE or not self.model:
            return [{} for _ in frames]

        if not frames:
            return []

        try:
            # Labels vorbereiten
            use_labels = labels if labels else self.all_labels

            with torch_device_context(self.device):
                # Preprocessing für alle Frames
                from PIL import Image

                batch_tensors = []
                for frame in frames:
                    # OpenCV BGR -> RGB
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        frame_rgb = frame[:, :, ::-1]  # BGR -> RGB
                    else:
                        frame_rgb = frame

                    # Numpy -> PIL -> Preprocessed Tensor
                    pil_image = Image.fromarray(frame_rgb.astype("uint8"))
                    tensor = self.preprocess(pil_image)
                    batch_tensors.append(tensor)

                # Stack zu Batch
                batch = torch.stack(batch_tensors).to(self.device)

                # Single Forward Pass für alle Frames
                with torch.no_grad():
                    image_features = self.model.encode_image(batch)
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                    # Text-Embeddings (gecacht oder neu berechnen)
                    if labels:
                        text_tokens = clip.tokenize(labels).to(self.device)
                        text_features = self.model.encode_text(text_tokens)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                    else:
                        text_features = self.text_embeddings

                    # Batch-Similarity berechnen
                    similarities = image_features @ text_features.T

                # Ergebnisse formatieren
                results = []
                for sim in similarities.cpu().numpy():
                    frame_result = {}
                    for label, score in zip(use_labels, sim):
                        frame_result[label] = float(score)
                    results.append(frame_result)

                logger.debug(f"Batch-Analyse abgeschlossen: {len(frames)} Frames")
                return results

        except Exception as e:
            logger.error(f"Batch-Analyse fehlgeschlagen: {e}", exc_info=True)
            return [{} for _ in frames]

    def analyze_video(
        self,
        video_path: str | Path,
        sample_interval: float = 1.0,
        max_frames: int = 30,
        progress_callback: Callable[[int, int], None] | None = None,
        use_batch_processing: bool = True,
    ) -> dict[str, any]:
        """
        Analysiert ein Video durch Frame-Sampling.

        K-02 FIX: Path Traversal Protection hinzugefuegt.
        PERFORMANCE: Optional Batch-Processing für höhere Effizienz.

        Args:
            video_path: Pfad zum Video
            sample_interval: Sekunden zwischen Samples
            max_frames: Maximale Anzahl zu analysierender Frames
            progress_callback: Optional callback(current, total) für UI-Updates
            use_batch_processing: True = Batch-Inference (schneller), False = Frame-by-Frame

        Returns:
            Aggregierte Analyse-Ergebnisse
        """
        import cv2

        if not CLIP_AVAILABLE or not self.model:
            return {"error": "CLIP nicht verfügbar"}

        try:
            # K-02 FIX: Validiere Pfad gegen Path Traversal
            try:
                validated_path = validate_file_path(
                    video_path, must_exist=True, extensions=ALLOWED_VIDEO_EXTENSIONS
                )
            except (ValueError, FileNotFoundError) as e:
                logger.error(f"Ungültiger Video-Pfad: {e}")
                return {"error": f"Ungültiger Pfad: {e}"}

            # PERF-02 FIX: Use context manager to ensure VideoCapture is released
            from ..utils.video_utils import open_video

            with open_video(str(validated_path)) as cap:
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                if fps == 0:
                    logger.error("Video-FPS ist 0")
                    return {"error": "Ungültiges Video"}

                frame_interval = int(fps * sample_interval)
                frames_to_analyze = min(total_frames // frame_interval, max_frames)

                logger.info(
                    f"Analysiere Video: {frames_to_analyze} Frames (Intervall: {sample_interval}s, Batch: {use_batch_processing})"
                )

                all_scores = {label: [] for label in self.all_labels}

                if use_batch_processing:
                    # Batch-Processing: Frames sammeln und stapelweise verarbeiten
                    frame_idx = 0
                    analyzed = 0
                    frame_buffer = []

                    while cap.isOpened() and analyzed < frames_to_analyze:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        if frame_idx % frame_interval == 0:
                            frame_buffer.append(frame)
                            analyzed += 1

                            # Batch voll oder letzte Frames?
                            if (
                                len(frame_buffer) >= self.batch_size
                                or analyzed >= frames_to_analyze
                            ):
                                # Batch verarbeiten
                                batch_results = self.analyze_batch(frame_buffer)
                                for result in batch_results:
                                    for label, score in result.items():
                                        all_scores[label].append(score)

                                # Progress-Callback
                                if progress_callback:
                                    progress_callback(analyzed, frames_to_analyze)

                                # Buffer leeren
                                frame_buffer = []

                        frame_idx += 1

                    # Restliche Frames verarbeiten
                    if frame_buffer:
                        batch_results = self.analyze_batch(frame_buffer)
                        for result in batch_results:
                            for label, score in result.items():
                                all_scores[label].append(score)

                else:
                    # Frame-by-Frame Processing (Original-Methode)
                    from tempfile import TemporaryDirectory

                    with TemporaryDirectory() as temp_dir:
                        frame_idx = 0
                        analyzed = 0

                        while cap.isOpened() and analyzed < frames_to_analyze:
                            ret, frame = cap.read()
                            if not ret:
                                break

                            if frame_idx % frame_interval == 0:
                                # Frame speichern und analysieren
                                frame_path = Path(temp_dir) / f"frame_{frame_idx}.jpg"
                                cv2.imwrite(str(frame_path), frame)

                                scores = self.analyze_frame(frame_path)
                                for label, score in scores.items():
                                    all_scores[label].append(score)

                                analyzed += 1

                                # Progress-Callback
                                if progress_callback:
                                    progress_callback(analyzed, frames_to_analyze)

                            frame_idx += 1

                analyzed = (
                    sum(len(scores) for scores in all_scores.values()) // len(self.all_labels)
                    if all_scores
                    else 0
                )

            # VideoCapture automatically released by context manager

            if analyzed == 0:
                return {"error": "Keine Frames analysiert"}

            # Durchschnitt berechnen
            avg_scores = {}
            for label, scores in all_scores.items():
                if scores:
                    avg_scores[label] = sum(scores) / len(scores)

            # Top-Kategorien extrahieren
            scene_type, scene_conf = self._get_top_category(avg_scores, self.SCENE_TYPES)
            mood, mood_conf = self._get_top_category(avg_scores, self.MOODS)
            content, content_conf = self._get_top_category(avg_scores, self.CONTENT_TYPES)

            logger.info(
                f"Video-Analyse komplett: Scene={scene_type}, Mood={mood}, Content={content}"
            )

            return {
                "scene_type": scene_type,
                "scene_confidence": scene_conf,
                "mood": mood,
                "mood_confidence": mood_conf,
                "content": content,
                "content_confidence": content_conf,
                "all_scores": avg_scores,
                "frames_analyzed": analyzed,
            }

        except Exception as e:
            logger.error(f"Video-Analyse fehlgeschlagen: {e}", exc_info=True)
            return {"error": str(e)}

    def analyze_videos_parallel(
        self, video_paths: list[str | Path], max_workers: int = 4, **analyze_kwargs
    ) -> list[dict[str, any]]:
        """
        Analysiert mehrere Videos parallel mit ThreadPoolExecutor.

        Args:
            video_paths: Liste von Video-Pfaden
            max_workers: Maximale Anzahl paralleler Worker
            **analyze_kwargs: Weitere Parameter für analyze_video()

        Returns:
            Liste von Analyse-Ergebnissen (gleiche Reihenfolge wie video_paths)
        """
        if not video_paths:
            return []

        logger.info(
            f"Starte parallele Analyse von {len(video_paths)} Videos (Workers: {max_workers})"
        )

        def analyze_single(video_path):
            try:
                return self.analyze_video(video_path, **analyze_kwargs)
            except Exception as e:
                logger.error(f"Parallele Analyse fehlgeschlagen für {video_path}: {e}")
                return {"error": str(e), "path": str(video_path)}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(analyze_single, video_paths))

        logger.info(f"Parallele Analyse abgeschlossen: {len(results)} Ergebnisse")
        return results

    def _get_top_category(
        self, scores: dict[str, float], categories: list[str]
    ) -> tuple[str, float]:
        """
        Findet die Top-Kategorie aus einer Gruppe.

        Args:
            scores: Alle Scores
            categories: Liste von Kategorien zum Filtern

        Returns:
            (top_label, confidence)
        """
        filtered = {k: v for k, v in scores.items() if k in categories}
        if not filtered:
            return ("unknown", 0.0)

        top_label = max(filtered, key=filtered.get)
        return (top_label, filtered[top_label])


# Convenience-Funktion
def is_clip_available() -> bool:
    """Prüft ob CLIP verfügbar ist."""
    return CLIP_AVAILABLE
