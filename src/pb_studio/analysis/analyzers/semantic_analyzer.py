import hashlib
import io
from collections import OrderedDict
from collections.abc import Callable
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from ...utils.gpu_memory import get_gpu_memory_info
from ...utils.logger import get_logger

logger = get_logger(__name__)

# Try to import AI dependencies
try:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    torch = None
    CLIPModel = None
    CLIPProcessor = None

# HIGH-02 FIX: Whitelist erlaubter CLIP-Modelle
ALLOWED_CLIP_MODELS = {
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-base-patch16",
    "openai/clip-vit-large-patch14",
    "openai/clip-vit-large-patch14-336",
}

# MEDIUM-03: Timeout fuer Model-Download (Sekunden)
MODEL_DOWNLOAD_TIMEOUT = 300


def _get_best_device() -> tuple[Union[str, "torch.device"], str]:
    """
    Ermittelt das beste verfuegbare Device (DirectML > CUDA > CPU).

    Returns:
        Tuple (device, device_type_string)
    """
    if not AI_AVAILABLE:
        return "cpu", "CPU"

    # Nutze dediziertes DirectML Device (RX 7800 XT statt integrierte GPU)
    try:
        from ...utils.gpu_memory import get_best_directml_device

        result = get_best_directml_device()
        if result is not None:
            device, idx, name = result
            logger.info(f"DirectML Device verfuegbar: {name} (AMD GPU)")
            return device, "DirectML"
    except ImportError:
        pass

    # Fallback auf Standard torch_directml
    try:
        import torch_directml

        device = torch_directml.device()
        logger.info("DirectML Device verfuegbar (AMD GPU)")
        return device, "DirectML"
    except ImportError:
        pass

    # CUDA fuer NVIDIA GPUs
    if torch.cuda.is_available():
        logger.info("CUDA Device verfuegbar (NVIDIA GPU)")
        return "cuda", "CUDA"

    # Fallback CPU
    logger.info("Kein GPU verfuegbar, nutze CPU")
    return "cpu", "CPU"


class SemanticAnalyzer:
    """
    Semantic Image Analysis using CLIP (Contrastive Language-Image Pre-Training).
    Allows detecting concepts in images using text descriptions.

    Optimiert fuer AMD GPU via torch-directml.

    Confidence-Threshold:
        Bei Softmax-basierten Scores wird ein Threshold angewendet.
        Liegt der hoechste Score unter dem Threshold, wird "unknown" zurueckgegeben.
        Dies verhindert, dass Bilder die zu KEINEM Label passen, trotzdem hohe Scores bekommen.
    """

    # Confidence-Schwellwert: Hoechster Score muss mindestens so hoch sein
    # HINWEIS: Softmax erzeugt oft Scores >0.5 selbst bei schlechten Matches!
    # Empirisch getestet: Noise/Pattern-Bilder liefern 0.5-0.6
    # Default 0.60 = konservativ, vermeidet False Positives
    DEFAULT_CONFIDENCE_THRESHOLD = 0.60

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        """
        Initialize SemanticAnalyzer.

        Args:
            model_name: HuggingFace model hub name for CLIP model.
                        Default: openai/clip-vit-base-patch32 (balanced speed/acc)
            confidence_threshold: Minimum confidence (0.0-1.0) fuer Label-Zuordnung.
                                 Liegt der hoechste Score darunter, gilt das Bild als "unknown".
                                 Default: 0.25 (25%)

        Raises:
            ValueError: Wenn model_name nicht in der Whitelist ist
        """
        # HIGH-02 FIX: Validiere model_name gegen Whitelist
        if model_name not in ALLOWED_CLIP_MODELS:
            raise ValueError(
                f"Model '{model_name}' nicht erlaubt. "
                f"Erlaubt: {', '.join(sorted(ALLOWED_CLIP_MODELS))}"
            )

        self.model_name = model_name
        self.confidence_threshold = max(0.0, min(1.0, confidence_threshold))

        # Helper caches (always init to avoid attribute errors)
        self._image_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._text_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_capacity = 1000
        self._cache_stats = {"image_hits": 0, "image_misses": 0, "text_hits": 0, "text_misses": 0}
        self._model = None
        self._processor = None

        if not AI_AVAILABLE:
            logger.warning(
                "AI dependencies (torch/transformers) not found. Semantic features disabled."
            )
            self.device = "cpu"
            self._device_type = "CPU"
            return

        self.device, self._device_type = _get_best_device()

        logger.info(
            f"SemanticAnalyzer configured with {model_name} on {self._device_type} "
            f"(confidence_threshold={self.confidence_threshold:.2f})"
        )

    def _load_model(self):
        """Lazy load model and processor with timeout."""
        if not AI_AVAILABLE:
            return

        if self._model is None:
            logger.info(f"Loading CLIP model: {self.model_name}...")
            try:
                import socket

                # MEDIUM-03 FIX: Setze Timeout fuer Downloads
                original_timeout = socket.getdefaulttimeout()
                socket.setdefaulttimeout(MODEL_DOWNLOAD_TIMEOUT)

                try:
                    # Use safetensors format to bypass CVE-2025-32434 (PyTorch < 2.6)
                    self._model = CLIPModel.from_pretrained(
                        self.model_name, use_safetensors=True
                    ).to(self.device)
                    # FIX: Set model to eval mode for inference (disables dropout, batch norm training)
                    self._model.eval()
                    self._processor = CLIPProcessor.from_pretrained(self.model_name)
                    logger.info("CLIP model loaded successfully (safetensors format, eval mode)")
                finally:
                    # Stelle originalen Timeout wieder her
                    socket.setdefaulttimeout(original_timeout)

            except Exception as e:
                logger.error(f"Failed to load CLIP model: {e}")
                raise

    def analyze(self, image: np.ndarray | Image.Image, labels: list[str]) -> dict[str, float]:
        """
        Analyze image against a list of text labels.

        Args:
            image: Input image (numpy array or PIL Image)
            labels: List of text labels to check (e.g., ["beach", "party", "dark"])

        Returns:
            Dictionary mapping labels to probability scores (0.0 - 1.0).
            Falls der hoechste Score unter dem confidence_threshold liegt,
            wird {"unknown": 1.0} zurueckgegeben.
        """
        if not labels:
            return {}

        if not AI_AVAILABLE:
            return {"unknown": 1.0}

        self._load_model()

        try:
            # Convert numpy to PIL if needed
            if isinstance(image, np.ndarray):
                # Ensure RGB
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # MEDIUM-04 FIX: OpenCV is BGR, PIL/CLIP expects RGB
                    image = image[:, :, ::-1]  # BGR to RGB
                image = Image.fromarray(image)

            inputs = self._processor(
                text=labels, images=image, return_tensors="pt", padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            # image-text similarity scores
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)  # Softmax erzwingt Summe = 1

            probs_list = probs.cpu().numpy()[0]
            max_prob = float(probs_list.max())

            # SOFTMAX-PROBLEM FIX: Confidence-Threshold
            # Wenn der hoechste Score zu niedrig ist, passt das Bild zu KEINEM Label
            if max_prob < self.confidence_threshold:
                logger.debug(
                    f"Max confidence {max_prob:.3f} below threshold {self.confidence_threshold:.3f}, "
                    f"returning 'unknown'"
                )
                return {"unknown": 1.0}

            result = {label: float(score) for label, score in zip(labels, probs_list)}

            return result

        except Exception as e:
            logger.error(f"Semantic analysis failed: {e}", exc_info=True)
            return {}

    def analyze_batch(
        self,
        images: list[np.ndarray | Image.Image],
        labels: list[str],
        batch_size: int | None = None,
        progress_callback: Callable | None = None,
    ) -> list[dict[str, float]]:
        """
        Batched Analyse mehrerer Bilder gegen gemeinsame Labels.

        Args:
            images: Liste von Bildern (numpy/PIL)
            labels: Liste der Text-Labels
            batch_size: Optionales Batch-Override (Default: automatisch)
            progress_callback: Callback(processed, total)

        Returns:
            Liste von Dictionaries mit Label->Score Mapping.
            Falls max_score < confidence_threshold: {"unknown": 1.0}
        """
        if not images or not labels:
            return []

        if not AI_AVAILABLE:
            return [{"unknown": 1.0} for _ in range(len(images))]

        self._load_model()
        bs = batch_size or self._suggest_batch_size()
        results: list[dict[str, float]] = []

        # Vorverarbeitung: numpy -> PIL
        prepped: list[Image.Image] = []
        for img in images:
            if isinstance(img, np.ndarray):
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = img[:, :, ::-1]
                img = Image.fromarray(img)
            prepped.append(img)

        total = len(prepped)
        for start in range(0, total, bs):
            chunk = prepped[start : start + bs]
            inputs = self._processor(
                text=labels, images=chunk, return_tensors="pt", padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits_per_image  # (batch, labels)
                probs = logits.softmax(dim=1).cpu().numpy()

            for row in probs:
                max_prob = float(row.max())

                # SOFTMAX-PROBLEM FIX: Confidence-Threshold auch in Batch
                if max_prob < self.confidence_threshold:
                    results.append({"unknown": 1.0})
                else:
                    results.append({label: float(score) for label, score in zip(labels, row)})

            if progress_callback:
                processed = min(start + bs, total)
                progress_callback(processed, total)

        # FIX: Clear GPU memory after batch processing to prevent memory leaks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    def get_embedding(self, image: np.ndarray | Image.Image) -> list[float] | None:
        """
        Get image embedding vector.

        Args:
           image: Input image

        Returns:
            List of floats representing the embedding vector, or None on error.
        """
        if not AI_AVAILABLE:
            return None

        self._load_model()
        try:
            cache_key = self._get_image_cache_key(image)
            cached = self._get_cached_image_embedding(cache_key)
            if cached is not None:
                return cached

            # Convert numpy to PIL if needed
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = image[:, :, ::-1]  # BGR to RGB
                image = Image.fromarray(image)

            inputs = self._processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                image_features = self._model.get_image_features(**inputs)

            # Normalize
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

            embedding = image_features.cpu().numpy()[0].tolist()
            self._set_cached_image_embedding(cache_key, embedding)
            return embedding

        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}", exc_info=True)
            return None

    def get_text_embedding(self, text: str) -> list[float] | None:
        """
        Get text embedding vector for search queries.

        Args:
            text: Input text (e.g. "party", "beach sunset")

        Returns:
            List of floats (embedding vector) or None on error.
        """
        if not AI_AVAILABLE:
            return None

        self._load_model()
        try:
            cache_key = self._get_text_cache_key(text)
            cached = self._text_cache.get(cache_key)
            if cached is not None:
                # Cache hit
                self._cache_stats["text_hits"] += 1
                self._text_cache.move_to_end(cache_key)
                return cached

            # Cache miss
            self._cache_stats["text_misses"] += 1
            inputs = self._processor(text=[text], return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                text_features = self._model.get_text_features(**inputs)

            # Normalize
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

            embedding = text_features.cpu().numpy()[0].tolist()
            self._text_cache[cache_key] = embedding
            self._evict_lru(self._text_cache)
            return embedding

        except Exception as e:
            logger.error(f"Text embedding extraction failed: {e}", exc_info=True)
            return None

    def get_device_type(self) -> str:
        """Gibt den aktiven Device-Typ zurueck (DirectML, CUDA, CPU)."""
        return self._device_type

    # ==================== Cache Helpers ====================

    def _evict_lru(self, cache: OrderedDict):
        """Entfernt aelteste Eintraege wenn Cache-Groesse ueberschritten wird."""
        while len(cache) > self._cache_capacity:
            cache.popitem(last=False)

    def _get_image_cache_key(self, image: np.ndarray | Image.Image) -> str:
        """Bild-Hash fuer Cache (verwendet Bytes + Shape/Mode)."""
        hasher = hashlib.md5()
        if isinstance(image, np.ndarray):
            hasher.update(str(image.shape).encode())
            hasher.update(str(image.dtype).encode())
            hasher.update(image.tobytes())
        elif isinstance(image, Image.Image):
            hasher.update(str((image.size, image.mode)).encode())
            with io.BytesIO() as buf:
                image.save(buf, format="PNG")
                hasher.update(buf.getvalue())
        else:
            hasher.update(str(type(image)).encode())
        return hasher.hexdigest()

    def _get_text_cache_key(self, text: str) -> str:
        """Text-Hash fuer Cache (einfacher MD5-Hash des Textes)."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _get_cached_image_embedding(self, key: str) -> list[float] | None:
        if key in self._image_cache:
            self._cache_stats["image_hits"] += 1
            self._image_cache.move_to_end(key)
            return self._image_cache[key]
        self._cache_stats["image_misses"] += 1
        return None

    def _set_cached_image_embedding(self, key: str, embedding: list[float]) -> None:
        self._image_cache[key] = embedding
        self._evict_lru(self._image_cache)

    def clear_cache(self) -> None:
        """Leert alle Embedding-Caches und Statistiken."""
        self._image_cache.clear()
        self._text_cache.clear()
        self._cache_stats = {"image_hits": 0, "image_misses": 0, "text_hits": 0, "text_misses": 0}

    def get_cache_stats(self) -> dict[str, int | float]:
        """
        Hole detaillierte Cache-Statistiken.

        Returns:
            Dictionary mit Cache-Metriken:
            - hits/misses fuer Image/Text
            - hit_rate (0.0-1.0)
            - cache_sizes
            - capacity
        """
        image_total = self._cache_stats["image_hits"] + self._cache_stats["image_misses"]
        text_total = self._cache_stats["text_hits"] + self._cache_stats["text_misses"]

        image_hit_rate = self._cache_stats["image_hits"] / image_total if image_total > 0 else 0.0
        text_hit_rate = self._cache_stats["text_hits"] / text_total if text_total > 0 else 0.0

        return {
            "image_hits": self._cache_stats["image_hits"],
            "image_misses": self._cache_stats["image_misses"],
            "image_hit_rate": image_hit_rate,
            "text_hits": self._cache_stats["text_hits"],
            "text_misses": self._cache_stats["text_misses"],
            "text_hit_rate": text_hit_rate,
            "image_cache_size": len(self._image_cache),
            "text_cache_size": len(self._text_cache),
            "capacity": self._cache_capacity,
        }

    def cache_stats(self) -> dict[str, int]:
        """
        DEPRECATED: Use get_cache_stats() instead.
        Liefert Cache-Metriken fuer Debugging.
        """
        return {
            "image_cache_size": len(self._image_cache),
            "text_cache_size": len(self._text_cache),
            "capacity": self._cache_capacity,
        }

    def _suggest_batch_size(self) -> int:
        """
        Schaetzt sinnvolle Batchsize anhand Device/GPU-Speicher.

        PERF-FIX: CPU batch_size=1 da CLIP zu speicherintensiv fuer Batch>1 auf CPU.
        DirectML batch_size reduziert da langsamer als CUDA.
        """
        # CPU: CLIP ist sehr speicherintensiv, batch=1 ist sicherer
        if self._device_type == "CPU":
            return 1

        info = get_gpu_memory_info() or {}
        total = info.get("total_gb") or 0

        # DirectML ist langsamer als CUDA, konservativere Batch-Sizes
        if self._device_type == "DirectML":
            if total >= 8:
                return 4
            return 2

        # CUDA mit gutem VRAM
        if total >= 16:
            return 16
        if total >= 8:
            return 8
        if total >= 4:
            return 4
        return 2
