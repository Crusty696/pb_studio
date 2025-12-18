"""
Object Detector - YOLO-basierte Objekterkennung fuer Video-Clips.

Analysiert:
- Erkannte Objekte (Person, Auto, Tier, etc.)
- Objekt-Anzahl pro Kategorie
- Confidence Scores
- Content-Tags basierend auf Features

Optimiert fuer AMD GPU via ONNX Runtime mit DirectML Provider.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ...ai.model_manager import get_model_manager
from ...utils.logger import get_logger

logger = get_logger(__name__)

# ONNX Runtime Provider Reihenfolge (DirectML fuer AMD GPUs)
ONNX_PROVIDERS = ["DmlExecutionProvider", "CPUExecutionProvider"]

# Konstanten fuer Detection
IOU_THRESHOLD = 0.45  # Non-Maximum Suppression IoU Threshold
DEFAULT_INPUT_SIZE = (640, 640)  # YOLO Input Size
ALLOWED_MODEL_NAMES = {"yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"}  # Erlaubte Modelle


@dataclass
class ObjectDetectionResult:
    """Ergebnis der Objekterkennung."""

    detected_objects: list[str]  # Eindeutige erkannte Objekte
    object_counts: dict[str, int]  # Anzahl pro Objekttyp
    confidence_scores: dict[str, list[float]]  # Confidence pro Objekt
    content_tags: list[str]  # Abgeleitete Content-Tags
    line_count: int  # Anzahl erkannter Linien
    green_ratio: float  # Gruenanteil (Natur)
    sky_ratio: float  # Himmelsanteil
    symmetry: float  # Symmetrie-Score

    def to_dict(self) -> dict:
        """Konvertiert zu Dictionary fuer DB-Speicherung."""
        return {
            "detected_objects": self.detected_objects,
            "object_counts": self.object_counts,
            "confidence_scores": self.confidence_scores,
            "content_tags": self.content_tags,
            "line_count": self.line_count,
            "green_ratio": self.green_ratio,
            "sky_ratio": self.sky_ratio,
            "symmetry": self.symmetry,
        }


# M-08 FIX: Singleton fuer leere Ergebnisse (vermeidet staendige Objekt-Erstellung)
EMPTY_DETECTION_RESULT = ObjectDetectionResult(
    detected_objects=[],
    object_counts={},
    confidence_scores={},
    content_tags=[],
    line_count=0,
    green_ratio=0.0,
    sky_ratio=0.0,
    symmetry=0.5,
)


class ObjectDetector:
    """YOLO-basierte Objekterkennung mit Feature-Tags."""

    # COCO Klassen-Namen (YOLOv8)
    COCO_CLASSES = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    def __init__(self, model_name: str = "yolov8n", confidence: float = 0.3, enabled: bool = True):
        """
        Args:
            model_name: YOLO Modell (yolov8n, yolov8s, etc.)
            confidence: Minimale Confidence fuer Detection
            enabled: Ob YOLO aktiviert ist

        Raises:
            ValueError: Wenn model_name nicht erlaubt ist
        """
        # HIGH-01 FIX: Validiere model_name gegen Whitelist
        if model_name not in ALLOWED_MODEL_NAMES:
            raise ValueError(
                f"Model '{model_name}' nicht erlaubt. "
                f"Erlaubt: {', '.join(sorted(ALLOWED_MODEL_NAMES))}"
            )

        self.model_name = model_name
        self.confidence = max(0.0, min(1.0, confidence))  # Clamp 0-1
        self.enabled = enabled

        # ONNX Runtime Session
        self._session = None
        self._model_loaded = False
        self._active_provider = None

        # Input Size fuer YOLO
        self._input_size = DEFAULT_INPUT_SIZE

        logger.info("ObjectDetector initialized (ONNX Runtime + DirectML)")

    @property
    def session(self):
        """Lazy-Loading der ONNX Runtime Session."""
        if not self.enabled:
            return None

        if not self._model_loaded:
            self._session = self._load_model()
            self._model_loaded = True

        return self._session

    def _load_model(self):
        """Laedt das YOLO ONNX Modell mit DirectML Provider."""
        try:
            import onnxruntime as ort

            # Try to get model from ModelManager (auto-download if needed)
            mm = get_model_manager()
            model_path = mm.download_model(f"{self.model_name}_onnx")

            if model_path is None:
                # Fallback: Check local files (legacy support)
                legacy_path = (
                    Path(__file__).parent.parent.parent.parent.parent / f"{self.model_name}.onnx"
                )
                if legacy_path.exists():
                    model_path = legacy_path
                else:
                    logger.warning(
                        f"ONNX Model '{self.model_name}' could not be loaded/downloaded."
                    )
                    self.enabled = False
                    return None

            logger.info(f"Loading YOLO model from: {model_path}")

            # Session mit DirectML Provider erstellen

            # Session mit DirectML Provider erstellen
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            session = ort.InferenceSession(
                str(model_path), sess_options=sess_options, providers=ONNX_PROVIDERS
            )

            # Aktiven Provider pruefen
            active_providers = session.get_providers()
            self._active_provider = active_providers[0] if active_providers else "Unknown"

            if "DmlExecutionProvider" in active_providers:
                logger.info("YOLO ONNX geladen mit DirectML (AMD GPU)")
            else:
                logger.info("YOLO ONNX geladen mit CPU Fallback")

            return session

        except ImportError:
            logger.warning("onnxruntime nicht installiert - YOLO deaktiviert")
            logger.warning("Installieren mit: pip install onnxruntime-directml")
            self.enabled = False
            return None
        except Exception as e:
            logger.warning(f"YOLO ONNX Modell konnte nicht geladen werden: {e}")
            self.enabled = False
            return None

    def analyze(self, frame: np.ndarray) -> ObjectDetectionResult:
        """
        Fuehrt vollstaendige Objekterkennung durch.

        Args:
            frame: OpenCV Frame (BGR), Shape (H, W, 3), dtype uint8

        Returns:
            ObjectDetectionResult mit allen Analyse-Daten
        """
        # MEDIUM-01 FIX: Input-Validierung
        if frame is None:
            return self._empty_result()

        if not isinstance(frame, np.ndarray):
            logger.warning(f"Frame ist kein np.ndarray: {type(frame)}")
            return self._empty_result()

        if frame.size == 0:
            return self._empty_result()

        if len(frame.shape) != 3 or frame.shape[2] != 3:
            logger.warning(f"Ungueltige Frame-Shape: {frame.shape}, erwartet (H, W, 3)")
            return self._empty_result()

        try:
            # YOLO Detection
            detected_objects, object_counts, confidence_scores = self._detect_objects(frame)

            # Feature-basierte Tags
            line_count = self._count_lines(frame)
            green_ratio = self._compute_green_ratio(frame)
            sky_ratio = self._compute_sky_ratio(frame)
            symmetry = self._compute_symmetry(frame)

            # Content-Tags ableiten
            content_tags = self._derive_content_tags(
                detected_objects, line_count, green_ratio, sky_ratio
            )

            return ObjectDetectionResult(
                detected_objects=detected_objects,
                object_counts=object_counts,
                confidence_scores=confidence_scores,
                content_tags=content_tags,
                line_count=line_count,
                green_ratio=round(green_ratio, 3),
                sky_ratio=round(sky_ratio, 3),
                symmetry=round(symmetry, 3),
            )

        except Exception as e:
            logger.error(f"Fehler bei Objekterkennung: {e}")
            return self._empty_result()

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        """
        Preprocesst Frame fuer YOLO ONNX Inference.

        Returns:
            Tuple (input_tensor, scale, padding)
        """
        h, w = frame.shape[:2]
        target_h, target_w = self._input_size

        # Scale berechnen (letterbox)
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Padding (letterbox)
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2

        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

        # BGR zu RGB
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

        # Normalize und transpose zu BCHW
        blob = rgb.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)  # HWC -> CHW
        blob = np.expand_dims(blob, axis=0)  # CHW -> BCHW

        return blob, scale, (pad_w, pad_h)

    def _postprocess(
        self, outputs: np.ndarray, scale: float, padding: tuple[int, int]
    ) -> list[tuple[int, float]]:
        """
        Postprocesst YOLO Output zu Detections.

        Args:
            outputs: YOLO Output Shape (1, 84, 8400)
            scale: Resize Scale
            padding: (pad_w, pad_h)

        Returns:
            List von (class_id, confidence) Tuples
        """
        # Transpose: (1, 84, 8400) -> (8400, 84)
        predictions = outputs[0].T

        # Confidence Score pro Detection (max class score)
        class_scores = predictions[:, 4:]  # 80 Klassen
        confidences = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        # Filter nach Confidence
        mask = confidences >= self.confidence
        filtered_confidences = confidences[mask]
        filtered_class_ids = class_ids[mask]

        # Bounding Boxes fuer NMS
        boxes = predictions[mask, :4]  # x_center, y_center, w, h

        if len(boxes) == 0:
            return []

        # Convert zu x1, y1, x2, y2 fuer NMS
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

        # NMS mit OpenCV
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(), filtered_confidences.tolist(), self.confidence, IOU_THRESHOLD
        )

        if len(indices) == 0:
            return []

        # Flatten indices (OpenCV gibt manchmal 2D Array zurueck)
        if isinstance(indices, np.ndarray):
            indices = indices.flatten()

        detections = []
        for idx in indices:
            cls_id = int(filtered_class_ids[idx])
            conf = float(filtered_confidences[idx])
            if 0 <= cls_id < len(self.COCO_CLASSES):
                detections.append((cls_id, conf))

        return detections

    def _detect_objects(
        self, frame: np.ndarray
    ) -> tuple[list[str], dict[str, int], dict[str, list[float]]]:
        """Fuehrt YOLO Detection durch mit ONNX Runtime (DirectML)."""
        if not self.enabled or self.session is None:
            return [], {}, {}

        try:
            # Preprocess
            input_tensor, scale, padding = self._preprocess(frame)

            # ONNX Inference
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_tensor})

            # Postprocess
            detections = self._postprocess(outputs[0], scale, padding)

            # Ergebnisse aggregieren
            detected_objects = []
            object_counts = {}
            confidence_scores = {}

            for cls_id, conf in detections:
                class_name = self.COCO_CLASSES[cls_id]

                if class_name not in detected_objects:
                    detected_objects.append(class_name)

                object_counts[class_name] = object_counts.get(class_name, 0) + 1

                if class_name not in confidence_scores:
                    confidence_scores[class_name] = []
                confidence_scores[class_name].append(round(conf, 2))

            return detected_objects, object_counts, confidence_scores

        except Exception as e:
            logger.error(f"YOLO Detection fehlgeschlagen: {e}")
            return [], {}, {}

    def _count_lines(self, frame: np.ndarray) -> int:
        """Zaehlt erkannte Linien (Hough Transform)."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10
            )

            return len(lines) if lines is not None else 0

        except Exception:
            return 0

    def _compute_green_ratio(self, frame: np.ndarray) -> float:
        """Berechnet Gruenanteil (Natur-Indikator)."""
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Gruen-Bereich in HSV
            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 255])

            mask = cv2.inRange(hsv, lower_green, upper_green)
            green_pixels = np.count_nonzero(mask)
            total_pixels = frame.shape[0] * frame.shape[1]

            return green_pixels / total_pixels

        except Exception:
            return 0.0

    def _compute_sky_ratio(self, frame: np.ndarray) -> float:
        """
        Berechnet Himmelsanteil mit mehreren Farbbereichen (Blau, Grau/Weiss, Sonnenuntergang)
        und hoehengewichteter Maske (oberer Bereich zaehlt staerker).
        """
        try:
            h, w = frame.shape[:2]
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Blaues Himmelsfenster
            mask_blue = cv2.inRange(hsv, np.array([90, 40, 50]), np.array([135, 255, 255]))
            # Wolken / Grau-Weiss (niedrige Saettigung, hoher Value)
            mask_cloud = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 50, 255]))
            # Sonnenuntergangs-Orange/Rosa
            mask_sunset1 = cv2.inRange(hsv, np.array([5, 60, 80]), np.array([25, 255, 255]))
            mask_sunset2 = cv2.inRange(hsv, np.array([150, 60, 80]), np.array([180, 255, 255]))

            sky_mask = cv2.bitwise_or(mask_blue, mask_cloud)
            sky_mask = cv2.bitwise_or(sky_mask, mask_sunset1)
            sky_mask = cv2.bitwise_or(sky_mask, mask_sunset2)

            # Hoehengewichtung: oberste Zeile Gewicht 1.0, unterste 0.2
            weights = np.linspace(1.0, 0.2, h, dtype=np.float32).reshape(h, 1)
            weighted_sky = np.sum((sky_mask > 0).astype(np.float32) * weights)
            weighted_total = float(np.sum(weights) * w)

            if weighted_total <= 0:
                return 0.0
            return float(weighted_sky / weighted_total)

        except Exception:
            return 0.0

    def _compute_symmetry(self, frame: np.ndarray) -> float:
        """Berechnet horizontale Symmetrie des Bildes."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            h, w = gray.shape[:2]

            if w < 2:
                return 0.5

            mid = w // 2
            left_half = gray[:, :mid]
            right_half = gray[:, mid : 2 * mid][:, ::-1]  # Horizontal gespiegelt

            # Bei ungerader Breite: rechte Seite um 1 Pixel kürzen
            if left_half.shape != right_half.shape:
                min_w = min(left_half.shape[1], right_half.shape[1])
                left_half = left_half[:, :min_w]
                right_half = right_half[:, :min_w]

            diff = np.abs(left_half.astype(np.float32) - right_half.astype(np.float32))
            symmetry = 1.0 - (np.mean(diff) / 255.0)
            return float(np.clip(symmetry, 0.0, 1.0))

        except Exception:
            return 0.5

    def _derive_content_tags(
        self, objects: list[str], line_count: int, green_ratio: float, sky_ratio: float
    ) -> list[str]:
        """Leitet Content-Tags aus Features ab."""
        tags = []

        # Objekt-basierte Tags
        nature_objects = {
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "potted plant",
        }
        vehicle_objects = {
            "car",
            "motorcycle",
            "bicycle",
            "bus",
            "train",
            "truck",
            "airplane",
            "boat",
        }
        person_related = {"person", "backpack", "handbag", "umbrella"}
        sports_objects = {
            "sports ball",
            "skis",
            "snowboard",
            "skateboard",
            "surfboard",
            "tennis racket",
            "baseball bat",
            "baseball glove",
            "frisbee",
            "kite",
        }

        if any(obj in nature_objects for obj in objects):
            tags.append("ANIMALS")

        if any(obj in vehicle_objects for obj in objects):
            tags.append("VEHICLES")

        if any(obj in person_related for obj in objects):
            tags.append("PEOPLE")

        if any(obj in sports_objects for obj in objects):
            tags.append("SPORTS")

        # Feature-basierte Tags
        if green_ratio > 0.3:
            tags.append("NATURE")
        elif green_ratio > 0.15:
            tags.append("OUTDOOR")

        if sky_ratio > 0.4:
            tags.append("SKY")
            tags.append("OUTDOOR")

        if line_count > 50:
            tags.append("GEOMETRIC")
        elif line_count > 20:
            tags.append("STRUCTURED")

        if green_ratio < 0.05 and sky_ratio < 0.1 and line_count > 30:
            tags.append("URBAN")

        if green_ratio < 0.05 and sky_ratio < 0.1 and line_count < 20:
            tags.append("INDOOR")

        # Duplikate entfernen
        return list(set(tags))

    def _empty_result(self) -> ObjectDetectionResult:
        """Gibt leeres Ergebnis zurueck (M-08 FIX: nutzt Singleton)."""
        return EMPTY_DETECTION_RESULT

    def analyze_video(
        self, video_path: str, positions: list[str] = None
    ) -> dict[str, ObjectDetectionResult]:
        """
        Analysiert mehrere Frames aus einem Video.

        Args:
            video_path: Pfad zum Video
            positions: Frame-Positionen ['start', 'middle', 'end']

        Returns:
            Dict {position: ObjectDetectionResult}
        """
        if positions is None:
            positions = ["middle"]

        results = {}
        cap = None

        # MEDIUM-08 FIX: Add try/finally to ensure cap.release() is always called
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {pos: self._empty_result() for pos in positions}

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            for position in positions:
                if position == "start":
                    frame_idx = 0
                elif position == "end":
                    frame_idx = max(0, frame_count - 1)
                else:
                    frame_idx = frame_count // 2

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if ret and frame is not None:
                    results[position] = self.analyze(frame)
                else:
                    results[position] = self._empty_result()

        except Exception as e:
            logger.error(f"Fehler bei Video-Objekterkennung: {e}")
            results = {pos: self._empty_result() for pos in positions}

        finally:
            # MEDIUM-08 FIX: Ensure VideoCapture is always released
            if cap is not None:
                cap.release()

        return results

    def is_available(self) -> bool:
        """Prueft ob YOLO ONNX verfuegbar ist."""
        if not self.enabled:
            return False

        try:
            # Versuche Session zu laden
            _ = self.session
            return self._session is not None
        except Exception:
            return False

    def set_enabled(self, enabled: bool):
        """Aktiviert/deaktiviert YOLO."""
        self.enabled = enabled
        if not enabled:
            self._session = None
            self._model_loaded = False

    def get_provider(self) -> str:
        """Gibt den aktiven ONNX Runtime Provider zurueck."""
        if self._active_provider:
            return self._active_provider
        return "Not loaded"
