"""
Scene Analyzer - Szenentyp-Analyse fuer Video-Clips.

Analysiert:
- Szenentypen (Portrait, Landschaft, Abstract, etc.)
- Edge Density und Textur
- Gesichtserkennung
- Tiefenschaerfe-Schaetzung
"""

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ...utils.logger import get_logger

logger = get_logger()


class SceneType:
    """Definiert moegliche Szenentypen."""

    PORTRAIT = "PORTRAIT"
    LANDSCAPE = "LANDSCAPE"
    CLOSEUP = "CLOSEUP"
    WIDE_SHOT = "WIDE_SHOT"
    ABSTRACT = "ABSTRACT"
    GEOMETRIC = "GEOMETRIC"
    BUSY = "BUSY"
    MINIMAL = "MINIMAL"
    CROWD = "CROWD"
    NATURE = "NATURE"
    URBAN = "URBAN"
    INDOOR = "INDOOR"


@dataclass
class SceneAnalysisResult:
    """Ergebnis der Szenentyp-Analyse."""

    scene_types: list[str]  # Multi-Label Szenentypen
    edge_density: float  # 0-1
    texture_variance: float  # Textur-Komplexitaet
    center_ratio: float  # Anteil Content in Bildmitte
    depth_of_field: float  # Geschaetzte Schaerfentiefe
    has_face: bool
    face_count: int
    face_size_ratio: float  # Groesse des groessten Gesichts relativ zum Bild
    confidence_scores: dict[str, float]  # Confidence pro Szenentyp

    def to_dict(self) -> dict:
        """Konvertiert zu Dictionary fuer DB-Speicherung."""
        return {
            "scene_types": self.scene_types,
            "edge_density": self.edge_density,
            "texture_variance": self.texture_variance,
            "center_ratio": self.center_ratio,
            "depth_of_field": self.depth_of_field,
            "has_face": self.has_face,
            "face_count": self.face_count,
            "face_size_ratio": self.face_size_ratio,
            "confidence_scores": self.confidence_scores,
        }


class SceneAnalyzer:
    """Analysiert Szenentyp und -eigenschaften."""

    def __init__(self, face_confidence_threshold: float = 0.5):
        """
        Initialisiert den SceneAnalyzer.

        Args:
            face_confidence_threshold: Confidence-Schwellenwert fuer DNN Face Detection (0-1)
        """
        self._face_net = None
        self._face_net_loaded = False
        self._face_cascade = None
        self._face_cascade_loaded = False
        self.face_confidence_threshold = face_confidence_threshold

    @property
    def face_net(self):
        """Lazy-Loading des DNN Face Detector (bevorzugt)."""
        if not self._face_net_loaded:
            try:
                # Modell-Pfade (im OpenCV DNN-Modul enthalten oder separat herunterladbar)
                # ROBUST FIX: Use configurable path or default 'data/models' relative to project root

                # Check for env var override
                env_model_dir = os.environ.get("PB_MODEL_DIR")
                if env_model_dir:
                    model_dir = Path(env_model_dir) / "face_detection"
                else:
                    # Default: data/models relative to project root (4 levels up from this file)
                    # src/pb_studio/analysis/analyzers/ -> ../../../../data/models
                    project_root = Path(__file__).parent.parent.parent.parent.parent
                    model_dir = project_root / "data" / "models" / "face_detection"

                # Create if not exists (so user knows where to put models)
                if not model_dir.exists():
                     try:
                         model_dir.mkdir(parents=True, exist_ok=True)
                     except Exception:
                         pass

                prototxt_path = model_dir / "deploy.prototxt"
                model_path = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"

                if prototxt_path.exists() and model_path.exists():
                    self._face_net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(model_path))
                    logger.info("DNN Face Detector erfolgreich geladen")
                else:
                    logger.info(
                        "DNN Face Detector Modelle nicht gefunden. "
                        f"Suche in: {model_dir}\n"
                        "Fallback auf Haar Cascade."
                    )

                self._face_net_loaded = True

            except Exception as e:
                logger.warning(f"DNN Face Detector konnte nicht geladen werden: {e}")
                self._face_net_loaded = True  # Nicht erneut versuchen

        return self._face_net

    @property
    def face_cascade(self):
        """Lazy-Loading des Face Cascade Classifiers (Fallback)."""
        if not self._face_cascade_loaded:
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self._face_cascade = cv2.CascadeClassifier(cascade_path)
                self._face_cascade_loaded = True
                logger.info("Haar Cascade Face Detector geladen (Fallback)")
            except Exception as e:
                logger.warning(f"Face Cascade konnte nicht geladen werden: {e}")
                self._face_cascade_loaded = True  # Nicht erneut versuchen
        return self._face_cascade

    def analyze(self, frame: np.ndarray) -> SceneAnalysisResult:
        """
        Fuehrt vollstaendige Szenen-Analyse durch.

        Args:
            frame: OpenCV Frame (BGR)

        Returns:
            SceneAnalysisResult mit allen Analyse-Daten
        """
        if frame is None or frame.size == 0:
            return self._empty_result()

        try:
            # Metriken berechnen
            edge_density = self._compute_edge_density(frame)
            texture_variance = self._compute_texture_variance(frame)
            center_ratio = self._compute_center_ratio(frame)
            depth_of_field = self._estimate_depth_of_field(frame)

            # Gesichtserkennung
            has_face, face_count, face_size_ratio = self._detect_faces(frame)

            # Szenentypen klassifizieren
            scene_types, confidence_scores = self._classify_scene(
                frame,
                edge_density,
                texture_variance,
                center_ratio,
                has_face,
                face_count,
                face_size_ratio,
            )

            return SceneAnalysisResult(
                scene_types=scene_types,
                edge_density=round(edge_density, 3),
                texture_variance=round(texture_variance, 3),
                center_ratio=round(center_ratio, 3),
                depth_of_field=round(depth_of_field, 3),
                has_face=has_face,
                face_count=face_count,
                face_size_ratio=round(face_size_ratio, 3),
                confidence_scores=confidence_scores,
            )

        except Exception as e:
            logger.error(f"Fehler bei Szenen-Analyse: {e}")
            return self._empty_result()

    def _compute_edge_density(self, frame: np.ndarray) -> float:
        """Berechnet Edge Density (0-1)."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Canny Edge Detection
            edges = cv2.Canny(gray, 50, 150)

            # Anteil der Edge-Pixel
            edge_pixels = np.count_nonzero(edges)
            total_pixels = edges.size

            return edge_pixels / total_pixels

        except Exception:
            return 0.0

    def _compute_texture_variance(self, frame: np.ndarray) -> float:
        """Berechnet Textur-Komplexitaet via Laplacian Variance."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Laplacian fuer Textur-Messung
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()

            # Normalisieren (typischer Bereich 0-5000)
            return min(variance / 5000, 1.0)

        except Exception:
            return 0.0

    def _compute_center_ratio(self, frame: np.ndarray) -> float:
        """Berechnet Anteil der Aktivitaet in der Bildmitte."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # Mittlerer Bereich (50% des Bildes)
            margin_x = w // 4
            margin_y = h // 4

            center = gray[margin_y : h - margin_y, margin_x : w - margin_x]
            full = gray

            # Varianz als Aktivitaetsmass
            center_var = np.var(center)
            full_var = np.var(full)

            if full_var < 1:
                return 0.5

            return min(center_var / full_var, 2.0) / 2.0

        except Exception:
            return 0.5

    def _estimate_depth_of_field(self, frame: np.ndarray) -> float:
        """
        Schaetzt Tiefenschaerfe basierend auf lokaler Schaerfe-Varianz.

        Returns:
            0 = flache DoF (Bokeh), 1 = tiefe DoF (alles scharf)
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # In Regionen aufteilen
            regions = []
            for y in range(0, h, h // 3):
                for x in range(0, w, w // 3):
                    region = gray[y : y + h // 3, x : x + w // 3]
                    if region.size > 0:
                        # Lokale Schaerfe via Laplacian
                        sharpness = cv2.Laplacian(region, cv2.CV_64F).var()
                        regions.append(sharpness)

            if not regions:
                return 0.5

            # Gleichmaessigkeit der Schaerfe
            mean_sharpness = np.mean(regions)
            std_sharpness = np.std(regions)

            if mean_sharpness < 1:
                return 0.5

            # Niedriger CV = gleichmaessig scharf = tiefe DoF
            cv = std_sharpness / mean_sharpness
            dof = 1.0 - min(cv, 1.0)

            return dof

        except Exception:
            return 0.5

    def _detect_faces(self, frame: np.ndarray) -> tuple[bool, int, float]:
        """
        Erkennt Gesichter im Frame mit DNN Face Detector (bevorzugt) oder Haar Cascade (Fallback).

        Returns:
            (has_face, face_count, largest_face_ratio)
        """
        h, w = frame.shape[:2]

        # Bevorzugt: DNN Face Detector
        if self.face_net is not None:
            try:
                return self._detect_faces_dnn(frame, h, w)
            except Exception as e:
                logger.warning(f"DNN Face Detection fehlgeschlagen, versuche Fallback: {e}")

        # Fallback: Haar Cascade
        if self.face_cascade is not None:
            try:
                return self._detect_faces_cascade(frame, h, w)
            except Exception as e:
                logger.debug(f"Haar Cascade Face Detection fehlgeschlagen: {e}")

        # Kein Detektor verfuegbar
        return False, 0, 0.0

    def _detect_faces_dnn(self, frame: np.ndarray, h: int, w: int) -> tuple[bool, int, float]:
        """
        Gesichtserkennung mit DNN (Caffe ResNet-10 300x300).

        Args:
            frame: BGR Frame
            h, w: Frame-Dimensionen

        Returns:
            (has_face, face_count, largest_face_ratio)
        """
        # Blob erstellen (300x300, mean subtraction)
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
            swapRB=False,
            crop=False,
        )

        # Inferenz
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        # Detections parsen (Format: [1, 1, N, 7])
        # Jedes Detection: [batch_id, class_id, confidence, x1, y1, x2, y2]
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Confidence-Filter
            if confidence > self.face_confidence_threshold:
                # Bounding Box (normalisierte Koordinaten)
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)

                # Validierung
                if x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h and x2 > x1 and y2 > y1:
                    faces.append((x1, y1, x2 - x1, y2 - y1))  # (x, y, width, height)

        if len(faces) == 0:
            return False, 0, 0.0

        # Groesstes Gesicht finden
        largest_area = 0
        for x, y, fw, fh in faces:
            area = fw * fh
            if area > largest_area:
                largest_area = area

        # Verhaeltnis zum Gesamtbild
        image_area = h * w
        face_ratio = largest_area / image_area

        return True, len(faces), face_ratio

    def _detect_faces_cascade(self, frame: np.ndarray, h: int, w: int) -> tuple[bool, int, float]:
        """
        Gesichtserkennung mit Haar Cascade (Fallback).

        Args:
            frame: BGR Frame
            h, w: Frame-Dimensionen

        Returns:
            (has_face, face_count, largest_face_ratio)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Gesichtserkennung
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) == 0:
            return False, 0, 0.0

        # Groesstes Gesicht finden
        largest_area = 0
        for x, y, fw, fh in faces:
            area = fw * fh
            if area > largest_area:
                largest_area = area

        # Verhaeltnis zum Gesamtbild
        image_area = h * w
        face_ratio = largest_area / image_area

        return True, len(faces), face_ratio

    def _classify_scene(
        self,
        frame: np.ndarray,
        edge_density: float,
        texture_variance: float,
        center_ratio: float,
        has_face: bool,
        face_count: int,
        face_size_ratio: float,
    ) -> tuple[list[str], dict[str, float]]:
        """
        Score-basierte Scene-Klassifikation mit Top-2 Selection.

        Behebt überlappende Thresholds durch:
        - Normalisierte Score-Berechnung für alle Scene Types (0-1)
        - Top-2 Selection (Primary + Secondary)
        - Mutual Exclusion Rules für widersprüchliche Szenen
        - Höhere Sensitivität durch niedrigere Thresholds

        Returns:
            (scene_types, confidence_scores) - Primary + optional Secondary
        """
        scores = {}
        h, w = frame.shape[:2]
        aspect_ratio = w / h if h > 0 else 1.0

        # Berechne normalisierte Scores für alle Scene Types (0-1)

        # PORTRAIT: Gesicht im Bild
        if has_face and face_size_ratio > 0.03:
            if face_size_ratio > 0.12:
                # Großes Gesicht -> eher CLOSEUP
                scores[SceneType.PORTRAIT] = 0.5
                scores[SceneType.CLOSEUP] = min(0.95, 0.6 + face_size_ratio * 2)
            elif face_size_ratio > 0.06:
                scores[SceneType.PORTRAIT] = 0.8
                scores[SceneType.CLOSEUP] = 0.3
            else:
                scores[SceneType.PORTRAIT] = 0.6
                scores[SceneType.CLOSEUP] = 0
        else:
            scores[SceneType.PORTRAIT] = 0
            scores[SceneType.CLOSEUP] = 0

        # CROWD: Mehrere Gesichter
        if face_count >= 2:
            scores[SceneType.CROWD] = min(0.4 + face_count * 0.15, 0.95)
        else:
            scores[SceneType.CROWD] = 0

        # BUSY: Hohe Edge-Dichte + Textur
        if edge_density > 0.08 and texture_variance > 0.25:
            busy_score = ((edge_density - 0.08) / 0.2 + (texture_variance - 0.25) / 0.5) / 2
            scores[SceneType.BUSY] = min(busy_score + 0.4, 0.95)
        else:
            scores[SceneType.BUSY] = 0

        # MINIMAL: Niedrige Komplexität
        if edge_density < 0.08 and texture_variance < 0.25:
            minimal_score = (0.08 - edge_density) / 0.08 * 0.5 + (
                0.25 - texture_variance
            ) / 0.25 * 0.5
            scores[SceneType.MINIMAL] = max(0.4, minimal_score)
        else:
            scores[SceneType.MINIMAL] = 0

        # ABSTRACT: Textur ohne klare Kanten
        if edge_density < 0.10 and texture_variance > 0.20:
            abstract_score = (texture_variance - 0.20) / 0.6 * 0.6 + (
                0.10 - edge_density
            ) / 0.10 * 0.4
            scores[SceneType.ABSTRACT] = min(abstract_score + 0.3, 0.85)
        else:
            scores[SceneType.ABSTRACT] = 0

        # GEOMETRIC: Klare Kanten ohne Textur
        if edge_density > 0.08 and texture_variance < 0.25:
            geo_score = (edge_density - 0.08) / 0.2 * 0.6 + (0.25 - texture_variance) / 0.25 * 0.4
            scores[SceneType.GEOMETRIC] = min(geo_score + 0.3, 0.85)
        else:
            scores[SceneType.GEOMETRIC] = 0

        # LANDSCAPE: Breites Format + dezentraler Fokus
        if aspect_ratio > 1.3 and center_ratio < 0.6:
            landscape_score = (aspect_ratio - 1.3) / 0.5 * 0.5 + (0.6 - center_ratio) / 0.6 * 0.5
            scores[SceneType.LANDSCAPE] = min(landscape_score + 0.3, 0.85)
        else:
            scores[SceneType.LANDSCAPE] = 0

        # WIDE_SHOT: Keine dominanten Gesichter
        if not has_face or face_size_ratio < 0.03:
            if edge_density > 0.03:
                wide_score = min(edge_density * 2, 0.7)
                scores[SceneType.WIDE_SHOT] = max(0.4, wide_score)
            else:
                scores[SceneType.WIDE_SHOT] = 0.4
        else:
            scores[SceneType.WIDE_SHOT] = 0

        # INDOOR: Zentraler Fokus + moderate Edge-Dichte
        if center_ratio > 0.6 and edge_density < 0.10:
            indoor_score = (center_ratio - 0.6) / 0.4 * 0.5 + (0.10 - edge_density) / 0.10 * 0.5
            scores[SceneType.INDOOR] = max(0.5, indoor_score + 0.3)
        else:
            scores[SceneType.INDOOR] = 0

        # NATURE: Heuristik - weiche Textur + dezentral
        if texture_variance > 0.2 and center_ratio < 0.5 and not has_face:
            nature_score = texture_variance * (1 - center_ratio)
            scores[SceneType.NATURE] = min(nature_score * 1.2, 0.7)
        else:
            scores[SceneType.NATURE] = 0

        # URBAN: Hohe Edge-Dichte + geometrisch
        if edge_density > 0.1 and texture_variance < 0.4:
            urban_score = edge_density * (1 - texture_variance)
            scores[SceneType.URBAN] = min(urban_score * 1.5, 0.7)
        else:
            scores[SceneType.URBAN] = 0

        # Mutual Exclusion Rules
        mutual_exclusive_pairs = [
            (SceneType.PORTRAIT, SceneType.WIDE_SHOT),
            (SceneType.CLOSEUP, SceneType.WIDE_SHOT),
            (SceneType.CLOSEUP, SceneType.LANDSCAPE),
            (SceneType.BUSY, SceneType.MINIMAL),
            (SceneType.ABSTRACT, SceneType.GEOMETRIC),
            (SceneType.NATURE, SceneType.URBAN),
        ]

        for scene1, scene2 in mutual_exclusive_pairs:
            if scene1 in scores and scene2 in scores:
                if scores[scene1] > 0 and scores[scene2] > 0:
                    if scores[scene1] < scores[scene2]:
                        scores[scene1] = 0
                    else:
                        scores[scene2] = 0

        # Top-2 Selection
        sorted_scenes = sorted(
            [(scene, score) for scene, score in scores.items() if score > 0.35],
            key=lambda x: x[1],
            reverse=True,
        )

        scene_types = []
        if len(sorted_scenes) > 0:
            scene_types.append(sorted_scenes[0][0])  # Primary
        if len(sorted_scenes) > 1 and sorted_scenes[1][1] > 0.45:
            # Secondary nur wenn deutlich über Threshold
            scene_types.append(sorted_scenes[1][0])

        # Fallback wenn keine Klassifikation
        if not scene_types:
            scene_types.append(SceneType.WIDE_SHOT)
            scores[SceneType.WIDE_SHOT] = 0.5

        # Confidence-Scores runden (alle > 0.25 für Analyse)
        confidence_scores = {k: round(v, 2) for k, v in scores.items() if v > 0.25}

        return scene_types, confidence_scores

    def _empty_result(self) -> SceneAnalysisResult:
        """Gibt leeres Ergebnis zurueck."""
        return SceneAnalysisResult(
            scene_types=[SceneType.WIDE_SHOT],
            edge_density=0.0,
            texture_variance=0.0,
            center_ratio=0.5,
            depth_of_field=0.5,
            has_face=False,
            face_count=0,
            face_size_ratio=0.0,
            confidence_scores={},
        )

    def analyze_video(
        self, video_path: str, positions: list[str] = None
    ) -> dict[str, SceneAnalysisResult]:
        """
        Analysiert mehrere Frames aus einem Video.

        Args:
            video_path: Pfad zum Video
            positions: Frame-Positionen ['start', 'middle', 'end']

        Returns:
            Dict {position: SceneAnalysisResult}
        """
        if positions is None:
            positions = ["middle"]

        results = {}

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                cap.release()  # PERF-02 FIX: Release even on failed open
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

            cap.release()

        except Exception as e:
            logger.error(f"Fehler bei Video-Szenenanalyse: {e}")
            results = {pos: self._empty_result() for pos in positions}

        return results

    def get_composition_info(self, frame: np.ndarray) -> dict:
        """
        Analysiert Bildkomposition (Rule of Thirds, etc.).

        Returns:
            Dict mit Kompositions-Infos
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # Drittelpunkte
            third_x = w // 3
            third_y = h // 3

            # Aktivitaet in den Dritteln messen
            regions = {
                "top_left": gray[0:third_y, 0:third_x],
                "top_center": gray[0:third_y, third_x : 2 * third_x],
                "top_right": gray[0:third_y, 2 * third_x : w],
                "mid_left": gray[third_y : 2 * third_y, 0:third_x],
                "mid_center": gray[third_y : 2 * third_y, third_x : 2 * third_x],
                "mid_right": gray[third_y : 2 * third_y, 2 * third_x : w],
                "bot_left": gray[2 * third_y : h, 0:third_x],
                "bot_center": gray[2 * third_y : h, third_x : 2 * third_x],
                "bot_right": gray[2 * third_y : h, 2 * third_x : w],
            }

            activity = {name: np.var(region) for name, region in regions.items()}

            # Dominante Region finden
            dominant_region = max(activity, key=activity.get)

            # Rule of Thirds Score
            power_points = ["top_left", "top_right", "bot_left", "bot_right"]
            thirds_score = sum(activity[p] for p in power_points) / (4 * max(activity.values()))

            return {
                "dominant_region": dominant_region,
                "thirds_score": round(thirds_score, 2),
                "activity_map": {k: round(v, 1) for k, v in activity.items()},
            }

        except Exception as e:
            logger.error(f"Kompositions-Analyse fehlgeschlagen: {e}")
            return {"dominant_region": "mid_center", "thirds_score": 0.5, "activity_map": {}}
