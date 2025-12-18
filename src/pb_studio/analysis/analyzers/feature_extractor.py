"""
Feature Extractor - Extrahiert Feature-Vektoren fuer Aehnlichkeitssuche.

Erstellt einen 512-dimensionalen Feature-Vektor aus:
- Color Histogram (HSV): 96 dims
- Edge Histogram: 64 dims
- Texture Features: 256 dims
- Spatial Color: 48 dims
- Additional Features: 48 dims
"""

from dataclasses import dataclass

import cv2
import numpy as np

from ...utils.logger import get_logger

logger = get_logger()


@dataclass
class FeatureVector:
    """Feature-Vektor fuer Aehnlichkeitssuche."""

    vector: np.ndarray  # 512-dimensional
    is_normalized: bool = False

    def normalize(self) -> "FeatureVector":
        """Normalisiert den Vektor (L2-Norm)."""
        if self.is_normalized:
            return self

        norm = np.linalg.norm(self.vector)
        if norm > 0:
            normalized = self.vector / norm
        else:
            normalized = self.vector

        return FeatureVector(vector=normalized, is_normalized=True)

    def to_list(self) -> list[float]:
        """Konvertiert zu Liste."""
        return self.vector.tolist()

    @classmethod
    def from_list(cls, data: list[float]) -> "FeatureVector":
        """Erstellt FeatureVector aus Liste."""
        return cls(vector=np.array(data, dtype=np.float32))


class FeatureExtractor:
    """Extrahiert Feature-Vektoren fuer visuelle Aehnlichkeitssuche."""

    VECTOR_SIZE = 512

    def __init__(self, resize_size: tuple[int, int] = (224, 224)):
        """
        Args:
            resize_size: Groesse fuer Frame-Resize vor Extraktion
        """
        self.resize_size = resize_size

        # Feature-Dimensionen
        self.color_hist_dims = 96  # 32 H + 32 S + 32 V
        self.edge_hist_dims = 64  # 8 Richtungen x 8 Regionen
        self.texture_dims = 256  # LBP-aehnlich
        self.spatial_dims = 48  # 3x3 Grid x 3 Channels + Momente
        self.additional_dims = 48  # Zusaetzliche Features

    def extract(self, frame: np.ndarray) -> FeatureVector | None:
        """
        Extrahiert Feature-Vektor aus einem Frame.

        Args:
            frame: OpenCV Frame (BGR)

        Returns:
            FeatureVector oder None bei Fehler
        """
        if frame is None or frame.size == 0:
            return None

        try:
            # Frame vorbereiten
            frame = cv2.resize(frame, self.resize_size)

            # Features extrahieren
            color_features = self._extract_color_histogram(frame)
            edge_features = self._extract_edge_histogram(frame)
            texture_features = self._extract_texture_features(frame)
            spatial_features = self._extract_spatial_features(frame)
            additional_features = self._extract_additional_features(frame)

            # Kombinieren
            vector = np.concatenate(
                [
                    color_features,
                    edge_features,
                    texture_features,
                    spatial_features,
                    additional_features,
                ]
            )

            # Auf exakt 512 Dimensionen bringen
            if len(vector) < self.VECTOR_SIZE:
                vector = np.pad(vector, (0, self.VECTOR_SIZE - len(vector)))
            elif len(vector) > self.VECTOR_SIZE:
                vector = vector[: self.VECTOR_SIZE]

            return FeatureVector(vector=vector.astype(np.float32))

        except Exception as e:
            logger.error(f"Fehler bei Feature-Extraktion: {e}")
            return None

    def _extract_color_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Extrahiert HSV Color Histogram (96 dims)."""
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Histogramme fuer H, S, V
            hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])

            # Normalisieren
            hist_h = cv2.normalize(hist_h, hist_h).flatten()
            hist_s = cv2.normalize(hist_s, hist_s).flatten()
            hist_v = cv2.normalize(hist_v, hist_v).flatten()

            return np.concatenate([hist_h, hist_s, hist_v])

        except Exception:
            return np.zeros(self.color_hist_dims)

    def _extract_edge_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Extrahiert Edge Histogram (64 dims)."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Sobel Gradienten
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            # Magnitude und Richtung
            magnitude = np.sqrt(gx**2 + gy**2)
            angle = np.arctan2(gy, gx) * 180 / np.pi + 180  # 0-360

            # In 8 Regionen aufteilen
            h, w = gray.shape
            regions = []
            for y in range(0, h, h // 4):
                for x in range(0, w, w // 2):
                    region_mag = magnitude[y : y + h // 4, x : x + w // 2]
                    region_ang = angle[y : y + h // 4, x : x + w // 2]

                    # Histogram ueber 8 Richtungen
                    hist, _ = np.histogram(
                        region_ang.flatten(), bins=8, range=(0, 360), weights=region_mag.flatten()
                    )
                    regions.extend(hist / (hist.sum() + 1e-10))

            return np.array(regions)

        except Exception:
            return np.zeros(self.edge_hist_dims)

    def _extract_texture_features(self, frame: np.ndarray) -> np.ndarray:
        """Extrahiert Textur-Features (256 dims)."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Vereinfachtes LBP-aehnliches Feature
            features = []

            # Multi-Scale Laplacian
            for ksize in [3, 5, 7, 9]:
                lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
                hist, _ = np.histogram(lap.flatten(), bins=32, range=(-255, 255))
                features.extend(hist / (hist.sum() + 1e-10))

            # Gabor-aehnliche Features
            for theta in range(0, 180, 45):
                for sigma in [1, 2]:
                    kernel = cv2.getGaborKernel((9, 9), sigma, theta * np.pi / 180, 10.0, 0.5, 0)
                    filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
                    features.append(np.mean(filtered))
                    features.append(np.std(filtered))

            # Auf 256 Dimensionen auffuellen
            features = np.array(features)
            if len(features) < self.texture_dims:
                features = np.pad(features, (0, self.texture_dims - len(features)))

            return features[: self.texture_dims]

        except Exception:
            return np.zeros(self.texture_dims)

    def _extract_spatial_features(self, frame: np.ndarray) -> np.ndarray:
        """Extrahiert Spatial Color Features (48 dims)."""
        try:
            # In 3x3 Grid aufteilen
            h, w = frame.shape[:2]
            features = []

            for y in range(0, h, h // 3):
                for x in range(0, w, w // 3):
                    region = frame[y : y + h // 3, x : x + w // 3]

                    # Durchschnittliche Farbe (BGR)
                    mean_color = np.mean(region, axis=(0, 1))
                    features.extend(mean_color / 255.0)

            # Farbmomente hinzufuegen
            for channel in range(3):
                ch = frame[:, :, channel].astype(np.float64)
                features.append(np.mean(ch) / 255.0)
                features.append(np.std(ch) / 128.0)
                features.append(self._skewness(ch) / 100.0)

            # Hu Momente
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            moments = cv2.moments(gray)
            hu = cv2.HuMoments(moments).flatten()
            features.extend(np.sign(hu) * np.log10(np.abs(hu) + 1e-10))

            features = np.array(features)
            if len(features) < self.spatial_dims:
                features = np.pad(features, (0, self.spatial_dims - len(features)))

            return features[: self.spatial_dims]

        except Exception:
            return np.zeros(self.spatial_dims)

    def _extract_additional_features(self, frame: np.ndarray) -> np.ndarray:
        """Extrahiert zusaetzliche Features (48 dims)."""
        try:
            features = []

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Helligkeit-Statistiken
            features.append(np.mean(gray) / 255.0)
            features.append(np.std(gray) / 128.0)
            features.append(np.median(gray) / 255.0)

            # Saettigungs-Statistiken
            sat = hsv[:, :, 1]
            features.append(np.mean(sat) / 255.0)
            features.append(np.std(sat) / 128.0)

            # Kanten-Dichte
            edges = cv2.Canny(gray, 50, 150)
            features.append(np.mean(edges) / 255.0)

            # Histogramm-Statistiken
            hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
            hist = hist / (hist.sum() + 1e-10)
            features.extend(hist)

            # Textur-Schaerfe
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            features.append(min(lap_var / 5000, 1.0))

            # Dynamic Range
            features.append((np.max(gray) - np.min(gray)) / 255.0)

            # Symmetrie-Berechnung (Horizontal Spiegelung)
            h, w = gray.shape
            if w < 2:
                symmetry = 0.5  # Fallback bei zu kleinem Bild
            else:
                mid = w // 2
                # Linke Haelfte
                left_half = gray[:, :mid]
                # Rechte Haelfte (gespiegelt, nur bis mid)
                right_half = gray[:, mid : 2 * mid][:, ::-1]

                # Boundary Check: Beide Haelften muessen gleiche Groesse haben
                if left_half.shape == right_half.shape and left_half.size > 0:
                    # Differenz zwischen Original und Spiegelbild
                    diff = np.abs(left_half.astype(np.float32) - right_half.astype(np.float32))
                    symmetry = float(np.clip(1.0 - (np.mean(diff) / 255.0), 0.0, 1.0))
                else:
                    # Fallback bei ungerader Breite oder leeren Arrays
                    symmetry = 0.5
            features.append(symmetry)

            features = np.array(features)
            if len(features) < self.additional_dims:
                features = np.pad(features, (0, self.additional_dims - len(features)))

            return features[: self.additional_dims]

        except Exception:
            return np.zeros(self.additional_dims)

    def _skewness(self, data: np.ndarray) -> float:
        """Berechnet Skewness."""
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-10:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)

    def extract_from_video(self, video_path: str, position: str = "middle") -> FeatureVector | None:
        """
        Extrahiert Feature-Vektor aus einem Video.

        Args:
            video_path: Pfad zum Video
            position: Frame-Position ('start', 'middle', 'end')

        Returns:
            FeatureVector oder None
        """
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                cap.release()  # PERF-02 FIX: Release even on failed open
                cap = None
                return None

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if position == "start":
                frame_idx = 0
            elif position == "end":
                frame_idx = max(0, frame_count - 1)
            else:
                frame_idx = frame_count // 2

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret or frame is None:
                return None

            return self.extract(frame)

        except Exception as e:
            logger.error(f"Fehler bei Video-Feature-Extraktion: {e}")
            return None

        finally:
            if cap:
                cap.release()

    def extract_multi_frame(self, video_path: str, num_frames: int = 3) -> FeatureVector | None:
        """
        Extrahiert gemittelten Feature-Vektor aus mehreren Frames.

        Args:
            video_path: Pfad zum Video
            num_frames: Anzahl zu extrahierender Frames

        Returns:
            Gemittelter FeatureVector
        """
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                cap.release()  # PERF-02 FIX: Release even on failed open
                cap = None
                return None

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = frame_count // (num_frames + 1)

            vectors = []
            for i in range(1, num_frames + 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
                ret, frame = cap.read()
                if ret and frame is not None:
                    fv = self.extract(frame)
                    if fv is not None:
                        vectors.append(fv.vector)

            if not vectors:
                return None

            # Durchschnitt berechnen
            avg_vector = np.mean(vectors, axis=0)
            return FeatureVector(vector=avg_vector.astype(np.float32))

        except Exception as e:
            logger.error(f"Fehler bei Multi-Frame-Extraktion: {e}")
            return None

        finally:
            if cap:
                cap.release()

    def save_vector(self, vector: FeatureVector, path: str) -> bool:
        """
        Speichert Feature-Vektor als .npy Datei.

        Args:
            vector: Zu speichernder FeatureVector
            path: Ziel-Pfad (.npy)

        Returns:
            True bei Erfolg
        """
        try:
            np.save(path, vector.vector)
            return True
        except Exception as e:
            logger.error(f"Fehler beim Speichern des Vektors: {e}")
            return False

    def load_vector(self, path: str) -> FeatureVector | None:
        """
        Laedt Feature-Vektor aus .npy Datei.

        Args:
            path: Pfad zur .npy Datei

        Returns:
            FeatureVector oder None
        """
        try:
            vector = np.load(path)
            return FeatureVector(vector=vector.astype(np.float32))
        except Exception as e:
            logger.error(f"Fehler beim Laden des Vektors: {e}")
            return None

    @staticmethod
    def cosine_similarity(v1: FeatureVector, v2: FeatureVector) -> float:
        """
        Berechnet Cosine-Aehnlichkeit zwischen zwei Vektoren.

        Returns:
            Aehnlichkeit 0-1
        """
        # Normalisieren
        v1_norm = v1.normalize()
        v2_norm = v2.normalize()

        # Dot Product (entspricht Cosine bei normalisierten Vektoren)
        similarity = np.dot(v1_norm.vector, v2_norm.vector)

        return float(np.clip(similarity, 0.0, 1.0))
