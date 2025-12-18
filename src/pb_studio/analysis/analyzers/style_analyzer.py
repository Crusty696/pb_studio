"""
Style Analyzer - Visual Style Analyse fuer Video-Clips.

Analysiert:
- Visuelle Styles (Vintage, Neon, Filmic, etc.)
- Farbpaletten-Komplexitaet
- Noise/Grain Level
- Schaerfe und Vignettierung
"""

from dataclasses import dataclass

import cv2
import numpy as np

from ...utils.logger import get_logger

logger = get_logger()


class VisualStyle:
    """Definiert moegliche Visual Styles."""

    VINTAGE = "VINTAGE"
    FILMIC = "FILMIC"
    NEON = "NEON"
    MINIMALIST = "MINIMALIST"
    PSYCHEDELIC = "PSYCHEDELIC"
    CINEMATIC = "CINEMATIC"
    DIGITAL = "DIGITAL"
    DREAMY = "DREAMY"
    HIGH_CONTRAST = "HIGH_CONTRAST"
    LOW_KEY = "LOW_KEY"
    HIGH_KEY = "HIGH_KEY"
    STANDARD = "STANDARD"


@dataclass
class StyleAnalysisResult:
    """Ergebnis der Style-Analyse."""

    styles: list[str]  # Erkannte Styles
    unique_colors: int  # Anzahl einzigartiger Farben
    noise_level: float  # Noise/Grain (0-1)
    sharpness: float  # Schaerfe
    vignette_score: float  # Vignettierung (0-1)
    saturation_mean: float  # Durchschnittliche Saettigung
    saturation_std: float  # Saettigungs-Variation
    dynamic_range: float  # Dynamic Range (0-1)
    mean_brightness: float  # Durchschnittliche Helligkeit

    def to_dict(self) -> dict:
        """Konvertiert zu Dictionary fuer DB-Speicherung."""
        return {
            "styles": self.styles,
            "unique_colors": self.unique_colors,
            "noise_level": self.noise_level,
            "sharpness": self.sharpness,
            "vignette_score": self.vignette_score,
            "saturation_mean": self.saturation_mean,
            "saturation_std": self.saturation_std,
            "dynamic_range": self.dynamic_range,
            "mean_brightness": self.mean_brightness,
        }


class StyleAnalyzer:
    """Analysiert visuelle Styles von Video-Frames."""

    def __init__(self):
        """Initialisiert den StyleAnalyzer."""
        self.initialized = True  # Placeholder flag

    def analyze(self, frame: np.ndarray) -> StyleAnalysisResult:
        """
        Fuehrt vollstaendige Style-Analyse durch.

        Args:
            frame: OpenCV Frame (BGR)

        Returns:
            StyleAnalysisResult mit allen Analyse-Daten
        """
        if frame is None or frame.size == 0:
            return self._empty_result()

        try:
            # Metriken berechnen
            unique_colors = self._count_unique_colors(frame)
            noise_level = self._compute_noise_level(frame)
            sharpness = self._compute_sharpness(frame)
            vignette_score = self._compute_vignette(frame)
            sat_mean, sat_std = self._compute_saturation_stats(frame)
            dynamic_range = self._compute_dynamic_range(frame)
            mean_brightness = self._compute_mean_brightness(frame)

            # Styles klassifizieren
            styles = self._classify_styles(
                unique_colors,
                noise_level,
                sharpness,
                vignette_score,
                sat_mean,
                sat_std,
                dynamic_range,
                mean_brightness,
            )

            return StyleAnalysisResult(
                styles=styles,
                unique_colors=unique_colors,
                noise_level=round(noise_level, 3),
                sharpness=round(sharpness, 1),
                vignette_score=round(vignette_score, 3),
                saturation_mean=round(sat_mean, 3),
                saturation_std=round(sat_std, 3),
                dynamic_range=round(dynamic_range, 3),
                mean_brightness=round(mean_brightness, 1),
            )

        except Exception as e:
            logger.error(f"Fehler bei Style-Analyse: {e}")
            return self._empty_result()

    def _count_unique_colors(self, frame: np.ndarray) -> int:
        """Zaehlt einzigartige Farben (quantisiert)."""
        try:
            # Frame verkleinern
            small = cv2.resize(frame, (100, 100))

            # Quantisieren auf 8 Stufen pro Kanal
            quantized = (small // 32) * 32

            # Einzigartige Farben zaehlen
            pixels = quantized.reshape(-1, 3)
            unique = np.unique(pixels, axis=0)

            return len(unique)

        except Exception:
            return 0

    def _compute_noise_level(self, frame: np.ndarray) -> float:
        """
        Berechnet Noise/Grain Level über das gesamte Bild.

        Performance-Vergleich:
        - ALT (100x100 Sample): ~0.5ms @ 1080p, ungenau bei nicht-uniformem Noise
        - NEU (Downscale 320x180): ~1.2ms @ 1080p, global repräsentativ
        - MULTI-REGION (5 Samples): ~2.5ms @ 1080p, präzise aber langsamer

        Methode: Downscale für Balance zwischen Präzision und Performance.
        """
        try:
            # Konvertiere zu Grayscale falls nötig
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            # Downscale für Performance bei globaler Analyse
            h, w = gray.shape[:2]
            target_size = (320, 180)  # Reduzierte Auflösung (~58k Pixel statt 2M)
            if h > target_size[1] or w > target_size[0]:
                gray = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)

            # Laplacian für Noise-Detection (Hochpass-Filter)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            noise_level = laplacian.std()

            # Normalisiere zu [0, 1] Range
            # Typische Werte: 0-50 für sauber, 50-150 für verrauscht
            normalized = min(1.0, noise_level / 100.0)

            return float(normalized)

        except Exception:
            return 0.0

    def _compute_noise_level_multiregion(self, frame: np.ndarray) -> float:
        """
        Multi-Region Noise Sampling (Alternative für höchste Präzision).

        Analysiert 5 strategische Regionen:
        - 4 Ecken (typisch dunkler, mehr Sensor-Noise)
        - 1 Center (meist besser ausgeleuchtet)

        Performance: ~2.5ms @ 1080p (ca. 2x langsamer als Downscale-Methode)
        Präzision: Erkennt regionale Noise-Variationen (z.B. ISO-Noise in Schatten)

        Verwendung: Nur bei Bedarf für hochpräzise Style-Klassifikation.
        """
        try:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            h, w = gray.shape[:2]

            # Definiere 5 strategische Regionen
            regions = [
                (0, 0, w // 3, h // 3),  # Top-Left
                (w // 3, 0, 2 * w // 3, h // 3),  # Top-Center
                (2 * w // 3, 0, w, h // 3),  # Top-Right
                (w // 3, h // 3, 2 * w // 3, 2 * h // 3),  # Center (wichtigste Region)
                (w // 3, 2 * h // 3, 2 * w // 3, h),  # Bottom-Center
            ]

            noise_levels = []
            for x1, y1, x2, y2 in regions:
                region = gray[y1:y2, x1:x2]
                if region.size > 0:
                    laplacian = cv2.Laplacian(region, cv2.CV_64F)
                    noise_levels.append(laplacian.std())

            if not noise_levels:
                return 0.0

            # Durchschnitt der Regionen (mit leichter Center-Gewichtung möglich)
            avg_noise = float(np.mean(noise_levels))

            # Normalisiere zu [0, 1]
            return min(1.0, avg_noise / 100.0)

        except Exception:
            return 0.0

    def _compute_sharpness(self, frame: np.ndarray) -> float:
        """Berechnet Bildschaerfe via Laplacian Variance."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            return laplacian.var()

        except Exception:
            return 0.0

    def _compute_vignette(self, frame: np.ndarray) -> float:
        """Berechnet Vignettierungs-Score."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # Mitte-Region (innere 50%)
            margin_x = w // 4
            margin_y = h // 4
            center = gray[margin_y : h - margin_y, margin_x : w - margin_x]

            # Ecken-Regionen
            corner_size = min(h, w) // 6
            corners = [
                gray[0:corner_size, 0:corner_size],  # oben links
                gray[0:corner_size, w - corner_size : w],  # oben rechts
                gray[h - corner_size : h, 0:corner_size],  # unten links
                gray[h - corner_size : h, w - corner_size : w],  # unten rechts
            ]

            center_brightness = np.mean(center)
            corner_brightness = np.mean([np.mean(c) for c in corners])

            # Vignette = Mitte heller als Ecken
            vignette = (center_brightness - corner_brightness) / 255.0

            return max(0, vignette)

        except Exception:
            return 0.0

    def _compute_saturation_stats(self, frame: np.ndarray) -> tuple:
        """Berechnet Saettigungs-Statistiken."""
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1] / 255.0

            return float(np.mean(saturation)), float(np.std(saturation))

        except Exception:
            return 0.5, 0.0

    def _compute_dynamic_range(self, frame: np.ndarray) -> float:
        """Berechnet Dynamic Range (Histogramm-Spread)."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Histogramm
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten()

            # Nicht-leere Bins finden
            non_zero = np.where(hist > 0)[0]

            if len(non_zero) < 2:
                return 0.0

            # Spread als Dynamic Range
            range_spread = (non_zero[-1] - non_zero[0]) / 255.0

            return range_spread

        except Exception:
            return 0.5

    def _compute_mean_brightness(self, frame: np.ndarray) -> float:
        """Berechnet durchschnittliche Helligkeit."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return float(np.mean(gray))

        except Exception:
            return 128.0

    def _classify_styles(
        self,
        unique_colors: int,
        noise_level: float,
        sharpness: float,
        vignette_score: float,
        sat_mean: float,
        sat_std: float,
        dynamic_range: float,
        mean_brightness: float,
    ) -> list[str]:
        """
        Score-basierte Style-Klassifikation mit Top-2 Selection.

        Behebt überlappende Thresholds durch:
        - Score-Berechnung für alle Styles (0-1)
        - Top-2 Selection (Primary + Secondary)
        - Mutual Exclusion Rules

        Returns:
            Liste mit Primary + optional Secondary Style
        """
        scores = {}

        # Berechne normalisierte Scores für alle Styles (0-1)

        # VINTAGE: Reduzierte Farben + Vignette
        if unique_colors < 800 and vignette_score > 0.05:
            color_factor = (800 - unique_colors) / 800
            vignette_factor = min(vignette_score / 0.15, 1.0)
            scores[VisualStyle.VINTAGE] = (color_factor + vignette_factor) / 2
        else:
            scores[VisualStyle.VINTAGE] = 0

        # FILMIC: Grain + moderate Saettigung
        if noise_level > 0.3 and sat_mean < 0.6:
            grain_factor = min((noise_level - 0.3) / 0.3, 1.0)
            sat_factor = (0.6 - sat_mean) / 0.6
            scores[VisualStyle.FILMIC] = (grain_factor + sat_factor) / 2
        else:
            scores[VisualStyle.FILMIC] = 0

        # NEON: Hohe Saettigung + hohe Schaerfe
        if sat_mean > 0.5 and sharpness > 800:
            sat_factor = (sat_mean - 0.5) / 0.5
            sharp_factor = min((sharpness - 800) / 1200, 1.0)
            scores[VisualStyle.NEON] = (sat_factor + sharp_factor) / 2
        else:
            scores[VisualStyle.NEON] = 0

        # MINIMALIST: Wenig Farben + hohe Dynamic Range
        if unique_colors < 500 and dynamic_range > 0.6:
            color_factor = (500 - unique_colors) / 500
            range_factor = (dynamic_range - 0.6) / 0.4
            scores[VisualStyle.MINIMALIST] = (color_factor + range_factor) / 2
        else:
            scores[VisualStyle.MINIMALIST] = 0

        # PSYCHEDELIC: Viele Farben + hohe Saettigung
        if unique_colors > 2000 and sat_mean > 0.4:
            color_factor = min((unique_colors - 2000) / 1000, 1.0)
            sat_factor = (sat_mean - 0.4) / 0.6
            scores[VisualStyle.PSYCHEDELIC] = (color_factor + sat_factor) / 2
        else:
            scores[VisualStyle.PSYCHEDELIC] = 0

        # CINEMATIC: Vignette + variable Saettigung
        if vignette_score > 0.03 and 0.3 < sat_mean < 0.7:
            vignette_factor = min(vignette_score / 0.1, 1.0)
            sat_factor = 1.0 - abs(sat_mean - 0.5) / 0.2
            scores[VisualStyle.CINEMATIC] = (vignette_factor + sat_factor) / 2
        else:
            scores[VisualStyle.CINEMATIC] = 0

        # DIGITAL: Hohe Schaerfe + wenig Noise
        if sharpness > 1000 and noise_level < 0.15:
            sharp_factor = min((sharpness - 1000) / 1000, 1.0)
            clean_factor = (0.15 - noise_level) / 0.15
            scores[VisualStyle.DIGITAL] = (sharp_factor + clean_factor) / 2
        else:
            scores[VisualStyle.DIGITAL] = 0

        # DREAMY: Niedrige Schaerfe + niedrige Saettigung
        if sharpness < 300 and sat_mean < 0.4:
            soft_factor = (300 - sharpness) / 300
            desat_factor = (0.4 - sat_mean) / 0.4
            scores[VisualStyle.DREAMY] = (soft_factor + desat_factor) / 2
        else:
            scores[VisualStyle.DREAMY] = 0

        # HIGH_CONTRAST: Hohe Dynamic Range
        scores[VisualStyle.HIGH_CONTRAST] = (
            max(0, (dynamic_range - 0.8) / 0.2) if dynamic_range > 0.8 else 0
        )

        # LOW_KEY: Dunkel
        scores[VisualStyle.LOW_KEY] = (80 - mean_brightness) / 80 if mean_brightness < 80 else 0

        # HIGH_KEY: Hell
        scores[VisualStyle.HIGH_KEY] = (mean_brightness - 180) / 75 if mean_brightness > 180 else 0

        # STANDARD: Baseline für alle Fälle
        scores[VisualStyle.STANDARD] = 0.3

        # Mutual Exclusion Rules
        mutual_exclusive_pairs = [
            (VisualStyle.LOW_KEY, VisualStyle.HIGH_KEY),
            (VisualStyle.MINIMALIST, VisualStyle.PSYCHEDELIC),
            (VisualStyle.DIGITAL, VisualStyle.VINTAGE),
            (VisualStyle.DIGITAL, VisualStyle.DREAMY),
        ]

        for style1, style2 in mutual_exclusive_pairs:
            if style1 in scores and style2 in scores:
                if scores[style1] > 0 and scores[style2] > 0:
                    if scores[style1] < scores[style2]:
                        scores[style1] = 0
                    else:
                        scores[style2] = 0

        # Top-2 Selection
        sorted_styles = sorted(
            [(style, score) for style, score in scores.items() if score > 0.3],
            key=lambda x: x[1],
            reverse=True,
        )

        styles = []
        if len(sorted_styles) > 0:
            styles.append(sorted_styles[0][0])  # Primary
        if len(sorted_styles) > 1 and sorted_styles[1][1] > 0.4:
            # Secondary nur wenn deutlich über Threshold
            styles.append(sorted_styles[1][0])

        # Fallback
        if not styles:
            styles.append(VisualStyle.STANDARD)

        return styles

    def _empty_result(self) -> StyleAnalysisResult:
        """Gibt leeres Ergebnis zurueck."""
        return StyleAnalysisResult(
            styles=[VisualStyle.STANDARD],
            unique_colors=0,
            noise_level=0.0,
            sharpness=0.0,
            vignette_score=0.0,
            saturation_mean=0.5,
            saturation_std=0.0,
            dynamic_range=0.5,
            mean_brightness=128.0,
        )

    def analyze_video(
        self, video_path: str, positions: list[str] = None
    ) -> dict[str, StyleAnalysisResult]:
        """
        Analysiert mehrere Frames aus einem Video.

        Args:
            video_path: Pfad zum Video
            positions: Frame-Positionen ['start', 'middle', 'end']

        Returns:
            Dict {position: StyleAnalysisResult}
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
            logger.error(f"Fehler bei Video-Style-Analyse: {e}")
            results = {pos: self._empty_result() for pos in positions}

        return results

    def get_style_profile(self, frame: np.ndarray) -> dict:
        """
        Erstellt detailliertes Style-Profil.

        Returns:
            Dict mit Style-Dimensionen
        """
        result = self.analyze(frame)

        profile = {
            "primary_styles": result.styles[:3],
            "color_complexity": "LOW"
            if result.unique_colors < 500
            else "MEDIUM"
            if result.unique_colors < 1500
            else "HIGH",
            "grain_level": "NONE"
            if result.noise_level < 0.1
            else "LIGHT"
            if result.noise_level < 0.3
            else "HEAVY",
            "focus_quality": "SOFT"
            if result.sharpness < 300
            else "NORMAL"
            if result.sharpness < 1000
            else "SHARP",
            "lighting": "LOW_KEY"
            if result.mean_brightness < 80
            else "HIGH_KEY"
            if result.mean_brightness > 180
            else "BALANCED",
            "style_vector": [
                result.unique_colors / 3000,
                result.noise_level,
                result.sharpness / 2000,
                result.vignette_score,
                result.saturation_mean,
                result.dynamic_range,
            ],
        }

        return profile
