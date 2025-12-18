"""
Mood Analyzer - Stimmungs-Analyse fuer Video-Clips.

Analysiert emotionale Stimmung basierend auf:
- Helligkeit und Kontrast
- Farbsaettigung
- Farbtemperatur (warm/kalt)
- Energie-Level
"""

from dataclasses import dataclass

import cv2
import numpy as np

from ...utils.logger import get_logger

logger = get_logger()


class Mood:
    """Definiert moegliche Stimmungen."""

    ENERGETIC = "ENERGETIC"
    CALM = "CALM"
    DARK = "DARK"
    BRIGHT = "BRIGHT"
    MELANCHOLIC = "MELANCHOLIC"
    EUPHORIC = "EUPHORIC"
    AGGRESSIVE = "AGGRESSIVE"
    PEACEFUL = "PEACEFUL"
    MYSTERIOUS = "MYSTERIOUS"
    CHEERFUL = "CHEERFUL"
    TENSE = "TENSE"
    DREAMY = "DREAMY"
    COOL = "COOL"
    WARM = "WARM"


@dataclass
class MoodAnalysisResult:
    """Ergebnis der Stimmungs-Analyse."""

    moods: list[str]  # Multi-Label Stimmungen
    mood_scores: dict[str, float]  # Score pro Stimmung
    brightness: float  # 0-255
    saturation: float  # 0-1
    contrast: float  # 0-1
    energy: float  # 0-1 (abgeleitet aus mehreren Faktoren)
    warm_ratio: float  # Anteil warmer Farben
    cool_ratio: float  # Anteil kalter Farben

    def to_dict(self) -> dict:
        """Konvertiert zu Dictionary fuer DB-Speicherung."""
        return {
            "moods": self.moods,
            "mood_scores": self.mood_scores,
            "brightness": self.brightness,
            "saturation": self.saturation,
            "contrast": self.contrast,
            "energy": self.energy,
            "warm_ratio": self.warm_ratio,
            "cool_ratio": self.cool_ratio,
        }


class MoodAnalyzer:
    """Analysiert emotionale Stimmung von Video-Frames."""

    def __init__(self):
        """Initialisiert den MoodAnalyzer."""
        # Schwellwerte fuer Klassifikation
        self.brightness_dark = 80
        self.brightness_bright = 180
        self.saturation_low = 0.3
        self.saturation_high = 0.6
        self.contrast_high = 0.5

    def analyze(self, frame: np.ndarray) -> MoodAnalysisResult:
        """
        Fuehrt vollstaendige Stimmungs-Analyse durch.

        Args:
            frame: OpenCV Frame (BGR)

        Returns:
            MoodAnalysisResult mit allen Analyse-Daten
        """
        if frame is None or frame.size == 0:
            return self._empty_result()

        try:
            # Basismetriken berechnen
            brightness = self._compute_brightness(frame)
            saturation = self._compute_saturation(frame)
            contrast = self._compute_contrast(frame)
            warm_ratio, cool_ratio = self._compute_color_temperature(frame)

            # Energie berechnen
            energy = self._compute_energy(brightness, saturation, contrast)

            # Stimmungen klassifizieren
            moods, mood_scores = self._classify_moods(
                brightness, saturation, contrast, energy, warm_ratio, cool_ratio
            )

            return MoodAnalysisResult(
                moods=moods,
                mood_scores=mood_scores,
                brightness=round(brightness, 1),
                saturation=round(saturation, 3),
                contrast=round(contrast, 3),
                energy=round(energy, 3),
                warm_ratio=round(warm_ratio, 3),
                cool_ratio=round(cool_ratio, 3),
            )

        except Exception as e:
            logger.error(f"Fehler bei Stimmungs-Analyse: {e}")
            return self._empty_result()

    def _compute_brightness(self, frame: np.ndarray) -> float:
        """Berechnet durchschnittliche Helligkeit (0-255)."""
        # HSV fuer Helligkeit
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return float(np.mean(hsv[:, :, 2]))

    def _compute_saturation(self, frame: np.ndarray) -> float:
        """Berechnet durchschnittliche Saettigung (0-1)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return float(np.mean(hsv[:, :, 1])) / 255.0

    def _compute_contrast(self, frame: np.ndarray) -> float:
        """
        Berechnet Kontrast basierend auf Standardabweichung der Helligkeit.

        Returns:
            Normalisierter Kontrast (0-1)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        std = np.std(gray)
        # Normalisieren (typischer Bereich 0-80)
        return min(std / 80.0, 1.0)

    def _compute_color_temperature(self, frame: np.ndarray) -> tuple[float, float]:
        """
        Berechnet Anteil warmer und kalter Farben.

        Returns:
            (warm_ratio, cool_ratio) jeweils 0-1
        """
        # Zu HSV konvertieren
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]  # 0-179 in OpenCV
        sat = hsv[:, :, 1]

        # Nur gesaettigte Pixel betrachten
        mask = sat > 50

        if not np.any(mask):
            return 0.5, 0.5

        hue_masked = hue[mask]

        # Warme Farben: Rot, Orange, Gelb (0-30 und 150-179)
        warm_mask = (hue_masked < 30) | (hue_masked > 150)
        warm_count = np.sum(warm_mask)

        # Kalte Farben: Blau, Gruen-Blau (80-130)
        cool_mask = (hue_masked > 80) & (hue_masked < 130)
        cool_count = np.sum(cool_mask)

        total = len(hue_masked)
        warm_ratio = warm_count / total if total > 0 else 0.0
        cool_ratio = cool_count / total if total > 0 else 0.0

        return warm_ratio, cool_ratio

    def _compute_energy(self, brightness: float, saturation: float, contrast: float) -> float:
        """
        Berechnet Energie-Level (0-1).

        Hohe Energie = hell, gesaettigt, kontrastreich
        Niedrige Energie = dunkel, entsaettigt, kontrastarm
        """
        # Gewichtete Kombination
        brightness_norm = brightness / 255.0
        energy = brightness_norm * 0.3 + saturation * 0.4 + contrast * 0.3
        return min(max(energy, 0.0), 1.0)

    def _classify_moods(
        self,
        brightness: float,
        saturation: float,
        contrast: float,
        energy: float,
        warm_ratio: float,
        cool_ratio: float,
    ) -> tuple[list[str], dict[str, float]]:
        """
        Score-basierte Mood-Klassifikation mit Top-2 Selection.

        Behebt überlappende Thresholds durch:
        - Score-Berechnung für alle Moods
        - Top-2 Selection (Primary + Secondary)
        - Mutual Exclusion Rules für widersprüchliche Moods

        Returns:
            (moods, mood_scores) - Primary + optional Secondary Mood
        """
        scores = {}

        # Berechne Scores für alle Moods (0-1 normalisiert)

        # DARK: Niedrige Helligkeit
        scores[Mood.DARK] = (
            max(0, (self.brightness_dark - brightness) / self.brightness_dark)
            if brightness < self.brightness_dark
            else 0
        )

        # BRIGHT: Hohe Helligkeit
        scores[Mood.BRIGHT] = (
            (brightness - self.brightness_bright) / (255 - self.brightness_bright)
            if brightness > self.brightness_bright
            else 0
        )

        # ENERGETIC: Hohe Energie
        scores[Mood.ENERGETIC] = (energy - 0.6) / 0.4 if energy > 0.6 else 0

        # CALM: Niedrige Energie + mittlere Helligkeit
        if energy < 0.4 and 60 < brightness < 180:
            scores[Mood.CALM] = (0.4 - energy) / 0.4
        else:
            scores[Mood.CALM] = 0

        # MELANCHOLIC: Dunkel + entsaettigt + kontrastreich (traurig/melancholisch)
        # Formel: (1-Helligkeit) * 0.4 + (1-Saettigung) * 0.4 + Kontrast * 0.2
        brightness_norm = brightness / 255.0
        dark_factor = 1.0 - brightness_norm
        desat_factor = 1.0 - saturation
        contrast_factor = contrast
        mel_score = dark_factor * 0.4 + desat_factor * 0.4 + contrast_factor * 0.2
        scores[Mood.MELANCHOLIC] = float(np.clip(mel_score, 0.0, 1.0))

        # EUPHORIC: Hell + gesaettigt
        if brightness > 150 and saturation > 0.5:
            scores[Mood.EUPHORIC] = saturation * (brightness / 255)
        else:
            scores[Mood.EUPHORIC] = 0

        # AGGRESSIVE: Hoher Kontrast + gesaettigt + warm
        if contrast > 0.5 and saturation > 0.4 and warm_ratio > 0.3:
            scores[Mood.AGGRESSIVE] = contrast * saturation
        else:
            scores[Mood.AGGRESSIVE] = 0

        # PEACEFUL: Hell + kontrastarm + ruhig (friedlich/entspannt)
        # Formel: Helligkeit * 0.3 + (1-Kontrast) * 0.4 + (1-Energie) * 0.3
        bright_factor = brightness / 255.0
        soft_factor = 1.0 - contrast
        calm_factor = 1.0 - energy
        peace_score = bright_factor * 0.3 + soft_factor * 0.4 + calm_factor * 0.3
        scores[Mood.PEACEFUL] = float(np.clip(peace_score, 0.0, 1.0))

        # MYSTERIOUS: Dunkel + gesaettigt
        if brightness < 100 and saturation > 0.4:
            scores[Mood.MYSTERIOUS] = saturation * ((100 - brightness) / 100)
        else:
            scores[Mood.MYSTERIOUS] = 0

        # CHEERFUL: Hell + gesaettigt + warm
        if brightness > 140 and saturation > 0.4 and warm_ratio > 0.3:
            scores[Mood.CHEERFUL] = saturation * (brightness / 255) * (warm_ratio * 1.5)
        else:
            scores[Mood.CHEERFUL] = 0

        # TENSE: Hoher Kontrast + dunkel
        if contrast > 0.5 and brightness < 120:
            scores[Mood.TENSE] = contrast * ((120 - brightness) / 120)
        else:
            scores[Mood.TENSE] = 0

        # DREAMY: Niedrige Saettigung + weich
        if saturation < 0.3 and contrast < 0.4:
            scores[Mood.DREAMY] = (0.3 - saturation) / 0.3 * (0.4 - contrast) / 0.4
        else:
            scores[Mood.DREAMY] = 0

        # COOL: Kalte Farben dominant
        if cool_ratio > warm_ratio and cool_ratio > 0.2:
            scores[Mood.COOL] = cool_ratio
        else:
            scores[Mood.COOL] = 0

        # WARM: Warme Farben dominant
        if warm_ratio > cool_ratio and warm_ratio > 0.2:
            scores[Mood.WARM] = warm_ratio
        else:
            scores[Mood.WARM] = 0

        # Mutual Exclusion Rules - Entferne widersprüchliche Kombinationen
        mutual_exclusive_pairs = [
            (Mood.DARK, Mood.BRIGHT),
            (Mood.ENERGETIC, Mood.CALM),
            (Mood.COOL, Mood.WARM),
        ]

        for mood1, mood2 in mutual_exclusive_pairs:
            if mood1 in scores and mood2 in scores:
                if scores[mood1] > 0 and scores[mood2] > 0:
                    # Behalte nur den höheren Score
                    if scores[mood1] < scores[mood2]:
                        scores[mood1] = 0
                    else:
                        scores[mood2] = 0

        # Top-2 Selection
        sorted_moods = sorted(
            [(mood, score) for mood, score in scores.items() if score > 0.3],
            key=lambda x: x[1],
            reverse=True,
        )

        moods = []
        if len(sorted_moods) > 0:
            moods.append(sorted_moods[0][0])  # Primary
        if len(sorted_moods) > 1:
            moods.append(sorted_moods[1][0])  # Secondary

        # Fallback wenn keine Klassifikation
        if not moods:
            if energy > 0.5:
                moods.append(Mood.ENERGETIC)
                scores[Mood.ENERGETIC] = energy
            else:
                moods.append(Mood.CALM)
                scores[Mood.CALM] = 1.0 - energy

        # Scores runden (alle mit Score > 0.2 für Analyse)
        mood_scores = {k: round(v, 2) for k, v in scores.items() if v > 0.2}

        return moods, mood_scores

    def _empty_result(self) -> MoodAnalysisResult:
        """Gibt leeres Ergebnis zurueck."""
        return MoodAnalysisResult(
            moods=[Mood.CALM],
            mood_scores={Mood.CALM: 0.5},
            brightness=128.0,
            saturation=0.5,
            contrast=0.5,
            energy=0.5,
            warm_ratio=0.5,
            cool_ratio=0.5,
        )

    def analyze_video(
        self, video_path: str, positions: list[str] = None
    ) -> dict[str, MoodAnalysisResult]:
        """
        Analysiert mehrere Frames aus einem Video.

        Args:
            video_path: Pfad zum Video
            positions: Frame-Positionen ['start', 'middle', 'end']

        Returns:
            Dict {position: MoodAnalysisResult}
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
            logger.error(f"Fehler bei Video-Stimmungsanalyse: {e}")
            results = {pos: self._empty_result() for pos in positions}

        return results

    def get_mood_profile(self, frame: np.ndarray) -> dict:
        """
        Erstellt detailliertes Stimmungsprofil.

        Returns:
            Dict mit allen Stimmungsdimensionen
        """
        result = self.analyze(frame)

        # Dimensionen als Skalen
        profile = {
            "energy_level": result.energy,
            "darkness": 1.0 - (result.brightness / 255),
            "colorfulness": result.saturation,
            "warmth": result.warm_ratio / max(result.warm_ratio + result.cool_ratio, 0.001),
            "intensity": result.contrast,
            "primary_moods": result.moods[:3],
            "mood_vector": [
                result.energy,
                result.saturation,
                result.contrast,
                result.brightness / 255,
                result.warm_ratio - result.cool_ratio,
            ],
        }

        return profile

    def compare_moods(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Vergleicht Stimmung zweier Frames.

        Returns:
            Aehnlichkeit 0-1 (1 = identisch)
        """
        result1 = self.analyze(frame1)
        result2 = self.analyze(frame2)

        # Metriken vergleichen
        diff_brightness = abs(result1.brightness - result2.brightness) / 255
        diff_saturation = abs(result1.saturation - result2.saturation)
        diff_contrast = abs(result1.contrast - result2.contrast)
        diff_energy = abs(result1.energy - result2.energy)
        diff_warmth = abs(result1.warm_ratio - result2.warm_ratio)

        # Durchschnittliche Abweichung
        avg_diff = (
            diff_brightness + diff_saturation + diff_contrast + diff_energy + diff_warmth
        ) / 5

        return 1.0 - avg_diff
