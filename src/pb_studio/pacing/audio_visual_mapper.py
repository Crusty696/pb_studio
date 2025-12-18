"""
PB_studio - Audio-Visual Mapper
================================

Phase 2: Music-Video Mapping Implementation

Maps audio features to visual properties for intelligent clip selection:
1. Audio → Emotion Space mapping
2. Audio → Color Palette generation
3. Color similarity scoring using CIEDE2000

Dependencies:
- AudioAnalyzer spectral features (brightness, richness, percussiveness)
- ColorAnalyzer with LAB color space
- MoodType and MotionType enums from advanced_pacing_engine

Author: PB_studio
Created: 2025-12-05
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from skimage.color import deltaE_ciede2000, rgb2lab

    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

from pb_studio.pacing.advanced_pacing_engine import MoodType, MotionType

logger = logging.getLogger(__name__)


# =============================================================================
# Emotion Space Model
# =============================================================================


@dataclass
class EmotionCoordinates:
    """
    2D Emotion Space based on Russell's Circumplex Model.

    Axes:
    - valence: -1.0 (negative/sad) to 1.0 (positive/happy)
    - arousal: -1.0 (calm/relaxed) to 1.0 (energetic/excited)
    """

    valence: float  # -1.0 to 1.0
    arousal: float  # -1.0 to 1.0

    def __post_init__(self):
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(-1.0, min(1.0, self.arousal))

    def distance_to(self, other: EmotionCoordinates) -> float:
        """Euclidean distance in emotion space."""
        return np.sqrt((self.valence - other.valence) ** 2 + (self.arousal - other.arousal) ** 2)

    def to_mood_type(self) -> MoodType:
        """Map emotion coordinates to nearest MoodType."""
        # Quadrant-based mapping
        if self.arousal >= 0:
            if self.valence >= 0:
                # High arousal, positive valence: energetic/happy
                if self.arousal > 0.7:
                    return MoodType.EUPHORIC
                elif self.arousal > 0.4:
                    return MoodType.ENERGETIC
                else:
                    return MoodType.CHEERFUL
            else:
                # High arousal, negative valence: tense/aggressive
                if self.arousal > 0.7:
                    return MoodType.AGGRESSIVE
                else:
                    return MoodType.TENSE
        else:
            if self.valence >= 0:
                # Low arousal, positive valence: calm/peaceful
                if self.valence > 0.5:
                    return MoodType.PEACEFUL
                else:
                    return MoodType.CALM
            else:
                # Low arousal, negative valence: sad/melancholic
                if self.arousal < -0.5:
                    return MoodType.MELANCHOLIC
                else:
                    return MoodType.MYSTERIOUS


# Emotion coordinates for each MoodType
MOOD_EMOTION_MAP: dict[MoodType, EmotionCoordinates] = {
    MoodType.CALM: EmotionCoordinates(valence=0.2, arousal=-0.6),
    MoodType.PEACEFUL: EmotionCoordinates(valence=0.5, arousal=-0.5),
    MoodType.CHEERFUL: EmotionCoordinates(valence=0.7, arousal=0.3),
    MoodType.ENERGETIC: EmotionCoordinates(valence=0.5, arousal=0.7),
    MoodType.MELANCHOLIC: EmotionCoordinates(valence=-0.6, arousal=-0.4),
    MoodType.EUPHORIC: EmotionCoordinates(valence=0.8, arousal=0.9),
    MoodType.AGGRESSIVE: EmotionCoordinates(valence=-0.5, arousal=0.8),
    MoodType.MYSTERIOUS: EmotionCoordinates(valence=-0.2, arousal=-0.3),
    MoodType.DREAMY: EmotionCoordinates(valence=0.3, arousal=-0.4),
    MoodType.COOL: EmotionCoordinates(valence=0.1, arousal=0.2),
    MoodType.WARM: EmotionCoordinates(valence=0.6, arousal=0.1),
    MoodType.NOSTALGIC: EmotionCoordinates(valence=0.2, arousal=-0.3),
    MoodType.ROMANTIC: EmotionCoordinates(valence=0.7, arousal=0.0),
    MoodType.TENSE: EmotionCoordinates(valence=-0.4, arousal=0.5),
    MoodType.PLAYFUL: EmotionCoordinates(valence=0.6, arousal=0.5),
}


# =============================================================================
# Audio-to-Color Mapping
# =============================================================================


@dataclass
class AudioDerivedPalette:
    """Color palette derived from audio features."""

    primary_color_lab: tuple[float, float, float]  # LAB color
    secondary_colors_lab: list[tuple[float, float, float]]
    temperature: str  # WARM, COOL, NEUTRAL
    brightness: str  # DARK, MEDIUM, BRIGHT
    saturation_level: float  # 0-1
    mood: MoodType
    emotion: EmotionCoordinates

    @property
    def primary_color_rgb(self) -> tuple[int, int, int]:
        """Convert primary LAB color to RGB."""
        return _lab_to_rgb(self.primary_color_lab)


# Color palette presets based on mood
MOOD_COLOR_PALETTES: dict[MoodType, dict[str, Any]] = {
    MoodType.CALM: {
        "primary_lab": (70, -5, -10),  # Soft blue-gray
        "temperature": "COOL",
        "brightness": "MEDIUM",
        "saturation": 0.3,
    },
    MoodType.PEACEFUL: {
        "primary_lab": (80, -10, 10),  # Soft green
        "temperature": "NEUTRAL",
        "brightness": "BRIGHT",
        "saturation": 0.4,
    },
    MoodType.CHEERFUL: {
        "primary_lab": (85, 10, 50),  # Warm yellow
        "temperature": "WARM",
        "brightness": "BRIGHT",
        "saturation": 0.7,
    },
    MoodType.ENERGETIC: {
        "primary_lab": (60, 50, 30),  # Vibrant orange
        "temperature": "WARM",
        "brightness": "MEDIUM",
        "saturation": 0.8,
    },
    MoodType.MELANCHOLIC: {
        "primary_lab": (40, 0, -20),  # Deep blue
        "temperature": "COOL",
        "brightness": "DARK",
        "saturation": 0.4,
    },
    MoodType.EUPHORIC: {
        "primary_lab": (70, 40, -30),  # Magenta/Pink
        "temperature": "WARM",
        "brightness": "BRIGHT",
        "saturation": 0.9,
    },
    MoodType.AGGRESSIVE: {
        "primary_lab": (45, 60, 40),  # Intense red
        "temperature": "WARM",
        "brightness": "DARK",
        "saturation": 0.9,
    },
    MoodType.MYSTERIOUS: {
        "primary_lab": (30, 10, -30),  # Deep purple
        "temperature": "COOL",
        "brightness": "DARK",
        "saturation": 0.5,
    },
    MoodType.DREAMY: {
        "primary_lab": (75, 20, -20),  # Lavender
        "temperature": "COOL",
        "brightness": "BRIGHT",
        "saturation": 0.5,
    },
    MoodType.COOL: {
        "primary_lab": (55, -10, -25),  # Cyan-blue
        "temperature": "COOL",
        "brightness": "MEDIUM",
        "saturation": 0.6,
    },
    MoodType.WARM: {
        "primary_lab": (65, 20, 40),  # Warm amber
        "temperature": "WARM",
        "brightness": "MEDIUM",
        "saturation": 0.6,
    },
    MoodType.NOSTALGIC: {
        "primary_lab": (55, 10, 20),  # Sepia-like
        "temperature": "WARM",
        "brightness": "MEDIUM",
        "saturation": 0.4,
    },
    MoodType.ROMANTIC: {
        "primary_lab": (65, 35, -10),  # Rose/Pink
        "temperature": "WARM",
        "brightness": "MEDIUM",
        "saturation": 0.6,
    },
    MoodType.TENSE: {
        "primary_lab": (35, 20, -10),  # Dark maroon
        "temperature": "NEUTRAL",
        "brightness": "DARK",
        "saturation": 0.5,
    },
    MoodType.PLAYFUL: {
        "primary_lab": (75, 30, 60),  # Bright orange-yellow
        "temperature": "WARM",
        "brightness": "BRIGHT",
        "saturation": 0.8,
    },
}


def _lab_to_rgb(lab: tuple[float, float, float]) -> tuple[int, int, int]:
    """Convert LAB color to RGB (approximate conversion)."""
    L, a, b = lab

    # LAB to XYZ
    y = (L + 16) / 116
    x = a / 500 + y
    z = y - b / 200

    def xyz_helper(t):
        if t > 0.2068966:
            return t**3
        else:
            return (t - 16 / 116) / 7.787

    x = xyz_helper(x) * 0.95047
    y = xyz_helper(y) * 1.00000
    z = xyz_helper(z) * 1.08883

    # XYZ to RGB
    r = x * 3.2406 + y * -1.5372 + z * -0.4986
    g = x * -0.9689 + y * 1.8758 + z * 0.0415
    b = x * 0.0557 + y * -0.2040 + z * 1.0570

    def gamma_correct(c):
        if c > 0.0031308:
            return 1.055 * (c ** (1 / 2.4)) - 0.055
        else:
            return 12.92 * c

    r = int(max(0, min(255, gamma_correct(r) * 255)))
    g = int(max(0, min(255, gamma_correct(g) * 255)))
    b = int(max(0, min(255, gamma_correct(b) * 255)))

    return (r, g, b)


# =============================================================================
# Audio-Visual Mapper Class
# =============================================================================


class AudioVisualMapper:
    """
    Maps audio features to visual properties for clip selection.

    Uses spectral features from AudioAnalyzer to derive:
    - Emotion coordinates (valence/arousal)
    - Color palettes
    - Motion intensity
    """

    def __init__(self):
        self._cache: dict[str, Any] = {}
        logger.info("AudioVisualMapper initialized")

    def map_audio_to_emotion_space(self, spectral_features: dict[str, Any]) -> EmotionCoordinates:
        """
        Map audio spectral features to emotion coordinates.

        Args:
            spectral_features: Dict with keys:
                - brightness: float (0-1, from spectral_centroid)
                - richness: float (0-1, from spectral_bandwidth)
                - percussiveness: float (0-1, from ZCR)
                - rms_energy: float (optional, energy level)

        Returns:
            EmotionCoordinates with valence and arousal
        """
        brightness = spectral_features.get("brightness", 0.5)
        richness = spectral_features.get("richness", 0.5)
        percussiveness = spectral_features.get("percussiveness", 0.5)
        energy = spectral_features.get("rms_energy", 0.5)

        # Map to valence (-1 to 1)
        # Higher brightness + richness → more positive valence
        valence = (brightness * 0.4 + richness * 0.3 - 0.35) * 2

        # Adjust valence based on mode (major/minor if available)
        if "mode" in spectral_features:
            mode = spectral_features["mode"]
            if mode == "major":
                valence = min(1.0, valence + 0.2)
            elif mode == "minor":
                valence = max(-1.0, valence - 0.2)

        # Map to arousal (-1 to 1)
        # Higher energy + percussiveness → higher arousal
        if isinstance(energy, np.ndarray):
            energy = float(np.mean(energy))
        arousal = (energy * 0.5 + percussiveness * 0.5 - 0.25) * 2

        return EmotionCoordinates(valence=valence, arousal=arousal)

    def generate_color_palette_from_audio(
        self, spectral_features: dict[str, Any], segment_type: str = "unknown"
    ) -> AudioDerivedPalette:
        """
        Generate a color palette from audio features.

        Args:
            spectral_features: Dict with spectral features from AudioAnalyzer
            segment_type: Music segment type (intro, verse, chorus, drop, etc.)

        Returns:
            AudioDerivedPalette with derived colors and properties
        """
        # Get emotion coordinates
        emotion = self.map_audio_to_emotion_space(spectral_features)
        mood = emotion.to_mood_type()

        # Override mood based on segment type if available
        segment_moods = self._get_segment_moods(segment_type)
        if segment_moods:
            # Pick the mood that's closest to our emotion coordinates
            best_mood = mood
            best_distance = float("inf")
            for seg_mood_str in segment_moods:
                try:
                    seg_mood = MoodType(seg_mood_str)
                    if seg_mood in MOOD_EMOTION_MAP:
                        dist = emotion.distance_to(MOOD_EMOTION_MAP[seg_mood])
                        if dist < best_distance:
                            best_distance = dist
                            best_mood = seg_mood
                except (ValueError, KeyError):
                    continue
            mood = best_mood

        # Get base palette for this mood
        palette_base = MOOD_COLOR_PALETTES.get(mood, MOOD_COLOR_PALETTES[MoodType.CALM])

        # Modulate palette based on audio features
        brightness = spectral_features.get("brightness", 0.5)
        richness = spectral_features.get("richness", 0.5)

        primary_lab = list(palette_base["primary_lab"])

        # Adjust L (lightness) based on audio brightness
        primary_lab[0] = primary_lab[0] * (0.7 + brightness * 0.6)
        primary_lab[0] = max(10, min(95, primary_lab[0]))

        # Adjust saturation based on richness
        saturation = palette_base["saturation"] * (0.5 + richness)
        saturation = max(0.1, min(1.0, saturation))

        # Generate secondary colors (complementary and analogous)
        secondary_colors = self._generate_secondary_colors(tuple(primary_lab))

        return AudioDerivedPalette(
            primary_color_lab=tuple(primary_lab),
            secondary_colors_lab=secondary_colors,
            temperature=palette_base["temperature"],
            brightness=palette_base["brightness"],
            saturation_level=saturation,
            mood=mood,
            emotion=emotion,
        )

    def score_color_palette_similarity(
        self,
        audio_palette: AudioDerivedPalette,
        clip_colors: list[tuple[int, int, int]],
        weights: dict[str, float] | None = None,
    ) -> float:
        """
        Score how well clip colors match the audio-derived palette.

        Uses CIEDE2000 perceptual color distance for accurate matching.

        Args:
            audio_palette: Palette derived from audio
            clip_colors: List of RGB colors from video clip
            weights: Optional weights for different color aspects

        Returns:
            Similarity score 0-1 (higher = better match)
        """
        if not clip_colors:
            return 0.0

        if not SKIMAGE_AVAILABLE:
            logger.warning("scikit-image not available, using fallback color matching")
            return self._fallback_color_similarity(audio_palette, clip_colors)

        weights = weights or {
            "primary_match": 0.5,
            "temperature_match": 0.25,
            "brightness_match": 0.25,
        }

        score = 0.0

        # Convert clip colors to LAB
        clip_colors_lab = []
        for rgb in clip_colors:
            rgb_arr = np.array([[rgb]], dtype=np.float64) / 255.0
            lab = rgb2lab(rgb_arr)[0, 0]
            clip_colors_lab.append(lab)

        # 1. Primary color match using CIEDE2000
        audio_primary_lab = np.array([audio_palette.primary_color_lab])

        min_delta_e = float("inf")
        for clip_lab in clip_colors_lab:
            delta_e = deltaE_ciede2000(audio_primary_lab, np.array([clip_lab]))[0]
            min_delta_e = min(min_delta_e, delta_e)

        # Convert deltaE to similarity (deltaE of 0 = perfect match, 100 = very different)
        primary_similarity = max(0, 1 - (min_delta_e / 50))
        score += primary_similarity * weights["primary_match"]

        # 2. Temperature match
        clip_temp = self._detect_temperature(clip_colors)
        if clip_temp == audio_palette.temperature:
            score += weights["temperature_match"]
        elif clip_temp == "NEUTRAL" or audio_palette.temperature == "NEUTRAL":
            score += weights["temperature_match"] * 0.5

        # 3. Brightness match
        clip_brightness = self._detect_brightness(clip_colors)
        if clip_brightness == audio_palette.brightness:
            score += weights["brightness_match"]
        elif (
            abs(
                ["DARK", "MEDIUM", "BRIGHT"].index(clip_brightness)
                - ["DARK", "MEDIUM", "BRIGHT"].index(audio_palette.brightness)
            )
            == 1
        ):
            score += weights["brightness_match"] * 0.5

        return min(1.0, score)

    def map_energy_to_motion(self, energy_level: float) -> MotionType:
        """
        Map audio energy level to appropriate motion type.

        Args:
            energy_level: Energy value 0-1

        Returns:
            MotionType matching the energy level
        """
        if energy_level < 0.2:
            return MotionType.STATIC
        elif energy_level < 0.4:
            return MotionType.SLOW
        elif energy_level < 0.6:
            return MotionType.MEDIUM
        elif energy_level < 0.8:
            return MotionType.FAST
        else:
            return MotionType.EXTREME

    def _get_segment_moods(self, segment_type: str) -> list[str]:
        """Get appropriate moods for a music segment type."""
        segment_mapping = {
            "intro": ["CALM", "PEACEFUL", "DREAMY", "MYSTERIOUS"],
            "verse": ["CALM", "PEACEFUL", "MELANCHOLIC", "COOL"],
            "chorus": ["ENERGETIC", "CHEERFUL", "WARM", "EUPHORIC"],
            "drop": ["ENERGETIC", "EUPHORIC", "AGGRESSIVE", "TENSE"],
            "bridge": ["DREAMY", "MYSTERIOUS", "MELANCHOLIC", "COOL"],
            "outro": ["CALM", "PEACEFUL", "DREAMY", "WARM"],
            "buildup": ["TENSE", "ENERGETIC", "MYSTERIOUS"],
            "breakdown": ["CALM", "DREAMY", "MELANCHOLIC"],
        }
        return segment_mapping.get(segment_type.lower(), [])

    def _generate_secondary_colors(
        self, primary_lab: tuple[float, float, float]
    ) -> list[tuple[float, float, float]]:
        """Generate complementary and analogous colors in LAB space."""
        L, a, b = primary_lab

        # Complementary (opposite on a-b plane)
        complementary = (L, -a * 0.8, -b * 0.8)

        # Analogous (rotated on a-b plane)
        angle1 = np.radians(30)
        angle2 = np.radians(-30)

        r = np.sqrt(a**2 + b**2)
        theta = np.arctan2(b, a)

        analogous1 = (L, r * np.cos(theta + angle1), r * np.sin(theta + angle1))
        analogous2 = (L, r * np.cos(theta + angle2), r * np.sin(theta + angle2))

        return [complementary, analogous1, analogous2]

    def _detect_temperature(self, colors: list[tuple[int, int, int]]) -> str:
        """Detect overall temperature from RGB colors."""
        warm_score = 0
        cool_score = 0

        for r, g, b in colors:
            if r > b:
                warm_score += r - b
            else:
                cool_score += b - r

        diff = warm_score - cool_score
        if diff > 50:
            return "WARM"
        elif diff < -50:
            return "COOL"
        else:
            return "NEUTRAL"

    def _detect_brightness(self, colors: list[tuple[int, int, int]]) -> str:
        """Detect overall brightness from RGB colors."""
        avg_brightness = np.mean([sum(c) / 3 for c in colors])

        if avg_brightness < 85:
            return "DARK"
        elif avg_brightness < 170:
            return "MEDIUM"
        else:
            return "BRIGHT"

    def _fallback_color_similarity(
        self, audio_palette: AudioDerivedPalette, clip_colors: list[tuple[int, int, int]]
    ) -> float:
        """Fallback color matching using simple RGB distance."""
        audio_rgb = audio_palette.primary_color_rgb

        min_distance = float("inf")
        for clip_rgb in clip_colors:
            dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(audio_rgb, clip_rgb)))
            min_distance = min(min_distance, dist)

        # Max RGB distance is ~441 (sqrt(255^2 * 3))
        similarity = max(0, 1 - (min_distance / 441))
        return similarity


# =============================================================================
# Convenience Functions
# =============================================================================


def create_audio_visual_mapper() -> AudioVisualMapper:
    """Create and return an AudioVisualMapper instance."""
    return AudioVisualMapper()


def map_spectral_to_mood(spectral_features: dict[str, Any]) -> MoodType:
    """Quick helper to map spectral features directly to MoodType."""
    mapper = AudioVisualMapper()
    emotion = mapper.map_audio_to_emotion_space(spectral_features)
    return emotion.to_mood_type()
