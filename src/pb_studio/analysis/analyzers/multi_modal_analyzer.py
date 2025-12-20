"""
Multi-Modal Audio-Video Analyzer - Kombiniert Audio + Video Intelligence.

Neue Kernfunktion für Overnight Development:
- Cross-Modal Content Understanding
- Audio-Beat + Video-Motion Sync
- Mood-Detection aus Audio + Video
- Intelligente Content-Matching für Pacing

Integration Features:
- Kombiniert AutoStemProcessor (Stems) + VideoIntelligenceEngine (Scene Recognition)
- Audio-Energy + Video-Energy Correlation
- Multi-Modal Feature Extraction für besseres Pacing
- Content-Aware Clip-Selection basierend auf Audio + Video
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy import stats

from ...audio.audio_analyzer import AudioAnalyzer
from ...audio.auto_stem_processor import get_auto_stem_processor
from ...pacing.motion_analyzer import MotionAnalyzer
from ...utils.logger import get_logger
from .video_intelligence_engine import VideoIntelligenceEngine, VideoSceneRecognition

logger = get_logger(__name__)


@dataclass
class MultiModalAnalysisResult:
    """Ergebnis der Multi-Modal Analyse."""

    # Audio Analysis Results
    audio_features: dict[str, Any]  # BPM, energy, stems info
    stems_available: bool
    stems_info: dict[str, Any]

    # Video Analysis Results
    video_features: dict[str, Any]  # Scene recognition, tags, motion data
    primary_scene: str
    scene_confidence: float

    # Cross-Modal Features
    audio_video_sync_score: float  # 0-1, how well audio matches video energy
    mood_alignment: str  # "high_energy", "calm", "dynamic", "mismatch"
    content_quality_score: float  # Overall content quality for pacing

    # Combined Intelligence
    recommended_pacing: str  # "fast", "medium", "slow", "adaptive"
    cut_suggestions: list[str]  # ["beat_sync", "energy_peaks", "scene_changes"]
    multi_modal_tags: list[str]  # Combined tags from audio + video

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "audio_features": self.audio_features,
            "stems_available": self.stems_available,
            "stems_info": self.stems_info,
            "video_features": self.video_features,
            "primary_scene": self.primary_scene,
            "scene_confidence": self.scene_confidence,
            "audio_video_sync_score": self.audio_video_sync_score,
            "mood_alignment": self.mood_alignment,
            "content_quality_score": self.content_quality_score,
            "recommended_pacing": self.recommended_pacing,
            "cut_suggestions": self.cut_suggestions,
            "multi_modal_tags": self.multi_modal_tags,
        }


class MultiModalAnalyzer:
    """
    Multi-Modal Audio-Video Content Analyzer.

    Kombiniert:
    - Audio Intelligence (AutoStemProcessor + AudioAnalyzer)
    - Video Intelligence (VideoIntelligenceEngine)
    - Cross-Modal Correlation Analysis
    - Content-Aware Pacing Recommendations
    """

    def __init__(self, video_engine_size: str = "base", stem_quality_mode: str = "auto"):
        """
        Initialize Multi-Modal Analyzer.

        Args:
            video_engine_size: Size des CLIP-Models ("base" oder "large")
            stem_quality_mode: Qualitätsmodus für Stem-Separation
        """
        # Video Intelligence - Map to correct CLIP model names
        clip_model_mapping = {
            "base": "openai/clip-vit-base-patch32",
            "large": "openai/clip-vit-large-patch14",  # Correct model for large size
        }
        clip_model = clip_model_mapping.get(video_engine_size, "openai/clip-vit-base-patch32")

        self.video_engine = VideoIntelligenceEngine(
            clip_model=clip_model, confidence_threshold=0.6, enable_traditional_cv=True
        )

        # Motion Intelligence
        self.motion_analyzer = MotionAnalyzer(sample_frames=30, frame_skip=5)

        # Audio Intelligence
        self.audio_analyzer = AudioAnalyzer()
        self.stem_quality_mode = stem_quality_mode

        logger.info(
            f"MultiModalAnalyzer initialized (video_model={video_engine_size}, stem_quality={stem_quality_mode})"
        )

    def analyze_multi_modal_content(
        self, audio_path: str, video_path: str, analyze_stems: bool = True
    ) -> MultiModalAnalysisResult:
        """
        Vollständige Multi-Modal Analyse von Audio + Video Content.

        Args:
            audio_path: Pfad zur Audio-Datei
            video_path: Pfad zur Video-Datei
            analyze_stems: Ob Stems analysiert werden sollen (falls verfügbar)

        Returns:
            MultiModalAnalysisResult mit allen Features
        """
        logger.info(
            f"Starting multi-modal analysis: {Path(audio_path).name} + {Path(video_path).name}"
        )

        # 1. Audio Analysis
        audio_features = self._analyze_audio_features(audio_path, analyze_stems)

        # 2. Video Analysis
        video_features = self._analyze_video_features(video_path)

        # 3. Cross-Modal Correlation
        sync_score = self._calculate_audio_video_sync(audio_features, video_features)

        # 4. Mood Alignment
        mood_alignment = self._determine_mood_alignment(audio_features, video_features)

        # 5. Content Quality Score
        quality_score = self._calculate_content_quality(audio_features, video_features, sync_score)

        # 6. Pacing Recommendations
        pacing_recommendation = self._recommend_pacing(
            audio_features, video_features, mood_alignment
        )

        # 7. Cut Suggestions
        cut_suggestions = self._generate_cut_suggestions(audio_features, video_features, sync_score)

        # 8. Combined Tags
        combined_tags = self._generate_multi_modal_tags(audio_features, video_features)

        result = MultiModalAnalysisResult(
            audio_features=audio_features,
            stems_available=audio_features.get("stems_available", False),
            stems_info=audio_features.get("stems_info", {}),
            video_features=video_features,
            primary_scene=video_features.get("primary_scene", "unknown"),
            scene_confidence=video_features.get("confidence", 0.0),
            audio_video_sync_score=sync_score,
            mood_alignment=mood_alignment,
            content_quality_score=quality_score,
            recommended_pacing=pacing_recommendation,
            cut_suggestions=cut_suggestions,
            multi_modal_tags=combined_tags,
        )

        logger.info(
            f"Multi-modal analysis complete: {mood_alignment} mood, {pacing_recommendation} pacing, quality={quality_score:.2f}"
        )
        return result

    def _analyze_audio_features(self, audio_path: str, analyze_stems: bool) -> dict[str, Any]:
        """Audio Feature Extraction mit Stems."""
        try:
            # Basic Audio Analysis
            audio_features = {
                "bpm": None,
                "energy": None,  # Will hold full spectral energy
                "tempo_stability": 0.0,
                "stems_available": False,
                "stems_info": {},
                "spectral_features": None,  # Detailed spectral features
            }

            # BPM Analysis
            bpm_result = self.audio_analyzer.analyze_bpm(audio_path)
            if bpm_result:
                audio_features["bpm"] = bpm_result.get("bpm", 120.0)
                audio_features["tempo_stability"] = bpm_result.get("tempo_stability", 0.5)

 feature-sophisticated-quality-analysis-2813999412410248020
            # Energy Analysis (via Spectral Features for timeline)
            spectral_features = self.audio_analyzer.extract_spectral_features(audio_path)
            if spectral_features:
                audio_features["energy_timeline"] = {
                    "times": spectral_features.get("frame_times", []),
                    "values": spectral_features.get("rms_energy", []),
                }
                audio_features["energy"] = spectral_features.get("mean_energy", 0.5)

            # Energy Analysis (legacy beatgrid check)
            beatgrid_result = self.audio_analyzer.analyze_beatgrid(audio_path)
            if beatgrid_result:
                # Keep backward compatibility if needed, though spectral analysis is better
                pass

            # Detailed Spectral Analysis (for correlation)
            spectral_result = self.audio_analyzer.extract_spectral_features(audio_path)
            if spectral_result:
                audio_features["spectral_features"] = spectral_result
                # Use rms_energy from spectral features as the main energy indicator
                audio_features["energy"] = spectral_result.get("rms_energy")
 main

            # Stems Analysis (if enabled)
            if analyze_stems:
                stem_processor = get_auto_stem_processor(self.stem_quality_mode)

                # Check for existing stems
                all_exist, existing_stems = stem_processor.stems_exist(audio_path)
                if all_exist:
                    audio_features["stems_available"] = True
                    audio_features["stems_info"] = {
                        "stems": existing_stems,
                        "model_info": stem_processor.get_model_info(),
                    }
                    logger.debug(f"Using existing stems: {list(existing_stems.keys())}")
                else:
                    logger.debug(f"No stems available for {Path(audio_path).name}")

            return audio_features

        except Exception as e:
            logger.error(f"Audio analysis failed for {audio_path}: {e}")
            return {
                "bpm": 120.0,
                "energy": None,
                "tempo_stability": 0.5,
                "stems_available": False,
                "stems_info": {},
                "spectral_features": None,
                "error": str(e),
            }

    def _analyze_video_features(self, video_path: str) -> dict[str, Any]:
        """Video Feature Extraction mit CLIP und Motion."""
        try:
            # Get video tags using VideoIntelligenceEngine
            video_tags = self.video_engine.get_video_tags(video_path)

            # Get motion analysis
            motion_result = self.motion_analyzer.analyze_clip(video_path)

            return {
                "tags": video_tags.get("tags", ["unknown"]),
                "primary_scene": video_tags.get("primary_scene", "unknown"),
                "confidence": video_tags.get("confidence", 0.0),
                "quality_score": video_tags.get("quality_score", 0.0),
                "frame_count": video_tags.get("frame_count", 0),
                "all_scores": video_tags.get("all_scores", {}),
 feature-sophisticated-quality-analysis-2813999412410248020
                "quality_timeline": video_tags.get("quality_timeline", []),

                "motion_score": motion_result.motion_score,
                "motion_series": motion_result.motion_series,
                "motion_times": motion_result.sample_times,
 main
            }

        except Exception as e:
            logger.error(f"Video analysis failed for {video_path}: {e}")
            return {
                "tags": ["unknown"],
                "primary_scene": "unknown",
                "confidence": 0.0,
                "quality_score": 0.0,
                "frame_count": 0,
                "all_scores": {},
                "motion_score": 0.0,
                "motion_series": [],
                "motion_times": [],
                "error": str(e),
            }

    def _calculate_audio_video_sync(self, audio_features: dict, video_features: dict) -> float:
        """Berechnet Audio-Video Synchronisation Score."""
        try:
            sync_score = 0.5  # Default neutral score

            # Factor 1: BPM vs Video Energy
            bpm = audio_features.get("bpm", 120)
            video_tags = video_features.get("tags", [])

            # High BPM should match energetic video content
            energetic_video_tags = ["party", "dancing", "energetic", "dynamic", "action"]
            calm_video_tags = ["peaceful", "calm", "nature", "sunset", "relaxing"]

            has_energetic = any(tag in video_tags for tag in energetic_video_tags)
            has_calm = any(tag in video_tags for tag in calm_video_tags)

            if bpm > 130:  # Fast tempo
                if has_energetic:
                    sync_score += 0.3
                elif has_calm:
                    sync_score -= 0.2
            elif bpm < 100:  # Slow tempo
                if has_calm:
                    sync_score += 0.3
                elif has_energetic:
                    sync_score -= 0.2

            # Factor 2: Video Quality vs Audio Quality
            video_quality = video_features.get("quality_score", 0.5)
            tempo_stability = audio_features.get("tempo_stability", 0.5)

            # Both high quality = better sync potential
            quality_factor = (video_quality + tempo_stability) / 2
            sync_score += (quality_factor - 0.5) * 0.2

            return max(0.0, min(1.0, sync_score))

        except Exception as e:
            logger.error(f"Sync calculation failed: {e}")
            return 0.5

    def _determine_mood_alignment(self, audio_features: dict, video_features: dict) -> str:
        """Bestimmt Mood Alignment zwischen Audio und Video."""
        try:
            bpm = audio_features.get("bpm", 120)
            video_tags = video_features.get("tags", [])

            # Energy Level Classification
            high_energy_audio = bpm > 130
            medium_energy_audio = 100 <= bpm <= 130
            low_energy_audio = bpm < 100

            high_energy_video_tags = ["party", "dancing", "energetic", "dynamic", "action", "crowd"]
            calm_video_tags = ["peaceful", "calm", "nature", "sunset", "beach", "relaxing"]

            has_high_energy_video = any(tag in video_tags for tag in high_energy_video_tags)
            has_calm_video = any(tag in video_tags for tag in calm_video_tags)

            # Determine alignment
            if high_energy_audio and has_high_energy_video:
                return "high_energy"
            elif low_energy_audio and has_calm_video:
                return "calm"
            elif medium_energy_audio:
                return "dynamic"
            else:
                return "mismatch"

        except Exception as e:
            logger.error(f"Mood alignment calculation failed: {e}")
            return "dynamic"

    def _calculate_content_quality(
        self, audio_features: dict, video_features: dict, sync_score: float
    ) -> float:
        """Berechnet Overall Content Quality Score."""
        try:
            scores = []

            # Audio Quality Indicators
            if audio_features.get("bpm"):
                scores.append(0.8)  # Has BPM detection
            tempo_stability = audio_features.get("tempo_stability", 0.5)
            scores.append(tempo_stability)

            # Video Quality Indicators
            video_quality = video_features.get("quality_score", 0.5)
            scene_confidence = video_features.get("confidence", 0.0)
            scores.extend([video_quality, scene_confidence])

            # Sync Quality
            scores.append(sync_score)

            # Stems Availability Bonus
            if audio_features.get("stems_available", False):
                scores.append(0.9)  # Stems available = higher quality potential

            return np.mean(scores) if scores else 0.5

        except Exception as e:
            logger.error(f"Quality score calculation failed: {e}")
            return 0.5

    def _recommend_pacing(
        self, audio_features: dict, video_features: dict, mood_alignment: str
    ) -> str:
        """Empfiehlt Pacing Strategy basierend auf Content."""
        try:
            bpm = audio_features.get("bpm", 120)

            # BPM-based base recommendation
            if bpm > 140:
                base_pacing = "fast"
            elif bpm > 110:
                base_pacing = "medium"
            else:
                base_pacing = "slow"

            # Mood-based adjustment
            if mood_alignment == "high_energy":
                return "fast"
            elif mood_alignment == "calm":
                return "slow"
            elif mood_alignment == "mismatch":
                return "adaptive"  # Let system adjust dynamically
            else:
                return base_pacing

        except Exception as e:
            logger.error(f"Pacing recommendation failed: {e}")
            return "medium"

    def _generate_cut_suggestions(
        self, audio_features: dict, video_features: dict, sync_score: float
    ) -> list[str]:
        """Generiert Cut-Strategy Vorschläge."""
        suggestions = []

        try:
            bpm = audio_features.get("bpm")
            stems_available = audio_features.get("stems_available", False)
            scene_confidence = video_features.get("confidence", 0.0)

            # Beat-based cuts (wenn BPM verfügbar)
            if bpm and bpm > 80:
                suggestions.append("beat_sync")

            # Stem-based cuts (wenn Stems verfügbar)
            if stems_available:
                suggestions.append("stem_triggers")  # Kick/Snare-based cuts

            # Energy-based cuts
            if audio_features.get("energy"):
                suggestions.append("energy_peaks")

            # Scene-based cuts (wenn gute Video-Erkennung)
            if scene_confidence > 0.6:
                suggestions.append("scene_changes")

            # Sync-based adaptive cuts
            if sync_score > 0.7:
                suggestions.append("audio_video_sync")

            return suggestions if suggestions else ["beat_sync"]  # Fallback

        except Exception as e:
            logger.error(f"Cut suggestions generation failed: {e}")
            return ["beat_sync"]

    def _generate_multi_modal_tags(self, audio_features: dict, video_features: dict) -> list[str]:
        """Kombiniert Audio + Video Tags für Multi-Modal Understanding."""
        try:
            combined_tags = []

            # Video Tags
            video_tags = video_features.get("tags", [])
            combined_tags.extend(video_tags[:3])  # Top 3 video tags

            # Audio-derived Tags
            bpm = audio_features.get("bpm", 120)
            if bpm > 130:
                combined_tags.extend(["high_energy", "fast_tempo"])
            elif bpm < 100:
                combined_tags.extend(["slow_tempo", "chill"])
            else:
                combined_tags.append("medium_tempo")

            # Stems-derived Tags
            if audio_features.get("stems_available", False):
                combined_tags.append("stems_enhanced")

            # Remove duplicates while preserving order
            seen = set()
            unique_tags = []
            for tag in combined_tags:
                if tag not in seen:
                    seen.add(tag)
                    unique_tags.append(tag)

            return unique_tags[:5]  # Max 5 tags

        except Exception as e:
            logger.error(f"Multi-modal tag generation failed: {e}")
            return ["unknown"]


def create_multi_modal_analyzer(preset: str = "balanced") -> MultiModalAnalyzer:
    """
    Factory function für verschiedene Multi-Modal Analyzer Presets.

    Args:
        preset: "speed", "balanced", "quality"

    Returns:
        Konfigurierte MultiModalAnalyzer Instance
    """
    presets = {
        "speed": {"video_engine_size": "base", "stem_quality_mode": "speed"},
        "balanced": {"video_engine_size": "base", "stem_quality_mode": "auto"},
        "quality": {"video_engine_size": "large", "stem_quality_mode": "quality"},
    }

    config = presets.get(preset, presets["balanced"])

    return MultiModalAnalyzer(
        video_engine_size=config["video_engine_size"], stem_quality_mode=config["stem_quality_mode"]
    )
