"""
Video Intelligence Engine - CLIP-basierte Szenen-Erkennung und Auto-Tagging.

Neue Funktionen für Overnight Development:
- CLIP-basierte Semantic Scene Recognition
- Automatisches Video-Tagging System
- Multi-Modal Content Understanding
- Hierarchische Tag-Strukturen
- Confidence-Scoring für alle Erkennungen

Integration mit bestehendem System:
- Erweitert SemanticAnalyzer um Video-spezifische Features
- Kompatibel mit SceneAnalyzer für traditionelle Computer Vision
- Direkt integriert in Video Analysis Workflow
"""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from ...utils.logger import get_logger
from .scene_analyzer import SceneAnalysisResult, SceneAnalyzer
from .semantic_analyzer import SemanticAnalyzer

logger = get_logger(__name__)


@dataclass
class VideoSceneRecognition:
    """Ergebnis der CLIP-basierten Szenen-Erkennung."""

    # Semantic Scene Labels (CLIP-basiert)
    scene_labels: dict[str, float]  # Label -> Confidence Score
    primary_scene: str  # Haupt-Szene mit höchstem Score
    confidence: float  # Confidence des primary_scene

    # Traditionelle Computer Vision (aus SceneAnalyzer)
    traditional_analysis: SceneAnalysisResult | None = None

    # Kombinierte Intelligenz
    combined_tags: list[str] = None  # Finale Tags aus beiden Systemen
    quality_score: float = 0.0  # Gesamt-Qualität der Erkennung (0-1)
    frame_index: int = 0  # Frame-Index für Referenz bei Video-Analyse

    def to_dict(self) -> dict:
        """Konvertiert zu Dictionary für DB-Speicherung."""
        result = {
            "scene_labels": self.scene_labels,
            "primary_scene": self.primary_scene,
            "confidence": self.confidence,
            "combined_tags": self.combined_tags,
            "quality_score": self.quality_score,
            "frame_index": self.frame_index,
        }

        if self.traditional_analysis:
            result["traditional_analysis"] = self.traditional_analysis.to_dict()

        return result


class VideoIntelligenceEngine:
    """
    Intelligente Video-Analyse mit CLIP + Computer Vision.

    Features:
    - CLIP-basierte Semantic Scene Recognition
    - Automatisches Multi-Level Tagging
    - Quality-bewusste Erkennung
    - Integration traditioneller CV-Methoden
    """

    # Standard Scene Labels für PB Studio
    DEFAULT_SCENE_LABELS = [
        # Umgebung/Ort
        "beach",
        "ocean",
        "water",
        "sea",
        "city",
        "urban",
        "street",
        "building",
        "nature",
        "forest",
        "mountain",
        "landscape",
        "indoor",
        "room",
        "house",
        "party",
        # Stimmung/Atmosphäre
        "sunset",
        "sunrise",
        "night",
        "evening",
        "bright",
        "dark",
        "colorful",
        "vibrant",
        "calm",
        "peaceful",
        "energetic",
        "dynamic",
        # Aktivitäten
        "dancing",
        "party",
        "celebration",
        "festival",
        "walking",
        "running",
        "sports",
        "music",
        "concert",
        "performance",
        "show",
        # Personen/Social
        "people",
        "crowd",
        "person",
        "group",
        "friends",
        "family",
        "couple",
        # Objekte/Content
        "car",
        "vehicle",
        "food",
        "drink",
        "animal",
        "dog",
        "cat",
        "bird",
        "flower",
        "plant",
        "tree",
    ]

    # Custom Label-Sets für verschiedene Anwendungsf älle
    PARTY_LABELS = ["party", "dancing", "celebration", "crowd", "music", "colorful", "energetic"]
    NATURE_LABELS = ["nature", "landscape", "peaceful", "calm", "beautiful", "outdoor", "scenic"]
    URBAN_LABELS = ["city", "urban", "street", "modern", "dynamic", "busy", "architecture"]

    def __init__(
        self,
        clip_model: str = "openai/clip-vit-base-patch32",
        confidence_threshold: float = 0.60,
        enable_traditional_cv: bool = True,
    ):
        """
        Initialize Video Intelligence Engine.

        Args:
            clip_model: CLIP model für semantic analysis
            confidence_threshold: Minimum confidence für Label-Zuordnung
            enable_traditional_cv: Ob traditionelle CV-Analyse verwendet werden soll
        """
        self.confidence_threshold = confidence_threshold
        self.enable_traditional_cv = enable_traditional_cv

        # CLIP-basierte Semantic Analysis
        self.semantic_analyzer = SemanticAnalyzer(
            model_name=clip_model, confidence_threshold=confidence_threshold
        )

        # Traditionelle Computer Vision (optional)
        self.scene_analyzer = SceneAnalyzer() if enable_traditional_cv else None

        logger.info(
            f"VideoIntelligenceEngine initialized (CLIP={clip_model}, CV={enable_traditional_cv})"
        )

    def analyze_frame(
        self,
        frame: np.ndarray | Image.Image | str,
        custom_labels: list[str] | None = None,
        use_quality_filtering: bool = True,
    ) -> VideoSceneRecognition:
        """
        Analysiert einen einzelnen Video-Frame für Scene Recognition.

        Args:
            frame: Video-Frame (numpy array, PIL Image, oder Pfad)
            custom_labels: Optionale custom Labels (default: DEFAULT_SCENE_LABELS)
            use_quality_filtering: Ob Quality-Filtering angewendet werden soll

        Returns:
            VideoSceneRecognition Ergebnis
        """
        try:
            # Frame preprocessing
            if isinstance(frame, str):
                frame = Image.open(frame)
            elif isinstance(frame, np.ndarray):
                # Convert BGR to RGB for CLIP
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)

            # Labels bestimmen
            labels = custom_labels if custom_labels else self.DEFAULT_SCENE_LABELS

            # CLIP-basierte Semantic Analysis
            semantic_scores = self.semantic_analyzer.analyze(frame, labels)

            # Finde primary scene (höchster Score)
            if semantic_scores:
                primary_scene = max(semantic_scores.items(), key=lambda x: x[1])
                primary_label, primary_confidence = primary_scene
            else:
                primary_label, primary_confidence = "unknown", 0.0

            # Traditionelle Computer Vision (optional)
            traditional_result = None
            if self.scene_analyzer and isinstance(frame, Image.Image):
                # Convert PIL to numpy for SceneAnalyzer
                frame_cv = np.array(frame)
                if frame_cv.shape[2] == 3:  # RGB to BGR for OpenCV
                    frame_cv = cv2.cvtColor(frame_cv, cv2.COLOR_RGB2BGR)
                traditional_result = self.scene_analyzer.analyze(frame_cv)

            # Kombiniere Results für finale Tags
            combined_tags = self._combine_analysis_results(semantic_scores, traditional_result)

            # Quality Score berechnen
            quality_score = self._calculate_quality_score(semantic_scores, traditional_result)

            # Filter by quality if enabled
            if use_quality_filtering and quality_score < 0.3:
                logger.debug(f"Low quality detection (score={quality_score:.2f}), using fallback")
                primary_label = "unknown"
                primary_confidence = 0.0
                combined_tags = ["unknown"]
                quality_score = 0.1

            result = VideoSceneRecognition(
                scene_labels=semantic_scores,
                primary_scene=primary_label,
                confidence=primary_confidence,
                traditional_analysis=traditional_result,
                combined_tags=combined_tags,
                quality_score=quality_score,
            )

            logger.debug(
                f"Scene Recognition: {primary_label} (conf={primary_confidence:.2f}, quality={quality_score:.2f})"
            )
            return result

        except Exception as e:
            logger.error(f"Error in frame analysis: {e}")
            # Fallback result
            return VideoSceneRecognition(
                scene_labels={},
                primary_scene="unknown",
                confidence=0.0,
                combined_tags=["unknown"],
                quality_score=0.0,
            )

    def analyze_video_sample(
        self, video_path: str, sample_frames: int = 5, custom_labels: list[str] | None = None
    ) -> list[VideoSceneRecognition]:
        """
        Analysiert Video-Sample mit mehreren Frames für robuste Erkennung.

        Args:
            video_path: Pfad zur Video-Datei
            sample_frames: Anzahl Frames zur Analyse (gleichmäßig verteilt)
            custom_labels: Optionale custom Labels

        Returns:
            Liste von VideoSceneRecognition Ergebnissen (eines pro Frame)
        """
        results = []

        try:
            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return results

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                logger.error(f"Video has no frames: {video_path}")
                return results

            # Gleichmäßig verteilte Frame-Indizes
            frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)

            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if ret:
                    result = self.analyze_frame(frame, custom_labels)
                    result.frame_index = frame_idx  # Add frame index for reference
                    results.append(result)
                else:
                    logger.warning(f"Could not read frame {frame_idx} from {video_path}")

            cap.release()
            logger.info(f"Analyzed {len(results)} frames from {video_path}")

        except Exception as e:
            logger.error(f"Error analyzing video {video_path}: {e}")

        return results

    def get_video_tags(
        self, video_path: str, confidence_threshold: float | None = None
    ) -> dict[str, any]:
        """
        Generiert finale Video-Tags basierend auf Frame-Analyse.

        Args:
            video_path: Pfad zur Video-Datei
            confidence_threshold: Überschreibt default threshold

        Returns:
            Dictionary mit finalen Tags und Metadaten
        """
        threshold = confidence_threshold or self.confidence_threshold

        # Analysiere Video-Sample
        frame_results = self.analyze_video_sample(video_path)

        if not frame_results:
            return {
                "tags": ["unknown"],
                "primary_scene": "unknown",
                "confidence": 0.0,
                "quality_score": 0.0,
                "frame_count": 0,
            }

        # Aggregiere Results über alle Frames
        all_labels = {}
        quality_scores = []

        for result in frame_results:
            quality_scores.append(result.quality_score)

            # Sammle alle Labels mit ihren Scores
            for label, score in result.scene_labels.items():
                if label not in all_labels:
                    all_labels[label] = []
                all_labels[label].append(score)

        # Berechne Durchschnitts-Scores
        aggregated_labels = {}
        for label, scores in all_labels.items():
            avg_score = np.mean(scores)
            if avg_score >= threshold:
                aggregated_labels[label] = avg_score

        # Bestimme primary scene
        if aggregated_labels:
            primary_scene = max(aggregated_labels.items(), key=lambda x: x[1])
            primary_label, primary_confidence = primary_scene
        else:
            primary_label, primary_confidence = "unknown", 0.0

        # Generiere finale Tags (Top 5)
        sorted_labels = sorted(aggregated_labels.items(), key=lambda x: x[1], reverse=True)
        final_tags = [label for label, score in sorted_labels[:5]]

        return {
            "tags": final_tags,
            "primary_scene": primary_label,
            "confidence": primary_confidence,
            "quality_score": np.mean(quality_scores) if quality_scores else 0.0,
            "frame_count": len(frame_results),
            "all_scores": aggregated_labels,
        }

    def _combine_analysis_results(
        self, semantic_scores: dict[str, float], traditional_result: SceneAnalysisResult | None
    ) -> list[str]:
        """Kombiniert CLIP + traditionelle CV für finale Tags."""
        tags = []

        # CLIP-basierte Tags (Top 3)
        if semantic_scores:
            sorted_semantic = sorted(semantic_scores.items(), key=lambda x: x[1], reverse=True)
            for label, score in sorted_semantic[:3]:
                if score >= self.confidence_threshold:
                    tags.append(label)

        # Traditionelle CV Tags (falls verfügbar)
        if traditional_result:
            # Füge Scene Types hinzu
            for scene_type in traditional_result.scene_types:
                if scene_type.lower() not in [tag.lower() for tag in tags]:
                    tags.append(scene_type.lower())

            # Füge Face Detection hinzu
            if traditional_result.has_face:
                if "person" not in tags and "people" not in tags:
                    tags.append("person" if traditional_result.face_count == 1 else "people")

        return tags if tags else ["unknown"]

    def _calculate_quality_score(
        self, semantic_scores: dict[str, float], traditional_result: SceneAnalysisResult | None
    ) -> float:
        """Berechnet Qualitäts-Score der Erkennung."""
        score = 0.0

        # CLIP-Quality (70% weight)
        if semantic_scores:
            max_semantic = max(semantic_scores.values())
            score += max_semantic * 0.7

        # Traditional CV Quality (30% weight)
        if traditional_result:
            # Edge density als Quality-Indikator
            cv_quality = min(traditional_result.edge_density * 2.0, 1.0)
            score += cv_quality * 0.3

        return min(score, 1.0)


def create_video_intelligence_engine(model_size: str = "base") -> VideoIntelligenceEngine:
    """
    Factory function für verschiedene Model-Größen.

    Args:
        model_size: "base", "large" für verschiedene CLIP-Models

    Returns:
        Konfigurierte VideoIntelligenceEngine
    """
    model_configs = {
        "base": "openai/clip-vit-base-patch32",
        "large": "openai/clip-vit-large-patch14",
    }

    model_name = model_configs.get(model_size, model_configs["base"])

    return VideoIntelligenceEngine(
        clip_model=model_name, confidence_threshold=0.60, enable_traditional_cv=True
    )
