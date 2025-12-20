"""
Video Compatibility Manager für Overnight Development.

Multi-Resolution und FPS Video Compatibility System:
✅ Auto-Detection von Video-Parametern (Resolution, FPS, Codec)
✅ Adaptive Video Processing für verschiedene Formate
✅ Intelligent Frame Sampling basierend auf FPS
✅ Optimal Scaling für AI-Modelle (224x224, 336x336, 512x512)
✅ Quality-Preserving Video Conversion
✅ Performance-Optimized Processing Pipelines
✅ Support für 4K, 1080p, 720p, 480p + variable FPS
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from ..core.hardware import get_device_info
from ..utils.gpu_memory import get_gpu_memory_info
from ..utils.logger import get_logger
from .video_analyzer import VideoAnalyzer, VideoFrameRate, VideoProperties, VideoResolution

logger = get_logger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for video processing."""

    # Target Properties
    target_width: int | None = None
    target_height: int | None = None
    target_fps: float | None = None

    # Quality Settings
    quality_preference: float = 0.7  # 0.0=speed, 1.0=quality
    preserve_aspect_ratio: bool = True

    # AI Model Optimization
    ai_model_input_size: tuple[int, int] = (224, 224)  # Default CLIP size
    enable_smart_cropping: bool = True

    # Performance Settings
    max_memory_usage_mb: int = 2000
    enable_gpu_acceleration: bool = True
    enable_hardware_decode: bool = True

    # Frame Sampling
    max_frames_for_analysis: int = 100  # Limit frames for AI analysis
    sample_strategy: str = "uniform"  # "uniform", "keyframes", "smart"

    # Output Settings
    maintain_sync: bool = True  # Maintain audio-video sync
    output_codec: str = "h264"


class CompatibilityChecker:
    """Check video compatibility and suggest optimizations."""

    def __init__(self):
        self.device_info = get_device_info()
        self.gpu_memory_info = get_gpu_memory_info()

        # Performance limits based on hardware
        self._max_resolution = self._determine_max_resolution()
        self._max_fps_processing = self._determine_max_fps()

    def _determine_max_resolution(self) -> tuple[int, int]:
        """Determine maximum processable resolution."""
        device_type = self.device_info.get("device_type", "cpu")

        if device_type == "cpu":
            return (1280, 720)  # CPU limited to 720p for real-time
        elif device_type == "directml":
            return (1920, 1080)  # DirectML can handle 1080p
        elif device_type == "cuda":
            gpu_memory = self.gpu_memory_info.get("total_gb", 4) if self.gpu_memory_info else 4
            if gpu_memory >= 8:
                return (3840, 2160)  # 4K with sufficient VRAM
            else:
                return (1920, 1080)  # 1080p for lower VRAM
        else:
            return (1280, 720)  # Conservative default

    def _determine_max_fps(self) -> float:
        """Determine maximum processable FPS."""
        device_type = self.device_info.get("device_type", "cpu")

        if device_type == "cpu":
            return 30.0  # CPU limited
        elif device_type in ["directml", "cuda"]:
            return 60.0  # GPU can handle higher FPS
        else:
            return 24.0  # Conservative

    def analyze_video_compatibility(self, video_path: str) -> VideoProperties:
        """Analyze video for compatibility using VideoAnalyzer."""
        analyzer = VideoAnalyzer(use_cache=False)
        properties = analyzer.analyze_video_properties(video_path)

        if properties:
            return properties

        logger.error(f"Video compatibility analysis failed for {video_path}")
        return VideoProperties(
            width=1280,
            height=720,
            fps=25.0,
            frame_count=0,
            duration=0.0,
            codec="unknown",
            quality_score=0.5,
        )

    def _calculate_quality_score(self, width: int, height: int, fps: float) -> float:
        """Calculate video quality score (0.0-1.0)."""
        # Resolution component (up to 0.6)
        resolution_score = min(width * height / (1920 * 1080), 1.0) * 0.6

        # FPS component (up to 0.4)
        fps_score = min(fps / 60.0, 1.0) * 0.4

        return min(resolution_score + fps_score, 1.0)

    def _calculate_optimal_sample_rate(self, fps: float, frame_count: int) -> float:
        """Calculate optimal frame sampling rate."""
        # For AI analysis, we don't need every frame for high FPS videos
        if fps >= 60:
            return 0.25  # Process every 4th frame for 60+ fps
        elif fps >= 30:
            return 0.5  # Process every 2nd frame for 30-60 fps
        else:
            return 1.0  # Process all frames for low fps

    def _get_recommended_resize(self, width: int, height: int) -> tuple[int, int] | None:
        """Get recommended resize for AI processing."""
        # If resolution exceeds processing limits, suggest resize
        max_w, max_h = self._max_resolution

        if width > max_w or height > max_h:
            # Calculate scaling factor preserving aspect ratio
            scale_w = max_w / width
            scale_h = max_h / height
            scale = min(scale_w, scale_h)

            new_w = int(width * scale)
            new_h = int(height * scale)

            # Ensure dimensions are even (required for many codecs)
            new_w = (new_w // 2) * 2
            new_h = (new_h // 2) * 2

            return (new_w, new_h)

        return None

    def _check_gpu_decode_support(self, codec: str) -> bool:
        """Check if codec supports GPU decoding."""
        gpu_supported_codecs = ["h264", "h265", "vp9", "av1"]
        return codec.lower() in gpu_supported_codecs


class VideoPreprocessor:
    """Preprocess videos for optimal AI analysis."""

    def __init__(self, config: ProcessingConfig | None = None):
        self.config = config or ProcessingConfig()
        self.compatibility_checker = CompatibilityChecker()

    def preprocess_for_ai_analysis(
        self, video_path: str, video_properties: VideoProperties | None = None
    ) -> dict[str, Any]:
        """
        Preprocess video for AI analysis with optimal frame sampling.

        Args:
            video_path: Path to video file
            video_properties: Pre-analyzed video properties

        Returns:
            Dictionary with preprocessed frames and metadata
        """
        if video_properties is None:
            video_properties = self.compatibility_checker.analyze_video_compatibility(video_path)

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")

            frames = []
            frame_timestamps = []
            frame_indices = []

            # Determine frame sampling strategy
            total_frames = video_properties.frame_count
            max_frames = min(self.config.max_frames_for_analysis, total_frames)

            if self.config.sample_strategy == "uniform":
                # Uniform sampling across video
                if total_frames <= max_frames:
                    frame_step = 1
                else:
                    frame_step = total_frames / max_frames

                sample_indices = [int(i * frame_step) for i in range(max_frames)]

            elif self.config.sample_strategy == "keyframes":
                # Keyframe detection using VideoAnalyzer (PySceneDetect)
                try:
                    # Initialize analyzer (use cache for performance if available)
                    # Use lower threshold (15.0) to be more sensitive for keyframe detection
                    analyzer = VideoAnalyzer(use_cache=True)
                    scene_timestamps = analyzer.get_scene_timestamps(video_path, threshold=15.0)

                    if scene_timestamps and len(scene_timestamps) > 0:
                        detected_indices = []
                        fps = video_properties.fps

                        # Process timestamps in pairs [start, end, start, end...]
                        for i in range(0, len(scene_timestamps), 2):
                            if i + 1 >= len(scene_timestamps):
                                break

                            start_sec = scene_timestamps[i]
                            end_sec = scene_timestamps[i+1]

                            # Select middle frame of the scene as keyframe
                            mid_sec = (start_sec + end_sec) / 2
                            mid_frame = int(mid_sec * fps)

                            # Ensure frame index is valid
                            if 0 <= mid_frame < total_frames:
                                detected_indices.append(mid_frame)

                        # Handle sampling if we found scenes
                        if detected_indices:
                            # If more scenes than max_frames, sample uniformly from scenes
                            if len(detected_indices) > max_frames:
                                step = len(detected_indices) / max_frames
                                sample_indices = [detected_indices[int(i * step)] for i in range(max_frames)]
                            else:
                                # Use all detected scenes
                                sample_indices = detected_indices

                            # Ensure sorted and unique
                            sample_indices = sorted(list(set(sample_indices)))
                        else:
                            # Fallback if timestamps yielded no valid frames
                            logger.warning(f"No valid frames extracted from {len(scene_timestamps)//2} scenes, falling back to uniform")
                            sample_indices = list(range(0, total_frames, max(1, total_frames // max_frames)))
                    else:
                        logger.info("No scenes detected, falling back to uniform sampling")
                        sample_indices = list(range(0, total_frames, max(1, total_frames // max_frames)))

                except Exception as e:
                    logger.warning(f"Keyframe detection failed, falling back to uniform: {e}")
                    sample_indices = list(range(0, total_frames, max(1, total_frames // max_frames)))

            else:  # smart sampling
                sample_indices = self._smart_frame_sampling(video_properties, max_frames)

            # Extract frames
            for idx in sample_indices[:max_frames]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()

                if not ret:
                    continue

                # Preprocess frame
                processed_frame = self._preprocess_frame(frame, video_properties)

                frames.append(processed_frame)
                frame_timestamps.append(idx / video_properties.fps)
                frame_indices.append(idx)

            cap.release()

            result = {
                "frames": frames,
                "timestamps": frame_timestamps,
                "indices": frame_indices,
                "original_properties": video_properties,
                "processing_config": self.config,
                "total_extracted": len(frames),
                "sampling_ratio": len(frames) / total_frames if total_frames > 0 else 0.0,
            }

            logger.info(
                f"Extracted {len(frames)} frames from {Path(video_path).name} "
                f"(sampling ratio: {result['sampling_ratio']:.2f})"
            )

            return result

        except Exception as e:
            logger.error(f"Video preprocessing failed: {e}")
            return {
                "frames": [],
                "timestamps": [],
                "indices": [],
                "original_properties": video_properties,
                "error": str(e),
            }

    def _smart_frame_sampling(self, properties: VideoProperties, max_frames: int) -> list[int]:
        """Implement smart frame sampling based on video properties."""
        total_frames = properties.frame_count

        if total_frames <= max_frames:
            return list(range(total_frames))

        # For high FPS videos, sample more aggressively
        if properties.fps >= 60:
            # Focus on key moments (beginning, middle, end + uniform sampling)
            key_frames = [
                0,
                total_frames // 4,
                total_frames // 2,
                3 * total_frames // 4,
                total_frames - 1,
            ]
            remaining = max_frames - len(key_frames)

            if remaining > 0:
                uniform_step = total_frames / remaining
                uniform_frames = [int(i * uniform_step) for i in range(remaining)]
                all_frames = sorted(list(set(key_frames + uniform_frames)))
                return all_frames[:max_frames]
            else:
                return key_frames

        else:
            # For normal FPS, use uniform sampling
            step = total_frames / max_frames
            return [int(i * step) for i in range(max_frames)]

    def _preprocess_frame(self, frame: np.ndarray, properties: VideoProperties) -> np.ndarray:
        """Preprocess individual frame for AI analysis."""
        # Resize if needed
        if self.config.ai_model_input_size != (frame.shape[1], frame.shape[0]):
            target_w, target_h = self.config.ai_model_input_size

            if self.config.preserve_aspect_ratio:
                # Calculate scaling preserving aspect ratio
                scale = min(target_w / frame.shape[1], target_h / frame.shape[0])
                new_w = int(frame.shape[1] * scale)
                new_h = int(frame.shape[0] * scale)

                # Resize and pad if necessary
                resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

                # Create padded frame
                padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                start_x = (target_w - new_w) // 2
                start_y = (target_h - new_h) // 2
                padded[start_y : start_y + new_h, start_x : start_x + new_w] = resized

                frame = padded
            else:
                # Direct resize without aspect ratio preservation
                frame = cv2.resize(
                    frame, self.config.ai_model_input_size, interpolation=cv2.INTER_LANCZOS4
                )

        # Convert BGR to RGB (OpenCV uses BGR, AI models expect RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame


class VideoCompatibilityManager:
    """
    Main manager for video compatibility and processing.

    DISTINCTION FROM VIDEO ANALYZER:
    - VideoCompatibilityManager: Focuses on technical properties (codec, resolution, fps),
      transcoding, and preparing frames for AI-models (resizing, sampling).
    - VideoAnalyzer: Focuses on content analysis (Scene Detection, Motion, Semantics).

    This class ensures videos are technically playable and optimized for the pipeline
    before deep analysis begins.
    """

    def __init__(self):
        self.compatibility_checker = CompatibilityChecker()
        self.default_config = ProcessingConfig()

        logger.info("Video Compatibility Manager initialized")

    def analyze_and_preprocess(
        self, video_path: str, processing_config: ProcessingConfig | None = None
    ) -> dict[str, Any]:
        """Complete video analysis and preprocessing pipeline."""

        config = processing_config or self.default_config

        # Step 1: Analyze video compatibility
        properties = self.compatibility_checker.analyze_video_compatibility(video_path)

        # Step 2: Generate processing recommendations
        recommendations = self._generate_processing_recommendations(properties, config)

        # Step 3: Preprocess for AI analysis
        preprocessor = VideoPreprocessor(config)
        preprocessed = preprocessor.preprocess_for_ai_analysis(video_path, properties)

        return {
            "properties": properties,
            "recommendations": recommendations,
            "preprocessed": preprocessed,
            "compatibility_score": self._calculate_compatibility_score(properties),
            "processing_time_estimate": self._estimate_processing_time(properties),
        }

    def _generate_processing_recommendations(
        self, properties: VideoProperties, config: ProcessingConfig
    ) -> dict[str, Any]:
        """Generate processing recommendations based on video properties."""
        recommendations = {
            "suggested_changes": [],
            "performance_impact": "low",
            "memory_optimization": [],
            "quality_optimization": [],
        }

        # Resolution recommendations
        if properties.recommended_resize:
            recommendations["suggested_changes"].append(
                {
                    "type": "resize",
                    "from": (properties.width, properties.height),
                    "to": properties.recommended_resize,
                    "reason": "Optimize for hardware limits",
                }
            )

        # FPS recommendations
        if properties.fps > 60:
            recommendations["suggested_changes"].append(
                {
                    "type": "frame_sampling",
                    "current_fps": properties.fps,
                    "suggested_sample_rate": properties.optimal_sample_rate,
                    "reason": "High FPS optimization for AI analysis",
                }
            )

        # Memory optimization
        if properties.total_memory_estimate_mb > config.max_memory_usage_mb:
            recommendations["memory_optimization"].append(
                f"High memory usage ({properties.total_memory_estimate_mb:.0f}MB). "
                f"Consider frame sampling or resize."
            )
            recommendations["performance_impact"] = "high"

        # Codec recommendations
        if not properties.supports_gpu_decode:
            recommendations["suggested_changes"].append(
                {
                    "type": "codec_warning",
                    "current_codec": properties.codec,
                    "reason": "Codec may not support GPU acceleration",
                }
            )

        return recommendations

    def _calculate_compatibility_score(self, properties: VideoProperties) -> float:
        """Calculate overall compatibility score (0.0-1.0)."""
        score = 0.0

        # Resolution score (30%)
        max_res = self.compatibility_checker._max_resolution
        res_score = min(1.0, min(max_res[0] / properties.width, max_res[1] / properties.height))
        score += res_score * 0.3

        # FPS score (20%)
        max_fps = self.compatibility_checker._max_fps_processing
        fps_score = min(1.0, max_fps / properties.fps)
        score += fps_score * 0.2

        # Codec compatibility (25%)
        codec_score = 1.0 if properties.supports_gpu_decode else 0.5
        score += codec_score * 0.25

        # Memory efficiency (25%)
        memory_score = max(0.0, 1.0 - properties.total_memory_estimate_mb / 4000)  # 4GB threshold
        score += memory_score * 0.25

        return min(1.0, score)

    def _estimate_processing_time(self, properties: VideoProperties) -> dict[str, float]:
        """Estimate processing time for different operations."""
        base_time_per_frame = 0.05  # 50ms per frame baseline

        # Adjust based on resolution
        resolution_factor = (properties.width * properties.height) / (1920 * 1080)

        # Adjust based on device type
        device_factor = {"cpu": 3.0, "directml": 1.5, "cuda": 1.0}.get(
            self.compatibility_checker.device_info.get("device_type", "cpu"), 2.0
        )

        frame_time = base_time_per_frame * resolution_factor * device_factor

        # Calculate for different analysis modes
        frames_to_process = min(properties.frame_count * properties.optimal_sample_rate, 100)

        return {
            "frame_extraction": frames_to_process * 0.01,  # 10ms per frame extraction
            "ai_analysis": frames_to_process * frame_time,
            "preprocessing": properties.frame_count * 0.005,  # 5ms per frame preprocessing
            "total_estimate": frames_to_process * (0.01 + frame_time)
            + properties.frame_count * 0.005,
        }

    def get_supported_formats(self) -> dict[str, list[str]]:
        """Get list of supported video formats."""
        return {
            "resolutions": [
                "8K (7680x4320)",
                "4K (3840x2160)",
                "1080p (1920x1080)",
                "720p (1280x720)",
                "480p (720x480)",
                "Custom",
            ],
            "frame_rates": [
                "24 fps (Cinema)",
                "25 fps (PAL)",
                "29.97 fps (NTSC)",
                "30 fps",
                "60 fps",
                "120 fps",
                "Variable FPS",
            ],
            "codecs": ["H.264/AVC", "H.265/HEVC", "VP9", "AV1", "MPEG-4", "MPEG-2", "ProRes"],
            "containers": [".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".m4v"],
        }


# Global instance
_video_compatibility_manager = None


def get_video_compatibility_manager() -> VideoCompatibilityManager:
    """Get global video compatibility manager."""
    global _video_compatibility_manager
    if _video_compatibility_manager is None:
        _video_compatibility_manager = VideoCompatibilityManager()
    return _video_compatibility_manager
