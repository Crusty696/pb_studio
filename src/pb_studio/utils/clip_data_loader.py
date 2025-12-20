"""
Clip Data Loader for PB_studio

Handles loading and parsing of clip data.
Supports lazy loading of heavy JSON fields to optimize list view performance.
"""

import json
from typing import Any

from ..utils.logger import get_logger

logger = get_logger(__name__)


def parse_json_safely(json_str: str | None, default: Any = None) -> Any:
    """
    Safely parses a JSON string.
    Returns default value (empty dict or list) if parsing fails or input is None.
    """
    if not json_str:
        return default if default is not None else {}
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default if default is not None else {}


class ClipDataLoader:
    """
    Helper class to transform DB clip objects into lightweight dictionary structures.
    Optimizes performance by avoiding parsing of unused JSON fields for list views.
    """

    @staticmethod
    def db_to_dict(clip_db, full_details: bool = False) -> dict[str, Any]:
        """
        Convert SQLAlchemy VideoClip object to dictionary.

        Args:
            clip_db: VideoClip object from database
            full_details: If True, parses all JSON fields (slow).
                          If False, keeps heavy JSON fields as strings or omits them (fast).

        Returns:
            Dictionary with clip data
        """
        # Basic metadata (always needed)
        analysis_data = {}
        is_analyzed = False
        unanalyzed_count = 0

        # Check analysis status
        if clip_db.analysis_status:
            is_analyzed = (
                clip_db.analysis_status.is_fully_analyzed()
                if hasattr(clip_db.analysis_status, "is_fully_analyzed")
                else False
            )

        # 1. Color Analysis
        if clip_db.colors:
            analysis_data["color"] = {
                "temperature": clip_db.colors.temperature,
                "temperature_score": clip_db.colors.temperature_score,
                "brightness": clip_db.colors.brightness,
                "brightness_value": clip_db.colors.brightness_value,
                "dominant_colors": clip_db.colors.get_dominant_colors(),
                "color_moods": clip_db.colors.get_color_moods(),
                # Scalar fields - cheap to include always
                "brightness_dynamics": clip_db.colors.brightness_dynamics,
                "color_dynamics": clip_db.colors.color_dynamics,
                "temporal_rhythm": clip_db.colors.temporal_rhythm,
            }

        # 2. Motion Analysis
        if clip_db.motion:
            analysis_data["motion"] = {
                "motion_type": clip_db.motion.motion_type,
                "motion_score": clip_db.motion.motion_score,
                "motion_rhythm": clip_db.motion.motion_rhythm,
                "camera_motion": clip_db.motion.camera_motion,
                "camera_magnitude": clip_db.motion.camera_magnitude,
            }

        # 3. Scene Analysis
        if clip_db.scene_type:
            analysis_data["scene"] = {
                "scene_types": clip_db.scene_type.get_scene_types(),
                "has_face": clip_db.scene_type.has_face,
                "face_count": clip_db.scene_type.face_count,
                # Scalar fields - cheap to include always
                "edge_density": clip_db.scene_type.edge_density,
                "texture_variance": clip_db.scene_type.texture_variance,
                "depth_of_field": clip_db.scene_type.depth_of_field,
            }

        # 4. Mood Analysis
        if clip_db.mood:
            analysis_data["mood"] = {
                "moods": clip_db.mood.get_moods(),
                "brightness": clip_db.mood.brightness,
                "saturation": clip_db.mood.saturation,
                "energy": clip_db.mood.energy,
                # Scalar fields
                "contrast": clip_db.mood.contrast,
                "warm_ratio": clip_db.mood.warm_ratio,
                "cool_ratio": clip_db.mood.cool_ratio,
            }
            if full_details:
                # Heavy JSON parsing only when needed
                analysis_data["mood"]["mood_scores"] = parse_json_safely(clip_db.mood.mood_scores, {})
            else:
                # Pass raw string for filtering or lazy parsing later
                analysis_data["mood"]["_mood_scores_raw"] = clip_db.mood.mood_scores

        # 5. Style Analysis
        if clip_db.style:
            analysis_data["style"] = {
                "styles": clip_db.style.get_styles(),
                "sharpness": clip_db.style.sharpness,
                "noise_level": clip_db.style.noise_level,
                "vignette_score": clip_db.style.vignette_score,
                # Scalar fields
                "dynamic_range": clip_db.style.dynamic_range,
                "saturation_mean": clip_db.style.saturation_mean,
                "saturation_std": clip_db.style.saturation_std,
                "mean_brightness": clip_db.style.mean_brightness,
            }

        # 6. Object Detection
        if clip_db.objects:
            analysis_data["objects"] = {
                "detected_objects": clip_db.objects.get_detected_objects(),
                "content_tags": clip_db.objects.get_content_tags(),
                # Scalar fields
                "line_count": clip_db.objects.line_count,
                "green_ratio": clip_db.objects.green_ratio,
                "sky_ratio": clip_db.objects.sky_ratio,
                "symmetry": clip_db.objects.symmetry,
            }
            if full_details:
                analysis_data["objects"]["object_counts"] = parse_json_safely(clip_db.objects.object_counts, {})
                analysis_data["objects"]["confidence_scores"] = parse_json_safely(clip_db.objects.confidence_scores, {})
            else:
                analysis_data["objects"]["_object_counts_raw"] = clip_db.objects.object_counts
                analysis_data["objects"]["_confidence_scores_raw"] = clip_db.objects.confidence_scores

        # Finalize analysis status
        # Restore needs_reanalysis logic: not analyzed if analysis_data empty OR needs_reanalysis flag set
        needs_analysis = not analysis_data or getattr(clip_db, "needs_reanalysis", True)

        # Consistent with DB worker: is_analyzed is True if fully_analyzed AND has data
        is_analyzed_final = is_analyzed and bool(analysis_data)

        if needs_analysis:
            unanalyzed_count = 1

        clip_data = {
            "id": clip_db.id,
            "name": clip_db.name,
            "file_path": clip_db.file_path,
            "duration": clip_db.duration or 0.0,
            "width": clip_db.width or 0,
            "height": clip_db.height or 0,
            "fps": clip_db.fps or 30.0,
            "date_added": str(clip_db.created_at) if clip_db.created_at else "",
            "thumbnail_path": clip_db.thumbnail_path,
            "analysis": analysis_data,
            "is_analyzed": is_analyzed_final,
            "content_fingerprint": getattr(clip_db, "content_fingerprint", None),
            "_unanalyzed_count": unanalyzed_count,
        }

        return clip_data

    @staticmethod
    def ensure_details(clip_data: dict[str, Any]) -> dict[str, Any]:
        """
        Ensures that all heavy JSON fields are parsed in the clip dictionary.
        Modifies the dictionary in-place.
        """
        analysis = clip_data.get("analysis", {})

        # Mood
        if "mood" in analysis:
            if "_mood_scores_raw" in analysis["mood"]:
                analysis["mood"]["mood_scores"] = parse_json_safely(analysis["mood"].pop("_mood_scores_raw"), {})

        # Objects
        if "objects" in analysis:
            if "_object_counts_raw" in analysis["objects"]:
                analysis["objects"]["object_counts"] = parse_json_safely(analysis["objects"].pop("_object_counts_raw"), {})
            if "_confidence_scores_raw" in analysis["objects"]:
                analysis["objects"]["confidence_scores"] = parse_json_safely(analysis["objects"].pop("_confidence_scores_raw"), {})

        return clip_data
