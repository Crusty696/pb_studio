"""
Database Worker Thread für PB_studio

Führt alle Datenbank-Operationen in Background Thread aus,
um UI non-blocking zu halten.

Author: PB_studio Development Team
"""

import json
import threading
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.sql.expression import false, true  # FIX H-02/H-03: SQLAlchemy Boolean Best Practice

from ..utils.logger import get_logger
from .models import VideoClip
from .models_analysis import (
    ClipAnalysisStatus,
    ClipColors,
    ClipMood,
    ClipMotion,
    ClipSceneType,
    ClipStyle,
)

logger = get_logger(__name__)


class DatabaseWorker(QObject):
    """
    Worker für alle DB-Operations im Background Thread.

    Signals:
        query_completed: Emitted when query finishes successfully (data: List[Dict])
        error_occurred: Emitted on error (error_msg: str)
        progress_update: Emitted for progress (current: int, total: int, message: str)
    """

    # Signals
    query_completed = pyqtSignal(list)  # List[Dict[str, Any]] - clip data
    error_occurred = pyqtSignal(str)  # Error message
    progress_update = pyqtSignal(int, int, str)  # current, total, message

    def __init__(self, session: Session | None):
        """
        Initialize Database Worker.

        Args:
            session: SQLAlchemy session for database operations
        """
        super().__init__()
        self.session = session
        # FIX: Use threading.Event() instead of bool for thread-safe cancellation
        # Bool flags are not atomic and can cause race conditions between threads
        self._cancel_event = threading.Event()

    def __del__(self):
        """
        FIX #10: Destruktor schließt die Session um Leaks zu vermeiden.

        Ohne diesen Destruktor bleibt die Session offen wenn der Worker
        garbage collected wird.
        """
        try:
            if hasattr(self, "session") and self.session is not None:
                self.session.close()
                self.session = None
        except Exception:
            # Fehler im Destruktor nicht propagieren
            pass

    def cleanup(self):
        """
        FIX #10: Explizite Cleanup-Methode für manuelle Freigabe.

        Sollte aufgerufen werden wenn der Worker nicht mehr benötigt wird.
        """
        try:
            if self.session is not None:
                self.session.close()
                self.session = None
            # FIX: Use Event.set() for thread-safe cancellation
            self._cancel_event.set()
        except Exception as e:
            logger.warning(f"Error during db_worker cleanup: {e}")

    @pyqtSlot()
    def query_clips(self, filters: dict[str, Any] | None = None):
        """
        Query clips in background thread with eager loading.

        Args:
            filters: Optional dictionary of filters to apply
        """
        try:
            # Cancel is per-operation; reset at the start of a new query.
            # FIX: Use Event.clear() for thread-safe reset
            self._cancel_event.clear()

            logger.info("DatabaseWorker: Starting clip query...")
            self.progress_update.emit(0, 100, "Querying database...")

            if self.session is None:
                from .connection import get_db_manager

                logger.warning(
                    "DatabaseWorker: session is None, creating new session via get_db_manager()"
                )
                dbm = get_db_manager()
                self.session = dbm.get_session()

            # Eager Loading: Alle Analyse-Daten in EINEM Query laden
            query = (
                self.session.query(VideoClip)
                .filter(VideoClip.is_available.is_(True))  # FIX H-02: SQLAlchemy Boolean Best Practice
                .options(
                    # Alle Relationships eager loaden (aus Phase 1)
                    joinedload(VideoClip.analysis_status),
                    joinedload(VideoClip.colors),
                    joinedload(VideoClip.motion),
                    joinedload(VideoClip.scene_type),
                    joinedload(VideoClip.mood),
                    joinedload(VideoClip.objects),
                    joinedload(VideoClip.style),
                    joinedload(VideoClip.fingerprint),
                )
            )

            # Apply filters if provided
            if filters:
                query = self._apply_filters(query, filters)

            # Execute query
            self.progress_update.emit(10, 100, "Loading clips...")
            db_clips = query.all()

            logger.info(f"DatabaseWorker: Loaded {len(db_clips)} clips from database")
            self.progress_update.emit(30, 100, f"Processing {len(db_clips)} clips...")

            # Convert to dictionaries with analysis data
            clips_data = []
            total = len(db_clips)
            unanalyzed_count = 0

            for i, clip_db in enumerate(db_clips):
                # FIX: Use Event.is_set() for thread-safe cancellation check
                if self._cancel_event.is_set():
                    logger.info("DatabaseWorker: Query cancelled by user")
                    return

                # Build analysis data dict (direkt aus Relationships)
                analysis_data = {}
                is_analyzed = False

                # Check analysis status
                if clip_db.analysis_status:
                    is_analyzed = (
                        clip_db.analysis_status.is_fully_analyzed()
                        if hasattr(clip_db.analysis_status, "is_fully_analyzed")
                        else False
                    )

                # Load color analysis
                if clip_db.colors:
                    analysis_data["color"] = {
                        "temperature": clip_db.colors.temperature,
                        "temperature_score": clip_db.colors.temperature_score,
                        "brightness": clip_db.colors.brightness,
                        "brightness_value": clip_db.colors.brightness_value,
                        "dominant_colors": clip_db.colors.get_dominant_colors(),
                        "color_moods": clip_db.colors.get_color_moods(),
                    }

                # Load motion analysis
                if clip_db.motion:
                    analysis_data["motion"] = {
                        "motion_type": clip_db.motion.motion_type,
                        "motion_score": clip_db.motion.motion_score,
                        "motion_rhythm": clip_db.motion.motion_rhythm,
                        "camera_motion": clip_db.motion.camera_motion,
                        "camera_magnitude": clip_db.motion.camera_magnitude,
                    }

                # Load scene analysis
                if clip_db.scene_type:
                    analysis_data["scene"] = {
                        "scene_types": clip_db.scene_type.get_scene_types(),
                        "has_face": clip_db.scene_type.has_face,
                        "face_count": clip_db.scene_type.face_count,
                        "edge_density": clip_db.scene_type.edge_density,
                        "depth_of_field": clip_db.scene_type.depth_of_field,
                    }

                # Load mood analysis
                if clip_db.mood:
                    # Defensive JSON-Parsing fuer mood_scores
                    mood_scores = {}
                    if clip_db.mood.mood_scores:
                        try:
                            mood_scores = json.loads(clip_db.mood.mood_scores)
                        except (json.JSONDecodeError, TypeError):
                            logger.warning(f"Clip {clip_db.id}: Invalid mood_scores JSON")

                    analysis_data["mood"] = {
                        "moods": clip_db.mood.get_moods(),
                        "mood_scores": mood_scores,
                        "brightness": clip_db.mood.brightness,
                        "saturation": clip_db.mood.saturation,
                        "energy": clip_db.mood.energy,
                    }

                # Load style analysis
                if clip_db.style:
                    analysis_data["style"] = {
                        "styles": clip_db.style.get_styles(),
                        "sharpness": clip_db.style.sharpness,
                        "noise_level": clip_db.style.noise_level,
                        "vignette_score": clip_db.style.vignette_score,
                    }

                # Load object detection
                if clip_db.objects:
                    # Defensive JSON-Parsing fuer object_counts
                    object_counts = {}
                    if clip_db.objects.object_counts:
                        try:
                            object_counts = json.loads(clip_db.objects.object_counts)
                        except (json.JSONDecodeError, TypeError):
                            logger.warning(f"Clip {clip_db.id}: Invalid object_counts JSON")

                    analysis_data["objects"] = {
                        "detected_objects": clip_db.objects.get_detected_objects(),
                        "object_counts": object_counts,
                        "content_tags": clip_db.objects.get_content_tags(),
                    }

                # FIX: Konsistente Logik für is_analyzed und needs_analysis
                # Ein Clip gilt als analysiert wenn:
                # 1. analysis_status.is_fully_analyzed() == True UND
                # 2. Es tatsächlich analysis_data gibt
                is_analyzed_final = is_analyzed and bool(analysis_data)

                # needs_analysis ist das exakte Gegenteil
                if not is_analyzed_final:
                    unanalyzed_count += 1

                # Build clip data dict
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
                    "_unanalyzed_count": unanalyzed_count,  # Metadata for UI
                }

                clips_data.append(clip_data)

                # Progress update every 20 clips
                if (i + 1) % 20 == 0 or (i + 1) == total:
                    progress = int(30 + (i / total) * 70)  # 30% to 100%
                    self.progress_update.emit(progress, 100, f"Processing clip {i + 1}/{total}...")

            # Emit results
            self.progress_update.emit(100, 100, f"Loaded {len(clips_data)} clips")
            logger.info(
                f"DatabaseWorker: Query complete - {len(clips_data)} clips, {unanalyzed_count} unanalyzed"
            )

            self.query_completed.emit(clips_data)

        except Exception as e:
            error_msg = f"Database query failed: {type(e).__name__}: {e}"
            logger.error(error_msg, exc_info=True)
            self.error_occurred.emit(error_msg)
        finally:
            # CRITICAL-03 FIX: Always cleanup session, even on exception
            # Prevents DB connection pool exhaustion
            self.cleanup()

    def _apply_filters(self, query, filters: dict[str, Any]):
        """
        Apply filters to the query dynamically.

        Supported filters:
            - motion_type: str - Filter by motion type (STATIC, SLOW, MEDIUM, FAST, EXTREME)
            - mood: str or List[str] - Filter by mood tags (JSON array contains)
            - scene_type: str or List[str] - Filter by scene type tags (JSON array contains)
            - style: str or List[str] - Filter by style tags (JSON array contains)
            - duration_range: Tuple[float, float] - Filter by duration (min, max) in seconds
            - brightness_range: Tuple[float, float] - Filter by brightness value (0.0-1.0)
            - temperature: str - Filter by color temperature (WARM, COOL, NEUTRAL)
            - has_face: bool - Filter clips with/without detected faces
            - energy_range: Tuple[float, float] - Filter by energy level (0.0-1.0)
            - is_analyzed: bool - Filter by analysis status

        Args:
            query: SQLAlchemy query object
            filters: Dictionary of filter criteria

        Returns:
            Modified query with filters applied
        """
        if not filters:
            return query

        # Filter: motion_type (requires JOIN to ClipMotion)
        if "motion_type" in filters and filters["motion_type"]:
            motion_type = filters["motion_type"]
            if isinstance(motion_type, list):
                query = query.join(ClipMotion).filter(ClipMotion.motion_type.in_(motion_type))
            else:
                query = query.join(ClipMotion).filter(ClipMotion.motion_type == motion_type)
            logger.debug(f"Filter applied: motion_type = {motion_type}")

        # Filter: mood (requires JOIN to ClipMood, JSON contains check)
        if "mood" in filters and filters["mood"]:
            mood_filter = filters["mood"]
            moods_to_check = [mood_filter] if isinstance(mood_filter, str) else mood_filter
            # SQLite JSON: Use LIKE for JSON array contains check
            mood_conditions = []
            for mood in moods_to_check:
                mood_conditions.append(ClipMood.moods.like(f'%"{mood}"%'))
            if mood_conditions:
                from sqlalchemy import or_

                query = query.join(ClipMood).filter(or_(*mood_conditions))
            logger.debug(f"Filter applied: mood in {moods_to_check}")

        # Filter: scene_type (requires JOIN to ClipSceneType, JSON contains check)
        if "scene_type" in filters and filters["scene_type"]:
            scene_filter = filters["scene_type"]
            scenes_to_check = [scene_filter] if isinstance(scene_filter, str) else scene_filter
            scene_conditions = []
            for scene in scenes_to_check:
                scene_conditions.append(ClipSceneType.scene_types.like(f'%"{scene}"%'))
            if scene_conditions:
                from sqlalchemy import or_

                query = query.join(ClipSceneType).filter(or_(*scene_conditions))
            logger.debug(f"Filter applied: scene_type in {scenes_to_check}")

        # Filter: style (requires JOIN to ClipStyle, JSON contains check)
        if "style" in filters and filters["style"]:
            style_filter = filters["style"]
            styles_to_check = [style_filter] if isinstance(style_filter, str) else style_filter
            style_conditions = []
            for style in styles_to_check:
                style_conditions.append(ClipStyle.styles.like(f'%"{style}"%'))
            if style_conditions:
                from sqlalchemy import or_

                query = query.join(ClipStyle).filter(or_(*style_conditions))
            logger.debug(f"Filter applied: style in {styles_to_check}")

        # Filter: duration_range (direct on VideoClip)
        if "duration_range" in filters and filters["duration_range"]:
            duration_range = filters["duration_range"]
            if isinstance(duration_range, (list, tuple)) and len(duration_range) == 2:
                min_duration, max_duration = duration_range
                if min_duration is not None:
                    query = query.filter(VideoClip.duration >= min_duration)
                if max_duration is not None:
                    query = query.filter(VideoClip.duration <= max_duration)
                logger.debug(f"Filter applied: duration_range = [{min_duration}, {max_duration}]")

        # Filter: brightness_range (requires JOIN to ClipColors)
        if "brightness_range" in filters and filters["brightness_range"]:
            brightness_range = filters["brightness_range"]
            if isinstance(brightness_range, (list, tuple)) and len(brightness_range) == 2:
                min_brightness, max_brightness = brightness_range
                # Only join if not already joined
                query = query.join(ClipColors, isouter=True)
                if min_brightness is not None:
                    query = query.filter(ClipColors.brightness_value >= min_brightness)
                if max_brightness is not None:
                    query = query.filter(ClipColors.brightness_value <= max_brightness)
                logger.debug(
                    f"Filter applied: brightness_range = [{min_brightness}, {max_brightness}]"
                )

        # Filter: temperature (requires JOIN to ClipColors)
        if "temperature" in filters and filters["temperature"]:
            temperature = filters["temperature"]
            # Ensure ClipColors is joined
            if "brightness_range" not in filters:
                query = query.join(ClipColors, isouter=True)
            if isinstance(temperature, list):
                query = query.filter(ClipColors.temperature.in_(temperature))
            else:
                query = query.filter(ClipColors.temperature == temperature)
            logger.debug(f"Filter applied: temperature = {temperature}")

        # Filter: has_face (requires JOIN to ClipSceneType)
        if "has_face" in filters and filters["has_face"] is not None:
            has_face = filters["has_face"]
            # Ensure ClipSceneType is joined
            if "scene_type" not in filters:
                query = query.join(ClipSceneType, isouter=True)
            query = query.filter(ClipSceneType.has_face == has_face)
            logger.debug(f"Filter applied: has_face = {has_face}")

        # Filter: energy_range (requires JOIN to ClipMood)
        if "energy_range" in filters and filters["energy_range"]:
            energy_range = filters["energy_range"]
            if isinstance(energy_range, (list, tuple)) and len(energy_range) == 2:
                min_energy, max_energy = energy_range
                # Ensure ClipMood is joined
                if "mood" not in filters:
                    query = query.join(ClipMood, isouter=True)
                if min_energy is not None:
                    query = query.filter(ClipMood.energy >= min_energy)
                if max_energy is not None:
                    query = query.filter(ClipMood.energy <= max_energy)
                logger.debug(f"Filter applied: energy_range = [{min_energy}, {max_energy}]")

        # Filter: is_analyzed (requires JOIN to ClipAnalysisStatus)
        if "is_analyzed" in filters and filters["is_analyzed"] is not None:
            is_analyzed = filters["is_analyzed"]
            query = query.join(ClipAnalysisStatus, isouter=True)
            if is_analyzed:
                # All analysis flags must be True
                query = query.filter(
                    # FIX H-03: SQLAlchemy Boolean Best Practice - use is_(True)
                    ClipAnalysisStatus.colors_analyzed.is_(True),
                    ClipAnalysisStatus.motion_analyzed.is_(True),
                    ClipAnalysisStatus.scene_analyzed.is_(True),
                    ClipAnalysisStatus.mood_analyzed.is_(True),
                    ClipAnalysisStatus.objects_analyzed.is_(True),
                    ClipAnalysisStatus.style_analyzed.is_(True),
                )
            else:
                # At least one analysis flag is False or status doesn't exist
                from sqlalchemy import or_

                query = query.filter(
                    or_(
                        # FIX H-03: SQLAlchemy Boolean Best Practice - use is_()
                        ClipAnalysisStatus.clip_id.is_(None),
                        ClipAnalysisStatus.colors_analyzed.is_(False),
                        ClipAnalysisStatus.motion_analyzed.is_(False),
                        ClipAnalysisStatus.scene_analyzed.is_(False),
                        ClipAnalysisStatus.mood_analyzed.is_(False),
                        ClipAnalysisStatus.objects_analyzed.is_(False),
                        ClipAnalysisStatus.style_analyzed.is_(False),
                    )
                )
            logger.debug(f"Filter applied: is_analyzed = {is_analyzed}")

        return query

    def cancel(self):
        """Cancel current operation."""
        # FIX: Use Event.set() for thread-safe cancel signaling
        self._cancel_event.set()
        logger.info("DatabaseWorker: Cancel requested")
