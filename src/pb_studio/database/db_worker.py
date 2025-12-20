"""
Database Worker Thread für PB_studio

Führt alle Datenbank-Operationen in Background Thread aus,
um UI non-blocking zu halten.

Author: PB_studio Development Team
"""

from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from sqlalchemy.orm import Session, joinedload

from ..utils.clip_data_loader import ClipDataLoader
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
        self._cancelled = False

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
            self._cancelled = True
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
            self._cancelled = False

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
                .filter(VideoClip.is_available == True)
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

            # PERF-OPTIMIZATION: Use ClipDataLoader for lazy loading of heavy JSON fields
            for i, clip_db in enumerate(db_clips):
                if self._cancelled:
                    logger.info("DatabaseWorker: Query cancelled by user")
                    return

                # Transform to dict using lightweight loader
                # full_details=False prevents parsing of large JSON fields used only in details view
                clip_data = ClipDataLoader.db_to_dict(clip_db, full_details=False)

                # Track unanalyzed count
                if clip_data.get("_unanalyzed_count", 0) > 0:
                    unanalyzed_count += 1

                clips_data.append(clip_data)

                # Progress update every 20 clips
                if (i + 1) % 20 == 0 or (i + 1) == total:
                    progress = int(30 + (i / total) * 70)  # 30% to 100%
                    self.progress_update.emit(progress, 100, f"Processing clip {i+1}/{total}...")

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

        # Filter: project_id (direct on VideoClip)
        if "project_id" in filters and filters["project_id"] is not None:
            query = query.filter(VideoClip.project_id == filters["project_id"])
            logger.debug(f"Filter applied: project_id = {filters['project_id']}")

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
                    ClipAnalysisStatus.colors_analyzed == True,
                    ClipAnalysisStatus.motion_analyzed == True,
                    ClipAnalysisStatus.scene_analyzed == True,
                    ClipAnalysisStatus.mood_analyzed == True,
                    ClipAnalysisStatus.objects_analyzed == True,
                    ClipAnalysisStatus.style_analyzed == True,
                )
            else:
                # At least one analysis flag is False or status doesn't exist
                from sqlalchemy import or_

                query = query.filter(
                    or_(
                        ClipAnalysisStatus.clip_id == None,
                        ClipAnalysisStatus.colors_analyzed == False,
                        ClipAnalysisStatus.motion_analyzed == False,
                        ClipAnalysisStatus.scene_analyzed == False,
                        ClipAnalysisStatus.mood_analyzed == False,
                        ClipAnalysisStatus.objects_analyzed == False,
                        ClipAnalysisStatus.style_analyzed == False,
                    )
                )
            logger.debug(f"Filter applied: is_analyzed = {is_analyzed}")

        return query

    def cancel(self):
        """Cancel current operation."""
        logger.info("DatabaseWorker: Cancel requested")
        self._cancelled = True
