"""
SQLAlchemy ORM Models fuer Video-Analyse in PB_studio

Erweiterte Datenbank-Tabellen fuer:
- Farb-Analyse (ClipColors)
- Bewegungs-Analyse (ClipMotion)
- Szenentyp-Analyse (ClipSceneType)
- Stimmungs-Analyse (ClipMood)
- Objekt-Erkennung (ClipObjects)
- Style-Analyse (ClipStyle)
- Fingerprints & Hashes (ClipFingerprint)
- Analyse-Status (ClipAnalysisStatus)
"""

import json
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from .models import Base


class ClipSemantics(Base):
    """Persistierte semantische Analyse-Ergebnisse (CLIP)."""

    __tablename__ = "clip_semantics"

    clip_id = Column(Integer, ForeignKey("video_clips.id", ondelete="CASCADE"), primary_key=True)
    scene_type = Column(String(100), nullable=True)
    mood = Column(String(100), nullable=True)
    content_tags = Column(Text, nullable=True)  # JSON-Array
    raw_results = Column(Text, nullable=True)  # Vollständiges Resultat als JSON

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    clip = relationship("VideoClip", backref="semantics")

    def set_results(self, data: dict):
        self.scene_type = data.get("scene_type")
        self.mood = data.get("mood")
        try:
            self.content_tags = json.dumps(data.get("content_tags") or [])
            self.raw_results = json.dumps(data)
        except Exception:
            self.content_tags = None
            self.raw_results = None

    def get_content_tags(self) -> list[str]:
        try:
            return json.loads(self.content_tags) if self.content_tags else []
        except Exception:
            return []


class ClipColors(Base):
    """Farb-Analyse Tabelle - speichert Farbpalette und Temperatur."""

    __tablename__ = "clip_colors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # BUGFIX #10: Add index to FK column for faster joins
    clip_id = Column(
        Integer, ForeignKey("video_clips.id", ondelete="CASCADE"), nullable=False, index=True
    )
    frame_position = Column(String(20), default="middle")  # 'start', 'middle', 'end', 'average'

    # Dominante Farben (JSON)
    dominant_colors = Column(Text, nullable=True)  # [{"rgb": [r,g,b], "pct": 45.2}, ...]

    # Farbtemperatur
    temperature = Column(String(20), nullable=True)  # 'WARM', 'COOL', 'NEUTRAL'
    temperature_score = Column(Float, nullable=True)  # 0.0 - 1.0

    # Helligkeit
    brightness = Column(String(20), nullable=True)  # 'DARK', 'MEDIUM', 'BRIGHT'
    brightness_value = Column(Float, nullable=True)  # 0.0 - 1.0

    # Mood-Tags basierend auf Farben
    color_moods = Column(Text, nullable=True)  # JSON: ["DARK", "VIBRANT", "COOL"]

    # Temporal Features (Phase 2)
    brightness_dynamics = Column(Float, nullable=True)  # 0.0 - 1.0 (Helligkeits-Variation)
    color_dynamics = Column(Float, nullable=True)  # 0.0 - 1.0 (Farbton-Variation)
    temporal_rhythm = Column(String(20), nullable=True)  # 'STEADY', 'DYNAMIC', 'FLASHY'

    analyzed_at = Column(DateTime, default=datetime.utcnow)

    # QUICK-WIN #2: Optimized indexes for common query patterns
    __table_args__ = (
        Index("idx_colors_clip", "clip_id"),
        Index("idx_colors_temp", "temperature"),
        Index("idx_colors_bright", "brightness"),
        # Composite index for temperature + brightness filtering (e.g., "WARM and BRIGHT clips")
        Index("idx_colors_temp_bright", "temperature", "brightness"),
        # Phase 2 Indexes (CRITICAL für Performance!):
        Index("idx_colors_temporal_rhythm", "temporal_rhythm"),
        Index("idx_colors_bright_dyn", "brightness_dynamics"),
        Index("idx_colors_color_dyn", "color_dynamics"),
        # Composite Index für Temporal-Queries (100x schneller):
        Index(
            "idx_colors_temporal_combo", "temporal_rhythm", "brightness_dynamics", "color_dynamics"
        ),
    )

    # Relationship zurück zu VideoClip
    clip = relationship("VideoClip", back_populates="colors")

    def get_dominant_colors(self) -> list[dict]:
        """Gibt dominante Farben als Liste zurueck."""
        if not self.dominant_colors:
            return []
        try:
            return json.loads(self.dominant_colors)
        except (json.JSONDecodeError, TypeError):
            return []

    def set_dominant_colors(self, colors: list[dict]) -> None:
        """Speichert dominante Farben als JSON."""
        self.dominant_colors = json.dumps(colors)

    def get_color_moods(self) -> list[str]:
        """Gibt Farb-Moods als Liste zurueck."""
        if not self.color_moods:
            return []
        try:
            return json.loads(self.color_moods)
        except (json.JSONDecodeError, TypeError):
            return []

    def set_color_moods(self, moods: list[str]) -> None:
        """Speichert Farb-Moods als JSON."""
        self.color_moods = json.dumps(moods)


class ClipMotion(Base):
    """Bewegungs-Analyse Tabelle - speichert Motion und Kamera-Bewegung."""

    __tablename__ = "clip_motion"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # BUGFIX #10: Add index to FK column for faster joins
    clip_id = Column(
        Integer, ForeignKey("video_clips.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Content Motion
    motion_type = Column(String(20), nullable=True)  # 'STATIC', 'SLOW', 'MEDIUM', 'FAST', 'EXTREME'
    motion_score = Column(Float, nullable=True)  # Numerischer Wert
    motion_rhythm = Column(String(20), nullable=True)  # 'STEADY', 'ERRATIC'
    motion_variation = Column(Float, nullable=True)  # Standardabweichung

    # Kamera-Bewegung
    camera_motion = Column(String(20), nullable=True)  # 'STATIC_CAM', 'PAN_LEFT', 'PAN_RIGHT', etc.
    camera_magnitude = Column(Float, nullable=True)

    # Optical Flow Details
    flow_magnitude_avg = Column(Float, nullable=True)
    flow_direction_dominant = Column(Float, nullable=True)  # Grad (0-360)

    analyzed_at = Column(DateTime, default=datetime.utcnow)

    # QUICK-WIN #2: Optimized indexes for common query patterns
    __table_args__ = (
        Index("idx_motion_clip", "clip_id"),
        Index("idx_motion_type", "motion_type"),
        Index("idx_motion_camera", "camera_motion"),
        # Composite index for motion type + score filtering (e.g., "FAST clips with high motion")
        Index("idx_motion_type_score", "motion_type", "motion_score"),
    )

    # Relationship zurück zu VideoClip
    clip = relationship("VideoClip", back_populates="motion")


class ClipSceneType(Base):
    """Szenentyp-Analyse Tabelle - speichert Scene Classification."""

    __tablename__ = "clip_scene_type"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # BUGFIX #10: Add index to FK column for faster joins
    clip_id = Column(
        Integer, ForeignKey("video_clips.id", ondelete="CASCADE"), nullable=False, index=True
    )
    frame_position = Column(String(20), default="middle")

    # Multi-Label Szenentypen (JSON)
    scene_types = Column(Text, nullable=True)  # ["PORTRAIT", "BUSY"]

    # Metriken
    edge_density = Column(Float, nullable=True)
    texture_variance = Column(Float, nullable=True)
    center_ratio = Column(Float, nullable=True)
    depth_of_field = Column(Float, nullable=True)

    # Face Detection
    has_face = Column(Boolean, default=False)
    face_count = Column(Integer, default=0)
    face_size_ratio = Column(Float, nullable=True)

    # Confidence Scores (JSON)
    confidence_scores = Column(Text, nullable=True)  # {"PORTRAIT": 0.8, "BUSY": 0.6}

    analyzed_at = Column(DateTime, default=datetime.utcnow)

    # BUGFIX #11: Add composite indexes for common query patterns
    __table_args__ = (
        Index("idx_scene_clip", "clip_id"),
        Index("idx_scene_face", "has_face"),
        # Composite index for face detection + face count queries
        Index("idx_scene_face_count", "has_face", "face_count"),
    )

    # Relationship zurück zu VideoClip
    clip = relationship("VideoClip", back_populates="scene_type")

    def get_scene_types(self) -> list[str]:
        """Gibt Szenentypen als Liste zurueck."""
        if not self.scene_types:
            return []
        try:
            return json.loads(self.scene_types)
        except (json.JSONDecodeError, TypeError):
            return []

    def set_scene_types(self, types: list[str]) -> None:
        """Speichert Szenentypen als JSON."""
        self.scene_types = json.dumps(types)


class ClipMood(Base):
    """Stimmungs-Analyse Tabelle - speichert Mood/Atmosphere."""

    __tablename__ = "clip_mood"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # BUGFIX #10: Add index to FK column for faster joins
    clip_id = Column(
        Integer, ForeignKey("video_clips.id", ondelete="CASCADE"), nullable=False, index=True
    )
    frame_position = Column(String(20), default="middle")

    # Multi-Label Stimmungen (JSON)
    moods = Column(Text, nullable=True)  # ["ENERGETIC", "COOL", "DARK"]
    mood_scores = Column(Text, nullable=True)  # {"ENERGETIC": 0.75, "COOL": 0.82}

    # Basis-Metriken
    brightness = Column(Float, nullable=True)  # 0-1
    saturation = Column(Float, nullable=True)  # 0-1
    contrast = Column(Float, nullable=True)  # 0-1
    energy = Column(Float, nullable=True)  # 0-1

    # Farbtemperatur
    warm_ratio = Column(Float, nullable=True)
    cool_ratio = Column(Float, nullable=True)

    analyzed_at = Column(DateTime, default=datetime.utcnow)

    # QUICK-WIN #2: Optimized indexes for common query patterns
    __table_args__ = (
        Index("idx_mood_clip", "clip_id"),
        Index("idx_mood_energy", "energy"),
        # Composite index for energy + contrast filtering (e.g., "high energy, high contrast clips")
        Index("idx_mood_energy_contrast", "energy", "contrast"),
        # Composite index for brightness + saturation filtering (e.g., "bright and saturated clips")
        Index("idx_mood_bright_sat", "brightness", "saturation"),
    )

    # Relationship zurück zu VideoClip
    clip = relationship("VideoClip", back_populates="mood")

    def get_moods(self) -> list[str]:
        """Gibt Moods als Liste zurueck."""
        if not self.moods:
            return []
        try:
            return json.loads(self.moods)
        except (json.JSONDecodeError, TypeError):
            return []

    def set_moods(self, mood_list: list[str]) -> None:
        """Speichert Moods als JSON."""
        self.moods = json.dumps(mood_list)


class ClipObjects(Base):
    """Objekt-Erkennung Tabelle - speichert YOLO Detections und Content Tags."""

    __tablename__ = "clip_objects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # BUGFIX #10: Add index to FK column for faster joins
    clip_id = Column(
        Integer, ForeignKey("video_clips.id", ondelete="CASCADE"), nullable=False, index=True
    )
    frame_position = Column(String(20), default="middle")

    # YOLO Detections (JSON)
    detected_objects = Column(Text, nullable=True)  # ["person", "car"]
    object_counts = Column(Text, nullable=True)  # {"person": 2, "car": 1}
    confidence_scores = Column(Text, nullable=True)  # {"person": [0.92, 0.85]}

    # Feature-basierte Tags (JSON)
    content_tags = Column(Text, nullable=True)  # ["NATURE", "OUTDOOR", "GEOMETRIC"]

    # Metriken
    line_count = Column(Integer, nullable=True)
    green_ratio = Column(Float, nullable=True)
    sky_ratio = Column(Float, nullable=True)
    symmetry = Column(Float, nullable=True)

    analyzed_at = Column(DateTime, default=datetime.utcnow)

    # BUGFIX #11: Add composite indexes for common query patterns
    __table_args__ = (
        Index("idx_objects_clip", "clip_id"),
        # Composite index for nature/outdoor content queries (green_ratio + sky_ratio)
        Index("idx_objects_nature", "green_ratio", "sky_ratio"),
    )

    # Relationship zurück zu VideoClip
    clip = relationship("VideoClip", back_populates="objects")

    def get_detected_objects(self) -> list[str]:
        """Gibt erkannte Objekte als Liste zurueck."""
        if not self.detected_objects:
            return []
        try:
            return json.loads(self.detected_objects)
        except (json.JSONDecodeError, TypeError):
            return []

    def set_detected_objects(self, objects: list[str]) -> None:
        """Speichert erkannte Objekte als JSON."""
        self.detected_objects = json.dumps(objects)

    def get_content_tags(self) -> list[str]:
        """Gibt Content-Tags als Liste zurueck."""
        if not self.content_tags:
            return []
        try:
            return json.loads(self.content_tags)
        except (json.JSONDecodeError, TypeError):
            return []

    def set_content_tags(self, tags: list[str]) -> None:
        """Speichert Content-Tags als JSON."""
        self.content_tags = json.dumps(tags)


class ClipStyle(Base):
    """Visual Style Tabelle - speichert aesthetische Merkmale."""

    __tablename__ = "clip_style"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # BUGFIX #10: Add index to FK column for faster joins
    clip_id = Column(
        Integer, ForeignKey("video_clips.id", ondelete="CASCADE"), nullable=False, index=True
    )
    frame_position = Column(String(20), default="middle")

    # Style-Tags (JSON)
    styles = Column(Text, nullable=True)  # ["VINTAGE", "FILMIC", "NEON"]

    # Metriken
    unique_colors = Column(Integer, nullable=True)
    noise_level = Column(Float, nullable=True)
    sharpness = Column(Float, nullable=True)
    vignette_score = Column(Float, nullable=True)
    saturation_mean = Column(Float, nullable=True)
    saturation_std = Column(Float, nullable=True)
    dynamic_range = Column(Float, nullable=True)
    mean_brightness = Column(Float, nullable=True)

    analyzed_at = Column(DateTime, default=datetime.utcnow)

    # BUGFIX #11: Add composite indexes for common query patterns
    __table_args__ = (
        Index("idx_style_clip", "clip_id"),
        # Composite index for vintage/filmic style queries (noise + vignette)
        Index("idx_style_vintage", "noise_level", "vignette_score"),
        # Composite index for sharpness + saturation queries
        Index("idx_style_sharp_sat", "sharpness", "saturation_mean"),
    )

    # Relationship zurück zu VideoClip
    clip = relationship("VideoClip", back_populates="style")

    def get_styles(self) -> list[str]:
        """Gibt Styles als Liste zurueck."""
        if not self.styles:
            return []
        try:
            return json.loads(self.styles)
        except (json.JSONDecodeError, TypeError):
            return []

    def set_styles(self, style_list: list[str]) -> None:
        """Speichert Styles als JSON."""
        self.styles = json.dumps(style_list)


class ClipFingerprint(Base):
    """Fingerprint Tabelle - speichert Hashes fuer Wiedererkennung und Aehnlichkeit."""

    __tablename__ = "clip_fingerprints"

    clip_id = Column(Integer, ForeignKey("video_clips.id", ondelete="CASCADE"), primary_key=True)

    # Content Hash fuer Wiedererkennung (SHA256 basiert)
    content_fingerprint = Column(String(64), unique=True, nullable=True)

    # Perceptual Hashes (256-bit hex strings)
    phash = Column(String(64), nullable=True)
    dhash = Column(String(64), nullable=True)
    ahash = Column(String(64), nullable=True)

    # Feature-Vektor Referenz
    vector_file = Column(String(500), nullable=True)  # Pfad zu .npy
    faiss_index_id = Column(Integer, nullable=True)  # ID im FAISS Index

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_fp_content", "content_fingerprint"),
        Index("idx_fp_phash", "phash"),
    )

    # Relationship zurück zu VideoClip
    clip = relationship("VideoClip", back_populates="fingerprint")


class ClipAnalysisStatus(Base):
    """Analyse-Status Tabelle - trackt welche Analysen durchgefuehrt wurden."""

    __tablename__ = "clip_analysis_status"

    clip_id = Column(Integer, ForeignKey("video_clips.id", ondelete="CASCADE"), primary_key=True)

    # Analyse-Flags
    colors_analyzed = Column(Boolean, default=False)
    motion_analyzed = Column(Boolean, default=False)
    scene_analyzed = Column(Boolean, default=False)
    mood_analyzed = Column(Boolean, default=False)
    objects_analyzed = Column(Boolean, default=False)
    style_analyzed = Column(Boolean, default=False)
    fingerprint_created = Column(Boolean, default=False)
    vector_extracted = Column(Boolean, default=False)

    # Versionen (fuer Re-Analyse bei Algorithmus-Update)
    colors_version = Column(Integer, default=0)
    motion_version = Column(Integer, default=0)
    scene_version = Column(Integer, default=0)
    mood_version = Column(Integer, default=0)
    objects_version = Column(Integer, default=0)
    style_version = Column(Integer, default=0)

    # Timestamps
    last_full_analysis = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship zurück zu VideoClip
    clip = relationship("VideoClip", back_populates="analysis_status")

    def is_fully_analyzed(self) -> bool:
        """Prueft ob alle Analysen durchgefuehrt wurden."""
        return all(
            [
                self.colors_analyzed,
                self.motion_analyzed,
                self.scene_analyzed,
                self.mood_analyzed,
                self.objects_analyzed,
                self.style_analyzed,
                self.fingerprint_created,
                self.vector_extracted,
            ]
        )

    def get_missing_analyses(self) -> list[str]:
        """Gibt Liste der fehlenden Analysen zurueck."""
        missing = []
        if not self.colors_analyzed:
            missing.append("colors")
        if not self.motion_analyzed:
            missing.append("motion")
        if not self.scene_analyzed:
            missing.append("scene")
        if not self.mood_analyzed:
            missing.append("mood")
        if not self.objects_analyzed:
            missing.append("objects")
        if not self.style_analyzed:
            missing.append("style")
        if not self.fingerprint_created:
            missing.append("fingerprint")
        if not self.vector_extracted:
            missing.append("vector")
        return missing


# Erweitere VideoClip Model um neue Felder
# Dies wird als Mixin hinzugefuegt, damit existierende Daten erhalten bleiben


class VideoClipAnalysisExtension:
    """
    Erweiterungsfelder fuer VideoClip.
    Diese muessen zur bestehenden VideoClip Klasse hinzugefuegt werden.
    """

    # Content Fingerprint fuer Wiedererkennung
    content_fingerprint = Column(String(64), unique=True, nullable=True)

    # Pfad-Tracking
    original_path = Column(String(500), nullable=True)  # Pfad beim ersten Import

    # Verfuegbarkeit
    is_available = Column(Boolean, default=True)  # Gerade erreichbar?
    last_seen_at = Column(DateTime, nullable=True)  # Wann zuletzt gesehen

    # Analyse-Status (Quick-Check)
    needs_reanalysis = Column(Boolean, default=True)


# Liste aller neuen Tabellen fuer Migration
ANALYSIS_TABLES = [
    ClipColors,
    ClipMotion,
    ClipSceneType,
    ClipMood,
    ClipObjects,
    ClipStyle,
    ClipFingerprint,
    ClipAnalysisStatus,
    ClipSemantics,
]
