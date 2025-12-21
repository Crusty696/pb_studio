"""
VideoAnalyzer - Haupt-Klasse fuer Video-Analyse in PB_studio

Koordiniert alle Analyse-Module und speichert Ergebnisse in der Datenbank.
"""

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..database.connection import get_db_manager
from ..utils.logger import get_logger

logger = get_logger()

# imagehash für Visual Hash Berechnung
try:
    import imagehash
    from PIL import Image

    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False
    logger.warning(
        "imagehash not installed - visual hashing disabled. Install with: pip install imagehash Pillow"
    )


class VideoAnalyzer:
    """
    Zentrale Klasse fuer Video-Analyse.

    Koordiniert:
    - Farb-Analyse
    - Bewegungs-Analyse
    - Szenentyp-Analyse
    - Mood-Analyse
    - Object Detection (YOLO)
    - Style-Analyse
    - Feature-Extraktion

    Usage:
        analyzer = VideoAnalyzer()
        results = analyzer.analyze_clip(clip_id)
        analyzer.batch_analyze([1, 2, 3], progress_callback=my_callback)
    """

    # Aktuelle Algorithmus-Versionen (fuer Re-Analyse bei Updates)
    VERSIONS = {
        "colors": 1,
        "motion": 1,
        "scene": 1,
        "mood": 1,
        "objects": 1,
        "style": 1,
    }

    def __init__(
        self, db_path: str | None = None, enable_yolo: bool = True, enable_motion: bool = True
    ):
        """
        Initialisiert den VideoAnalyzer.

        Args:
            db_path: Optionaler Pfad zur Datenbank
            enable_yolo: YOLO Object Detection aktivieren (default: True)
            enable_motion: Motion-Analyse aktivieren (default: True - jetzt optimiert!)
        """
        self.db_manager = get_db_manager()
        self.enable_yolo = enable_yolo
        self.enable_motion = enable_motion

        # Lazy-Load der Analyzer (erst bei Bedarf)
        self._color_analyzer = None
        self._motion_analyzer = None
        self._scene_analyzer = None
        self._mood_analyzer = None
        self._object_detector = None
        self._style_analyzer = None
        self._feature_extractor = None
        self._semantic_analyzer = None
        self._video_intelligence_engine = None

        # Cache dir for embeddings
        self.embedding_dir = Path("video_cache/embeddings")
        self.embedding_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"VideoAnalyzer initialisiert (YOLO: {enable_yolo}, Motion: {enable_motion})")

    # ==========================================
    # Lazy-Loading der Analyzer
    # ==========================================

    @property
    def color_analyzer(self):
        """Lazy-Load ColorAnalyzer."""
        if self._color_analyzer is None:
            from .analyzers.color_analyzer import ColorAnalyzer

            self._color_analyzer = ColorAnalyzer()
        return self._color_analyzer

    @property
    def motion_analyzer(self):
        """Lazy-Load MotionAnalyzer (optimized: 5 samples @ 160px)."""
        if self._motion_analyzer is None:
            from .analyzers.motion_analyzer import MotionAnalyzer

            self._motion_analyzer = MotionAnalyzer(sample_frames=5, resize_width=160)
        return self._motion_analyzer

    @property
    def scene_analyzer(self):
        """Lazy-Load SceneAnalyzer."""
        if self._scene_analyzer is None:
            from .analyzers.scene_analyzer import SceneAnalyzer

            self._scene_analyzer = SceneAnalyzer()
        return self._scene_analyzer

    @property
    def mood_analyzer(self):
        """Lazy-Load MoodAnalyzer."""
        if self._mood_analyzer is None:
            from .analyzers.mood_analyzer import MoodAnalyzer

            self._mood_analyzer = MoodAnalyzer()
        return self._mood_analyzer

    @property
    def object_detector(self):
        """Lazy-Load ObjectDetector."""
        if self._object_detector is None:
            from .analyzers.object_detector import ObjectDetector

            self._object_detector = ObjectDetector(enabled=self.enable_yolo)
        return self._object_detector

    @property
    def style_analyzer(self):
        """Lazy-Load StyleAnalyzer."""
        if self._style_analyzer is None:
            from .analyzers.style_analyzer import StyleAnalyzer

            self._style_analyzer = StyleAnalyzer()
        return self._style_analyzer

    @property
    def feature_extractor(self):
        """Lazy-Load FeatureExtractor."""
        if self._feature_extractor is None:
            from .analyzers.feature_extractor import FeatureExtractor

            self._feature_extractor = FeatureExtractor()
        return self._feature_extractor

    @property
    def semantic_analyzer(self):
        """Lazy-Load SemanticAnalyzer."""
        if self._semantic_analyzer is None:
            from .analyzers.semantic_analyzer import SemanticAnalyzer

            self._semantic_analyzer = SemanticAnalyzer()
        return self._semantic_analyzer

    @property
    def video_intelligence_engine(self):
        """Lazy-Load VideoIntelligenceEngine für Auto-Tagging."""
        if self._video_intelligence_engine is None:
            from .analyzers.video_intelligence_engine import VideoIntelligenceEngine

            self._video_intelligence_engine = VideoIntelligenceEngine(
                confidence_threshold=0.6, enable_traditional_cv=True
            )
        return self._video_intelligence_engine

    def unload_models(self):
        """
        Entlaedt alle ML-Models aus GPU Memory.

        Rufe diese Methode auf nach:
        - Batch-Verarbeitung von vielen Clips
        - Vor Wechsel zu anderen GPU-intensiven Tasks
        - Bei Low-Memory-Warnungen
        """
        models_unloaded = []

        # CLIP (semantic_analyzer)
        if self._semantic_analyzer is not None:
            if hasattr(self._semantic_analyzer, "_model") and self._semantic_analyzer._model:
                del self._semantic_analyzer._model
            if (
                hasattr(self._semantic_analyzer, "_processor")
                and self._semantic_analyzer._processor
            ):
                del self._semantic_analyzer._processor
            self._semantic_analyzer = None
            models_unloaded.append("CLIP")

        # YOLO (object_detector)
        if self._object_detector is not None:
            if hasattr(self._object_detector, "_model") and self._object_detector._model:
                del self._object_detector._model
            self._object_detector = None
            models_unloaded.append("YOLO")

        # ResNet (feature_extractor)
        if self._feature_extractor is not None:
            if hasattr(self._feature_extractor, "_model") and self._feature_extractor._model:
                del self._feature_extractor._model
            self._feature_extractor = None
            models_unloaded.append("ResNet")

        # Style Analyzer
        if self._style_analyzer is not None:
            self._style_analyzer = None
            models_unloaded.append("StyleAnalyzer")

        # Motion Analyzer
        if self._motion_analyzer is not None:
            self._motion_analyzer = None
            models_unloaded.append("MotionAnalyzer")

        # Video Intelligence Engine
        if self._video_intelligence_engine is not None:
            # Cleanup CLIP models from VideoIntelligenceEngine
            if hasattr(self._video_intelligence_engine, "semantic_analyzer"):
                semantic = self._video_intelligence_engine.semantic_analyzer
                if hasattr(semantic, "_model") and semantic._model:
                    del semantic._model
                if hasattr(semantic, "_processor") and semantic._processor:
                    del semantic._processor
            self._video_intelligence_engine = None
            models_unloaded.append("VideoIntelligenceEngine")

        # GPU Memory freigeben
        if models_unloaded:
            from ..utils.gpu_memory import clear_gpu_memory

            clear_gpu_memory()
            logger.info(f"Models entladen: {', '.join(models_unloaded)}")
        else:
            logger.debug("Keine Models zu entladen")

    # ==========================================
    # Video Handling
    # ==========================================

    def open_video(self, video_path: str) -> cv2.VideoCapture | None:
        """
        Oeffnet ein Video und gibt VideoCapture zurueck.

        Args:
            video_path: Pfad zum Video

        Returns:
            cv2.VideoCapture oder None bei Fehler
        """
        path = Path(video_path)
        if not path.exists():
            logger.warning(f"Video nicht gefunden: {video_path}")
            return None

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()  # PERF-02 FIX: Release even on failed open
            logger.warning(f"Video konnte nicht geoeffnet werden: {video_path}")
            return None

        return cap

    def get_video_info(self, cap: cv2.VideoCapture) -> dict[str, Any]:
        """Gibt Video-Metadaten zurueck."""
        return {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1),
        }

    def get_sample_frames(
        self, cap: cv2.VideoCapture, positions: list[str] = None
    ) -> dict[str, np.ndarray]:
        """
        Liest Sample-Frames aus dem Video.

        Args:
            cap: VideoCapture Objekt
            positions: Liste von Positionen ('start', 'middle', 'end') oder None fuer alle

        Returns:
            Dict mit Position -> Frame Mapping
        """
        if positions is None:
            positions = ["start", "middle", "end"]

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 3:
            total_frames = 3

        position_map = {
            "start": 0,
            "middle": total_frames // 2,
            "end": max(0, total_frames - 5),
        }

        frames = {}
        for pos in positions:
            frame_idx = position_map.get(pos, total_frames // 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frames[pos] = frame

        return frames

    # ==========================================
    # Einzel-Analyse Methoden
    # ==========================================

    def analyze_colors(self, frame: np.ndarray) -> dict:
        """Analysiert Farben eines Frames."""
        result = self.color_analyzer.analyze(frame)
        return result.to_dict() if hasattr(result, "to_dict") else result

    def analyze_motion(self, video_path: str) -> dict:
        """Analysiert Bewegung im Video."""
        result = self.motion_analyzer.analyze(video_path)
        return result.to_dict() if hasattr(result, "to_dict") else result

    def analyze_scene_type(self, frame: np.ndarray) -> dict:
        """Analysiert Szenentyp eines Frames."""
        result = self.scene_analyzer.analyze(frame)
        return result.to_dict() if hasattr(result, "to_dict") else result

    def analyze_mood(self, frame: np.ndarray) -> dict:
        """Analysiert Stimmung eines Frames."""
        result = self.mood_analyzer.analyze(frame)
        return result.to_dict() if hasattr(result, "to_dict") else result

    def analyze_objects(self, frame: np.ndarray) -> dict:
        """Erkennt Objekte in einem Frame (YOLO + Feature-basiert)."""
        result = self.object_detector.analyze(frame)
        return result.to_dict() if hasattr(result, "to_dict") else result

    def analyze_style(self, frame: np.ndarray) -> dict:
        """Analysiert visuellen Style eines Frames."""
        result = self.style_analyzer.analyze(frame)
        return result.to_dict() if hasattr(result, "to_dict") else result

    def extract_feature_vector(self, frame: np.ndarray) -> np.ndarray | None:
        """Extrahiert Feature-Vektor fuer FAISS."""
        result = self.feature_extractor.extract(frame)
        if result is not None:
            return result.vector
        return None

    def analyze_semantic(self, frame: np.ndarray, labels: list[str] | None = None) -> dict:
        """
        Semantische Analyse mit CLIP.
        Prueft das Bild auf bestimmte Begriffe.
        """
        if labels is None:
            # Default labels relevante fuer Video-Editing
            labels = [
                "party",
                "crowd",
                "dancing",
                "dj",
                "stage",
                "lights",
                "nature",
                "beach",
                "forest",
                "sky",
                "city",
                "street",
                "indoor",
                "outdoor",
                "day",
                "night",
                "calm",
                "action",
                "colorful",
                "dark",
            ]

        return self.semantic_analyzer.analyze(frame, labels)

    def extract_semantic_embedding(self, frame: np.ndarray) -> list[float] | None:
        """
        Extrahiert semantisches CLIP-Embedding.
        """
        return self.semantic_analyzer.get_embedding(frame)

    def analyze_video_tags(self, video_path: str, custom_labels: list[str] | None = None) -> dict:
        """
        Auto-Tagging für Videos mit VideoIntelligenceEngine.

        NEW OVERNIGHT DEV FEATURE: Intelligente Video-Tag-Generierung mit CLIP + Traditional CV

        Args:
            video_path: Pfad zur Video-Datei
            custom_labels: Optionale custom Labels für spezifische Use Cases

        Returns:
            Dictionary mit Tags, Confidence-Scores und Metadaten:
            {
                "tags": ["beach", "ocean", "sunset"],
                "primary_scene": "beach",
                "confidence": 0.85,
                "quality_score": 0.78,
                "frame_count": 5,
                "all_scores": {"beach": 0.85, "ocean": 0.72, ...}
            }
        """
        try:
            return self.video_intelligence_engine.get_video_tags(
                video_path=video_path,
                confidence_threshold=None,  # Use engine default
            )
        except Exception as e:
            logger.error(f"Auto-tagging failed for {video_path}: {e}")
            return {
                "tags": ["unknown"],
                "primary_scene": "unknown",
                "confidence": 0.0,
                "quality_score": 0.0,
                "frame_count": 0,
                "error": str(e),
            }

    def analyze_frame_scene_recognition(
        self, frame: np.ndarray, custom_labels: list[str] | None = None
    ) -> dict:
        """
        CLIP-basierte Szenen-Erkennung für einzelne Frames.

        NEW OVERNIGHT DEV FEATURE: Enhanced Scene Recognition mit Traditional CV

        Args:
            frame: Video-Frame als numpy array
            custom_labels: Optionale Labels (default: VideoIntelligenceEngine.DEFAULT_SCENE_LABELS)

        Returns:
            Dictionary mit Scene Recognition Ergebnis:
            {
                "scene_labels": {"beach": 0.85, "ocean": 0.72, ...},
                "primary_scene": "beach",
                "confidence": 0.85,
                "combined_tags": ["beach", "ocean", "landscape"],
                "quality_score": 0.78
            }
        """
        try:
            result = self.video_intelligence_engine.analyze_frame(frame, custom_labels)
            return result.to_dict()
        except Exception as e:
            logger.error(f"Frame scene recognition failed: {e}")
            return {
                "scene_labels": {},
                "primary_scene": "unknown",
                "confidence": 0.0,
                "combined_tags": ["unknown"],
                "quality_score": 0.0,
                "error": str(e),
            }

    # ==========================================
    # Visual Hash Berechnung
    # ==========================================

    def _calculate_visual_hash(self, frame: np.ndarray) -> str | None:
        """
        Berechne perceptual hash (pHash) für Frame.

        pHash ist robust gegen:
        - Leichte Skalierung
        - Kompression
        - Helligkeit/Kontrast-Änderungen

        Args:
            frame: Frame als numpy array (BGR)

        Returns:
            Hex-String des pHash oder None bei Fehler
        """
        if not IMAGEHASH_AVAILABLE:
            return None

        try:
            # Konvertiere zu PIL Image
            if len(frame.shape) == 3:
                # BGR zu RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb = frame

            pil_image = Image.fromarray(rgb)

            # Berechne pHash (16x16, robust gegen Skalierung/Kompression)
            phash = imagehash.phash(pil_image, hash_size=16)

            return str(phash)
        except Exception as e:
            logger.warning(f"pHash calculation failed: {e}")
            return None

    def _calculate_dhash(self, frame: np.ndarray) -> str | None:
        """
        Berechne difference hash (dHash) für Frame.

        dHash ist schneller als pHash und gut für:
        - Duplikat-Erkennung
        - Ähnliche Frames
        - Schnelle Vergleiche

        Args:
            frame: Frame als numpy array (BGR)

        Returns:
            Hex-String des dHash oder None bei Fehler
        """
        if not IMAGEHASH_AVAILABLE:
            return None

        try:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            dhash = imagehash.dhash(pil_image, hash_size=8)
            return str(dhash)
        except Exception as e:
            logger.warning(f"dHash calculation failed: {e}")
            return None

    def find_similar_frames(self, clip_id: int, phash: str, threshold: int = 10) -> list[tuple]:
        """
        Finde ähnliche Frames basierend auf pHash.

        Args:
            clip_id: ID des zu vergleichenden Clips
            phash: Perceptual Hash als Hex-String
            threshold: Maximale Hamming-Distanz (default: 10)

        Returns:
            Liste von (clip_id, distance) Tupeln, sortiert nach Distanz
        """
        if not IMAGEHASH_AVAILABLE:
            logger.warning("imagehash not available - cannot find similar frames")
            return []

        session = self.db_manager.get_session()
        try:
            from ..database.models_analysis import ClipFingerprint

            # Hole alle Fingerprints
            fingerprints = (
                session.query(ClipFingerprint).filter(ClipFingerprint.phash.isnot(None)).all()
            )

            similar = []
            target_hash = imagehash.hex_to_hash(phash)

            for fp in fingerprints:
                if fp.clip_id == clip_id:
                    continue  # Skip self

                try:
                    stored_hash = imagehash.hex_to_hash(fp.phash)
                    distance = target_hash - stored_hash
                    if distance <= threshold:
                        similar.append((fp.clip_id, distance))
                except Exception as e:
                    logger.debug(f"Error comparing hash for clip {fp.clip_id}: {e}")
                    continue

            return sorted(similar, key=lambda x: x[1])

        finally:
            session.close()

    # ==========================================
    # Haupt-Analyse Methoden
    # ==========================================

    def _safe_analyze(
        self,
        func: Callable,
        *args,
        error_key: str = "unknown",
        clip_id: int | None = None,
        **kwargs,
    ) -> dict:
        """
        Fuehrt Analyse-Funktion mit Error-Isolation aus.

        Args:
            func: Analyse-Funktion (z.B. self.analyze_colors)
            *args: Positionale Argumente fuer func
            error_key: Name der Analyse fuer Logging
            clip_id: Clip-ID fuer Logging (optional)
            **kwargs: Keyword-Argumente fuer func

        Returns:
            Analyse-Resultat (Dict) oder {'error': str} bei Fehler
        """
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            clip_info = f"Clip {clip_id}: " if clip_id else ""
            logger.warning(f"{clip_info}{error_key} fehlgeschlagen: {e}")
            return {"error": str(e)}

    def analyze_clip(self, clip_id: int, save_to_db: bool = True) -> dict:
        """
        Fuehrt vollstaendige Analyse eines Clips durch.

        Args:
            clip_id: ID des Clips in der Datenbank
            save_to_db: Ergebnisse in DB speichern

        Returns:
            Dict mit allen Analyse-Ergebnissen
        """
        session = self.db_manager.get_session()

        try:
            # Clip aus DB laden
            from ..database.models import VideoClip

            clip = session.query(VideoClip).filter(VideoClip.id == clip_id).first()

            if not clip:
                logger.error(f"Clip {clip_id} nicht gefunden")
                return {"error": f"Clip {clip_id} nicht gefunden"}

            video_path = clip.file_path

            # Video oeffnen
            cap = self.open_video(video_path)
            if cap is None:
                # Clip als nicht verfuegbar markieren
                clip.is_available = False
                session.commit()
                return {"error": f"Video nicht erreichbar: {video_path}"}

            # Clip als verfuegbar markieren
            clip.is_available = True
            clip.last_seen_at = datetime.utcnow()

            # Video Info
            video_info = self.get_video_info(cap)

            # Sample Frames holen
            frames = self.get_sample_frames(cap, ["middle"])
            middle_frame = frames.get("middle")

            if middle_frame is None:
                cap.release()
                return {"error": "Konnte keine Frames lesen"}

            # Alle Analysen durchfuehren (mit Fehler-Isolation)
            results = {
                "clip_id": clip_id,
                "video_info": video_info,
            }

            # Farb-Analyse (Frame-basiert)
            results["colors"] = self._safe_analyze(
                self.analyze_colors, middle_frame, error_key="Farb-Analyse", clip_id=clip_id
            )

            # Temporal Features (Video-basiert)
            temporal_features = self._safe_analyze(
                self.color_analyzer.analyze_temporal_features,
                str(video_path),
                error_key="Temporal-Features",
                clip_id=clip_id,
            )
            if temporal_features and "error" not in temporal_features:
                # Merge in colors result
                if "colors" in results and "error" not in results["colors"]:
                    results["colors"].update(temporal_features)

            # Scene-Analyse
            results["scene"] = self._safe_analyze(
                self.analyze_scene_type, middle_frame, error_key="Scene-Analyse", clip_id=clip_id
            )

            # Mood-Analyse
            results["mood"] = self._safe_analyze(
                self.analyze_mood, middle_frame, error_key="Mood-Analyse", clip_id=clip_id
            )

            # Object-Detection
            results["objects"] = self._safe_analyze(
                self.analyze_objects, middle_frame, error_key="Object-Detection", clip_id=clip_id
            )

            # Style-Analyse
            results["style"] = self._safe_analyze(
                self.analyze_style, middle_frame, error_key="Style-Analyse", clip_id=clip_id
            )

            # Feature-Extraktion
            results["feature_vector"] = self._safe_analyze(
                self.extract_feature_vector,
                middle_frame,
                error_key="Feature-Extraktion",
                clip_id=clip_id,
            )

            # Semantische Analyse
            results["semantic"] = self._safe_analyze(
                self.analyze_semantic,
                middle_frame,
                error_key="Semantische Analyse",
                clip_id=clip_id,
            )

            # Embedding-Extraktion
            results["semantic_embedding"] = self._safe_analyze(
                self.extract_semantic_embedding,
                middle_frame,
                error_key="Embedding-Extraktion",
                clip_id=clip_id,
            )

            # Save embedding to file
            if results["semantic_embedding"]:
                embedding_path = self.embedding_dir / f"{clip_id}.npy"
                np.save(str(embedding_path), np.array(results["semantic_embedding"]))
                results["embedding_path"] = str(embedding_path)

            # Visual Hashes berechnen
            results["phash"] = self._safe_analyze(
                self._calculate_visual_hash,
                middle_frame,
                error_key="pHash-Berechnung",
                clip_id=clip_id,
            )
            results["dhash"] = self._safe_analyze(
                self._calculate_dhash, middle_frame, error_key="dHash-Berechnung", clip_id=clip_id
            )

            # Merge semantic tags into object tags (so they get saved to DB)
            # BUG FIX: Check for 'error' key to avoid comparing string with float
            if (
                results.get("semantic")
                and results.get("objects")
                and "error" not in results["semantic"]
                and "error" not in results["objects"]
            ):
                # Filter high probability tags (> 0.2)
                semantic_tags = [
                    k
                    for k, v in results["semantic"].items()
                    if isinstance(v, (int, float)) and v > 0.2
                ]
                
                # Ensure objects result has content_tags list
                if "content_tags" not in results["objects"] or results["objects"]["content_tags"] is None:
                    results["objects"]["content_tags"] = []
                    
                # Combine unique
                current_tags = results["objects"]["content_tags"]
                if isinstance(current_tags, list):
                    results["objects"]["content_tags"] = list(set(current_tags + semantic_tags))

            # Motion braucht den Video-Pfad als String (nicht das cap Objekt)
            # Release cap before motion analysis to free resources, as MotionAnalyzer opens its own capture
            if cap:
                cap.release()
                cap = None

            if self.enable_motion:
                results["motion"] = self.analyze_motion(str(video_path))
            else:
                results["motion"] = {"motion_type": "UNKNOWN", "skipped": True}

            # In DB speichern
            if save_to_db:
                self._save_analysis_results(session, clip_id, results)
                clip.needs_reanalysis = False
                session.commit()

            logger.info(f"Clip {clip_id} analysiert: {clip.name}")
            return results

        except Exception as e:
            logger.error(f"Fehler bei Analyse von Clip {clip_id}: {e}")
            import traceback

            traceback.print_exc()
            session.rollback()
            return {"error": str(e)}

        finally:
            # Safe cleanup of resources
            if "cap" in locals() and cap is not None:
                cap.release()

            session.close()
            # GPU Memory Cleanup
            from ..utils.gpu_memory import clear_gpu_memory

            clear_gpu_memory()

    def _save_analysis_results(self, session, clip_id: int, results: dict) -> None:
        """Speichert Analyse-Ergebnisse in der Datenbank."""
        from ..database.models_analysis import (
            ClipAnalysisStatus,
            ClipColors,
            ClipFingerprint,
            ClipMood,
            ClipMotion,
            ClipObjects,
            ClipSceneType,
            ClipStyle,
        )

        # Loesche alte Eintraege
        session.query(ClipColors).filter(ClipColors.clip_id == clip_id).delete()
        session.query(ClipMotion).filter(ClipMotion.clip_id == clip_id).delete()
        session.query(ClipSceneType).filter(ClipSceneType.clip_id == clip_id).delete()
        session.query(ClipMood).filter(ClipMood.clip_id == clip_id).delete()
        session.query(ClipObjects).filter(ClipObjects.clip_id == clip_id).delete()
        session.query(ClipObjects).filter(ClipObjects.clip_id == clip_id).delete()
        session.query(ClipStyle).filter(ClipStyle.clip_id == clip_id).delete()
        session.query(ClipFingerprint).filter(ClipFingerprint.clip_id == clip_id).delete()

        # Farb-Analyse speichern
        if results.get("colors") and "error" not in results["colors"]:
            colors = results["colors"]
            color_entry = ClipColors(
                clip_id=clip_id,
                frame_position="middle",
                dominant_colors=json.dumps(colors.get("dominant_colors", [])),
                temperature=colors.get("temperature"),
                temperature_score=colors.get("temperature_score"),
                brightness=colors.get("brightness"),
                brightness_value=colors.get("brightness_value"),
                color_moods=json.dumps(colors.get("color_moods", [])),
                # Temporal Features (Phase 2)
                brightness_dynamics=colors.get("brightness_dynamics"),
                color_dynamics=colors.get("color_dynamics"),
                temporal_rhythm=colors.get("temporal_rhythm"),
            )
            session.add(color_entry)

        # Motion-Analyse speichern
        if results.get("motion") and "error" not in results["motion"]:
            motion = results["motion"]
            motion_entry = ClipMotion(
                clip_id=clip_id,
                motion_type=motion.get("motion_type"),
                motion_score=motion.get("motion_score"),
                motion_rhythm=motion.get("motion_rhythm"),
                motion_variation=motion.get("motion_variation"),
                camera_motion=motion.get("camera_motion"),
                camera_magnitude=motion.get("camera_magnitude"),
            )
            session.add(motion_entry)

        # Scene-Analyse speichern
        if results.get("scene") and "error" not in results["scene"]:
            scene = results["scene"]
            scene_entry = ClipSceneType(
                clip_id=clip_id,
                frame_position="middle",
                scene_types=json.dumps(scene.get("types", [])),
                edge_density=scene.get("metrics", {}).get("edge_density"),
                texture_variance=scene.get("metrics", {}).get("texture_variance"),
                has_face=scene.get("has_face", False),
                face_count=scene.get("face_count", 0),
                confidence_scores=json.dumps(scene.get("confidence", {})),
            )
            session.add(scene_entry)

        # Mood-Analyse speichern
        if results.get("mood") and "error" not in results["mood"]:
            mood = results["mood"]
            mood_entry = ClipMood(
                clip_id=clip_id,
                frame_position="middle",
                moods=json.dumps(mood.get("moods", [])),
                mood_scores=json.dumps(mood.get("scores", {})),
                brightness=mood.get("metrics", {}).get("brightness"),
                saturation=mood.get("metrics", {}).get("saturation"),
                contrast=mood.get("metrics", {}).get("contrast"),
                energy=mood.get("metrics", {}).get("energy"),
                warm_ratio=mood.get("metrics", {}).get("warm_ratio"),
                cool_ratio=mood.get("metrics", {}).get("cool_ratio"),
            )
            session.add(mood_entry)

        # Objects-Analyse speichern
        if results.get("objects") and "error" not in results["objects"]:
            obj = results["objects"]
            obj_entry = ClipObjects(
                clip_id=clip_id,
                frame_position="middle",
                detected_objects=json.dumps(obj.get("detected_objects", [])),
                object_counts=json.dumps(obj.get("object_counts", {})),
                content_tags=json.dumps(obj.get("content_tags", [])),
                line_count=obj.get("metrics", {}).get("line_count"),
                green_ratio=obj.get("metrics", {}).get("green_ratio"),
                sky_ratio=obj.get("metrics", {}).get("sky_ratio"),
                symmetry=obj.get("metrics", {}).get("symmetry"),
            )
            session.add(obj_entry)

        # Style-Analyse speichern
        if results.get("style") and "error" not in results["style"]:
            style = results["style"]
            style_entry = ClipStyle(
                clip_id=clip_id,
                frame_position="middle",
                styles=json.dumps(style.get("styles", [])),
                unique_colors=style.get("metrics", {}).get("unique_colors"),
                noise_level=style.get("metrics", {}).get("noise_level"),
                sharpness=style.get("metrics", {}).get("sharpness"),
                vignette_score=style.get("metrics", {}).get("vignette"),
                saturation_mean=style.get("metrics", {}).get("saturation"),
                dynamic_range=style.get("metrics", {}).get("dynamic_range"),
            )
            session.add(style_entry)

        # Fingerprint/Embedding speichern
        if "embedding_path" in results or "phash" in results or "dhash" in results:
            fingerprint_entry = ClipFingerprint(
                clip_id=clip_id,
                vector_file=results.get("embedding_path"),
                phash=results.get("phash")
                if results.get("phash") and "error" not in results["phash"]
                else None,
                dhash=results.get("dhash")
                if results.get("dhash") and "error" not in results["dhash"]
                else None,
            )
            session.add(fingerprint_entry)

        # Analyse-Status aktualisieren
        status = (
            session.query(ClipAnalysisStatus).filter(ClipAnalysisStatus.clip_id == clip_id).first()
        )

        if status is None:
            status = ClipAnalysisStatus(clip_id=clip_id)
            session.add(status)

        status.colors_analyzed = "colors" in results
        status.motion_analyzed = "motion" in results
        status.scene_analyzed = "scene" in results
        status.mood_analyzed = "mood" in results
        status.objects_analyzed = "objects" in results
        status.style_analyzed = "style" in results
        status.fingerprint_created = "phash" in results or "dhash" in results
        status.vector_extracted = "embedding_path" in results or "feature_vector" in results
        status.last_full_analysis = datetime.utcnow()
        status.colors_version = self.VERSIONS["colors"]
        status.motion_version = self.VERSIONS["motion"]
        status.scene_version = self.VERSIONS["scene"]
        status.mood_version = self.VERSIONS["mood"]
        status.objects_version = self.VERSIONS["objects"]
        status.style_version = self.VERSIONS["style"]

    def batch_analyze(
        self,
        clip_ids: list[int],
        progress_callback: Callable[[int, int, str], None] | None = None,
        stop_flag: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """
        Analysiert mehrere Clips im Batch.

        Args:
            clip_ids: Liste der Clip-IDs
            progress_callback: Callback(current, total, clip_name)
            stop_flag: Callable das True zurueckgibt wenn abgebrochen werden soll

        Returns:
            Dict mit Statistiken
        """
        total = len(clip_ids)
        success = 0
        errors = []

        logger.info(f"Starte Batch-Analyse von {total} Clips...")

        for i, clip_id in enumerate(clip_ids):
            # Abbruch pruefen
            if stop_flag and stop_flag():
                logger.info("Batch-Analyse abgebrochen")
                break

            # Progress Callback
            if progress_callback:
                session = self.db_manager.get_session()
                from ..database.models import VideoClip

                clip = session.query(VideoClip).filter(VideoClip.id == clip_id).first()
                clip_name = clip.name if clip else f"Clip {clip_id}"
                session.close()
                progress_callback(i + 1, total, clip_name)

            # Analysieren
            result = self.analyze_clip(clip_id)

            if "error" in result:
                errors.append({"clip_id": clip_id, "error": result["error"]})
            else:
                success += 1

            # GPU Memory freigeben alle 10 Clips
            if (i + 1) % 10 == 0:
                self.unload_models()
                logger.debug(f"Models nach {i + 1} Clips entladen")

        # Final cleanup
        self.unload_models()

        stats = {
            "total": total,
            "success": success,
            "errors": len(errors),
            "error_details": errors,
        }

        logger.info(f"Batch-Analyse abgeschlossen: {success}/{total} erfolgreich")
        return stats

    def get_unanalyzed_clips(self) -> list[int]:
        """Gibt Liste der nicht analysierten Clip-IDs zurueck."""
        session = self.db_manager.get_session()
        try:
            from ..database.models import VideoClip

            clips = session.query(VideoClip.id).filter(VideoClip.needs_reanalysis == True).all()
            return [c[0] for c in clips]
        finally:
            session.close()
            # GPU Memory Cleanup
            from ..utils.gpu_memory import clear_gpu_memory

            clear_gpu_memory()

    def get_analysis_stats(self) -> dict:
        """Gibt Statistiken ueber den Analyse-Status zurueck."""
        session = self.db_manager.get_session()
        try:
            from ..database.models import VideoClip

            total = session.query(VideoClip).count()
            analyzed = session.query(VideoClip).filter(VideoClip.needs_reanalysis == False).count()
            available = session.query(VideoClip).filter(VideoClip.is_available == True).count()

            return {
                "total_clips": total,
                "analyzed": analyzed,
                "unanalyzed": total - analyzed,
                "available": available,
                "unavailable": total - available,
            }
        finally:
            session.close()
            # GPU Memory Cleanup
            from ..utils.gpu_memory import clear_gpu_memory

            clear_gpu_memory()
