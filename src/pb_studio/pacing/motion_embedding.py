"""
Motion Embedding Extractor für FAISS Vector Similarity Search.

Konvertiert Clip-Analyse-Daten zu numerischen Vektoren für effizientes Matching.

Features:
- 27-dimensionale Vektoren aus Motion/Mood/Energy/Temporal-Daten (Phase 2)
- One-Hot/Multi-Hot Encoding für kategorische Features
- Normalisierte float32 numpy arrays (FAISS-kompatibel)
- Query-Vector-Generierung für Similarity Search

Vector Structure (27 dims):
[0]     motion_score        (0-1)
[1]     energy              (0-1)
[2-6]   motion_type_one_hot (STATIC, SLOW, MEDIUM, FAST, EXTREME)
[7-20]  mood_multi_hot      (14 moods)
[21]    camera_magnitude    (0-1)
[22]    brightness_dynamics (0-1) [Phase 2]
[23]    color_dynamics      (0-1) [Phase 2]
[24-26] temporal_rhythm_one_hot (STEADY, DYNAMIC, FLASHY) [Phase 2]

Author: PB_studio Development Team
"""

import hashlib
from typing import Any

import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    Konvertiert Wert sicher zu float.

    HIGH-07 FIX: Verhindert TypeError/ValueError bei ungültigen Werten.

    Args:
        value: Zu konvertierender Wert
        default: Fallback-Wert bei Fehler

    Returns:
        Float-Wert oder default bei Fehler
    """
    if value is None:
        return default
    try:
        result = float(value)
        # NaN/Inf check
        if not np.isfinite(result):
            logger.debug(f"Non-finite value {value}, using default {default}")
            return default
        return result
    except (TypeError, ValueError) as e:
        logger.debug(f"Could not convert {type(value).__name__} to float: {e}")
        return default


# Vector Dimensionen
DIM_MOTION = 1  # motion_score
DIM_ENERGY = 1  # energy
DIM_MOTION_TYPE = 5  # STATIC, SLOW, MEDIUM, FAST, EXTREME
DIM_MOOD = 14  # 14 verschiedene Moods
DIM_CAMERA = 1  # camera_magnitude
DIM_BRIGHTNESS_DYN = 1  # brightness_dynamics (Phase 2)
DIM_COLOR_DYN = 1  # color_dynamics (Phase 2)
DIM_TEMPORAL_RHYTHM = 3  # temporal_rhythm One-Hot (Phase 2)
TOTAL_DIM = 27  # Gesamt-Dimensionen (Phase 2: erweitert von 22 auf 27)

# AI Embedding Dimension (X-CLIP 512D) - muss vor validate_embedding() definiert sein
AI_EMBEDDING_DIM = 512

# Kategorische Features
MOTION_TYPES = ["STATIC", "SLOW", "MEDIUM", "FAST", "EXTREME"]

MOODS = [
    "ENERGETIC",
    "CALM",
    "DARK",
    "BRIGHT",
    "MELANCHOLIC",
    "EUPHORIC",
    "AGGRESSIVE",
    "PEACEFUL",
    "MYSTERIOUS",
    "CHEERFUL",
    "TENSE",
    "DREAMY",
    "COOL",
    "WARM",
]

TEMPORAL_RHYTHMS = ["STEADY", "DYNAMIC", "FLASHY"]  # Phase 2


def _get_deterministic_seed(analysis_data: dict[str, Any]) -> int:
    """
    Get deterministic seed from analysis_data.

    CRITICAL FIX: Python's hash() is randomized in 3.3+ for security.
    We need true determinism for reproducible embeddings.

    Args:
        analysis_data: Analysis data dict

    Returns:
        Deterministic integer seed
    """
    # Sort dict items for determinism
    data_str = str(sorted(analysis_data.items()))
    hash_obj = hashlib.md5(data_str.encode())
    # Use first 8 hex digits as integer (32 bits)
    return int(hash_obj.hexdigest()[:8], 16)


def extract_motion_embedding(analysis_data: dict[str, Any]) -> np.ndarray:
    """
    Extrahiert 27-dimensionalen Motion-Embedding-Vektor aus Clip-Analyse-Daten.

    Args:
        analysis_data: Dict mit 'motion', 'mood', 'color' etc. Analyse-Daten

    Returns:
        numpy array shape (27,) dtype float32

    Example:
        >>> analysis = {
        ...     'motion': {'motion_score': 0.75, 'motion_type': 'FAST'},
        ...     'mood': {'energy': 0.8, 'moods': ['ENERGETIC', 'CHEERFUL']}
        ... }
        >>> vector = extract_motion_embedding(analysis)
        >>> vector.shape
        (27,)
        >>> vector.dtype
        dtype('float32')
    """
    vector = np.zeros(TOTAL_DIM, dtype=np.float32)

    # Check if analysis data is empty (unanalyzed clip)
    motion_data = analysis_data.get("motion", {})
    mood_data = analysis_data.get("mood", {})

    # Detect unanalyzed clips and use clip_id for deterministic randomness
    is_analyzed = bool(motion_data) or bool(mood_data)

    if not is_analyzed:
        # For unanalyzed clips: use deterministic pseudo-random values based on analysis_data id
        # This ensures variety in FAISS matching while being reproducible
        # CRITICAL FIX: Use deterministic hash instead of Python's randomized hash()
        clip_id = analysis_data.get("id", _get_deterministic_seed(analysis_data))

        # IMPROVED: Better seeding for more diversity
        # Mix clip_id with prime numbers to spread out similar IDs
        seed = (clip_id * 16807 + 982451653) % 2147483647  # Linear congruential generator
        rng = np.random.RandomState(seed)

        # IMPROVED: Wider range for more diversity (0.0-1.0 instead of 0.2-0.8)
        vector[0] = rng.uniform(0.0, 1.0)  # motion_score
        vector[1] = rng.uniform(0.0, 1.0)  # energy

        # Random motion type
        motion_type_idx = rng.randint(0, 5)
        vector[2 + motion_type_idx] = 1.0

        # IMPROVED: Random 2-5 moods (more variation)
        num_moods = rng.randint(2, 6)
        mood_indices = rng.choice(14, size=num_moods, replace=False)
        for idx in mood_indices:
            vector[7 + idx] = 1.0

        # IMPROVED: Wider range for camera (0.0-1.0)
        vector[21] = rng.uniform(0.0, 1.0)

        # Phase 2: Temporal Features pseudo-random
        vector[22] = rng.uniform(0.0, 1.0)  # brightness_dynamics
        vector[23] = rng.uniform(0.0, 1.0)  # color_dynamics

        # Random temporal rhythm
        temporal_rhythm_idx = rng.randint(0, 3)
        vector[24 + temporal_rhythm_idx] = 1.0

        logger.debug(f"Generated pseudo-random embedding for unanalyzed clip (id={clip_id})")
        return vector

    # [0] Motion Score (0-1)
    # HIGH-07 FIX: Type-safe float conversion with _safe_float()
    if isinstance(motion_data, dict):
        vector[0] = np.clip(_safe_float(motion_data.get("motion_score"), 0.5), 0.0, 1.0)
    else:
        vector[0] = 0.5  # Default

    # [1] Energy (0-1)
    # HIGH-07 FIX: Type-safe float conversion with _safe_float()
    if isinstance(mood_data, dict):
        vector[1] = np.clip(_safe_float(mood_data.get("energy"), 0.5), 0.0, 1.0)
    else:
        vector[1] = 0.5  # Default

    # [2-6] Motion Type (One-Hot Encoding)
    motion_type = (
        motion_data.get("motion_type", "MEDIUM") if isinstance(motion_data, dict) else "MEDIUM"
    )
    if motion_type in MOTION_TYPES:
        idx = MOTION_TYPES.index(motion_type)
        vector[2 + idx] = 1.0
    else:
        # Default: MEDIUM (idx=2)
        vector[4] = 1.0

    # [7-20] Moods (Multi-Hot Encoding - kann mehrere sein)
    clip_moods = mood_data.get("moods", []) if isinstance(mood_data, dict) else []
    if isinstance(clip_moods, list):
        for mood in clip_moods:
            if mood in MOODS:
                idx = MOODS.index(mood)
                vector[7 + idx] = 1.0

    # [21] Camera Motion Magnitude (0-1)
    # HIGH-07 FIX: Type-safe float conversion with _safe_float()
    if isinstance(motion_data, dict):
        vector[21] = np.clip(_safe_float(motion_data.get("camera_magnitude"), 0.0), 0.0, 1.0)
    else:
        vector[21] = 0.0

    # ==================== Phase 2: Temporal Features ====================
    # [22] Brightness Dynamics (0-1)
    color_data = analysis_data.get("colors", {})
    if isinstance(color_data, dict):
        vector[22] = np.clip(_safe_float(color_data.get("brightness_dynamics"), 0.0), 0.0, 1.0)
    else:
        vector[22] = 0.0

    # [23] Color Dynamics (0-1)
    if isinstance(color_data, dict):
        vector[23] = np.clip(_safe_float(color_data.get("color_dynamics"), 0.0), 0.0, 1.0)
    else:
        vector[23] = 0.0

    # [24-26] Temporal Rhythm (One-Hot Encoding)
    temporal_rhythm = (
        color_data.get("temporal_rhythm", "STEADY") if isinstance(color_data, dict) else "STEADY"
    )
    if temporal_rhythm in TEMPORAL_RHYTHMS:
        idx = TEMPORAL_RHYTHMS.index(temporal_rhythm)
        vector[24 + idx] = 1.0
    else:
        # Default: STEADY (idx=0)
        vector[24] = 1.0

    return vector


def create_query_vector(
    target_motion_score: float = 0.5,
    target_energy: float = 0.5,
    target_motion_type: str = "MEDIUM",
    target_moods: list[str] = None,
    target_brightness_dynamics: float = 0.0,
    target_color_dynamics: float = 0.0,
    target_temporal_rhythm: str = "STEADY",
) -> np.ndarray:
    """
    Erstellt Query-Vektor für FAISS Similarity Search.

    Args:
        target_motion_score: Ziel-Motion-Intensität (0-1)
        target_energy: Ziel-Energie-Level (0-1)
        target_motion_type: Ziel-Motion-Typ (STATIC, SLOW, MEDIUM, FAST, EXTREME)
        target_moods: Liste von Ziel-Moods (z.B. ['ENERGETIC', 'CHEERFUL'])
        target_brightness_dynamics: Ziel-Helligkeits-Dynamik (0-1) [Phase 2]
        target_color_dynamics: Ziel-Farb-Dynamik (0-1) [Phase 2]
        target_temporal_rhythm: Ziel-Temporal-Rhythmus (STEADY, DYNAMIC, FLASHY) [Phase 2]

    Returns:
        numpy array shape (27,) dtype float32

    Example:
        >>> query = create_query_vector(
        ...     target_motion_score=0.8,
        ...     target_energy=0.7,
        ...     target_motion_type='FAST',
        ...     target_moods=['ENERGETIC']
        ... )
        >>> query[0]  # motion_score
        0.8
        >>> query[5]  # FAST one-hot (idx=3 in MOTION_TYPES)
        1.0
    """
    vector = np.zeros(TOTAL_DIM, dtype=np.float32)

    # [0] Motion Score
    vector[0] = np.clip(_safe_float(target_motion_score, 0.5), 0.0, 1.0)

    # [1] Energy
    vector[1] = np.clip(_safe_float(target_energy, 0.5), 0.0, 1.0)

    # [2-6] Motion Type One-Hot
    if target_motion_type in MOTION_TYPES:
        idx = MOTION_TYPES.index(target_motion_type)
        vector[2 + idx] = 1.0
    else:
        # Default: MEDIUM (idx=2)
        vector[4] = 1.0
        logger.warning(
            f"Unknown motion type '{target_motion_type}', using MEDIUM. Valid types: {MOTION_TYPES}"
        )

    # [7-20] Moods Multi-Hot
    target_moods = target_moods or []
    for mood in target_moods:
        if mood in MOODS:
            idx = MOODS.index(mood)
            vector[7 + idx] = 1.0
        else:
            logger.warning(f"Unknown mood '{mood}', skipping. Valid moods: {MOODS}")

    # ==================== Phase 2: Temporal Features ====================
    # [22] Brightness Dynamics
    vector[22] = np.clip(target_brightness_dynamics, 0.0, 1.0)

    # [23] Color Dynamics
    vector[23] = np.clip(target_color_dynamics, 0.0, 1.0)

    # [24-26] Temporal Rhythm One-Hot
    if target_temporal_rhythm in TEMPORAL_RHYTHMS:
        idx = TEMPORAL_RHYTHMS.index(target_temporal_rhythm)
        vector[24 + idx] = 1.0
    else:
        # Default: STEADY (idx=0)
        vector[24] = 1.0
        logger.warning(
            f"Unknown temporal rhythm '{target_temporal_rhythm}', using STEADY. "
            f"Valid rhythms: {TEMPORAL_RHYTHMS}"
        )

    return vector


def vector_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Berechnet L2-Distanz zwischen zwei Vektoren (wie FAISS IndexFlatL2).

    Args:
        vec1: Vektor 1
        vec2: Vektor 2

    Returns:
        L2-Distanz (niedriger = ähnlicher)

    Note:
        Dies ist nur für Testing/Debugging. FAISS verwendet optimierte C++ Implementation.
    """
    return float(np.linalg.norm(vec1 - vec2))


# Utility Functions


def get_embedding_dimension() -> int:
    """Returns total embedding dimension (27)."""
    return TOTAL_DIM


def get_motion_type_index(motion_type: str) -> int:
    """
    Returns index of motion type in MOTION_TYPES list.

    Args:
        motion_type: Motion type string

    Returns:
        Index (0-4) or 2 (MEDIUM) as default
    """
    if motion_type in MOTION_TYPES:
        return MOTION_TYPES.index(motion_type)
    return 2  # Default: MEDIUM


def get_mood_indices(moods: list[str]) -> list[int]:
    """
    Returns indices of moods in MOODS list.

    Args:
        moods: List of mood strings

    Returns:
        List of indices
    """
    indices = []
    for mood in moods:
        if mood in MOODS:
            indices.append(MOODS.index(mood))
        else:
            # MEDIUM-04 FIX: Add logging for invalid moods
            logger.warning(f"Invalid mood '{mood}' ignored. Valid moods: {MOODS}")
    return indices


def validate_embedding(embedding: np.ndarray, expected_dim: int = None) -> bool:
    """
    Validates embedding vector.

    Args:
        embedding: Embedding vector
        expected_dim: Expected dimension (None = use TOTAL_DIM for manual, or auto-detect)

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(embedding, np.ndarray):
        logger.error("Embedding is not a numpy array")
        return False

    if embedding.dtype != np.float32:
        logger.error(f"Embedding dtype is {embedding.dtype}, expected float32")
        return False

    # Allow both 27D (manual) and 512D (AI) embeddings
    if expected_dim is not None:
        if embedding.shape != (expected_dim,):
            logger.error(f"Embedding shape is {embedding.shape}, expected ({expected_dim},)")
            return False
    else:
        if embedding.shape not in [(TOTAL_DIM,), (AI_EMBEDDING_DIM,)]:
            logger.error(
                f"Embedding shape is {embedding.shape}, expected ({TOTAL_DIM},) or ({AI_EMBEDDING_DIM},)"
            )
            return False

    if not np.all(np.isfinite(embedding)):
        logger.error("Embedding contains NaN or Inf values")
        return False

    return True


# =============================================================================
# Phase KI: Deep Learning Video Embeddings (Optional)
# =============================================================================


def extract_ai_video_embedding(video_path: str) -> np.ndarray | None:
    """
    Extrahiert 512D KI-Embedding aus Video (optional).

    Nutzt X-CLIP für semantische Video-Analyse. Wenn KI nicht verfügbar,
    wird None zurückgegeben und der Caller sollte auf manuelle Features fallbacken.

    Args:
        video_path: Pfad zur Video-Datei

    Returns:
        np.ndarray shape (512,) dtype float32, oder None wenn KI nicht verfügbar

    Example:
        >>> embedding = extract_ai_video_embedding("video.mp4")
        >>> if embedding is not None:
        ...     print(f"AI Embedding: {embedding.shape}")
        ... else:
        ...     print("Fallback zu manuellen Features")
    """
    try:
        from ..video.video_analyzer_ai import AIVideoAnalyzer

        analyzer = AIVideoAnalyzer()
        if not analyzer.available:
            return None
        return analyzer.extract_video_embedding(video_path)
    except ImportError:
        logger.debug("AI Video Analyzer nicht verfügbar (Dependencies fehlen)")
        return None
    except Exception as e:
        logger.warning(f"AI Embedding Extraktion fehlgeschlagen: {e}")
        return None


def get_ai_embedding_dimension() -> int:
    """Returns AI embedding dimension (512 for X-CLIP)."""
    return AI_EMBEDDING_DIM


def is_ai_embedding_available() -> bool:
    """
    Prüft ob KI-basierte Video-Embeddings verfügbar sind.

    Returns:
        True wenn X-CLIP geladen werden kann
    """
    try:
        from ..video.video_analyzer_ai import AIVideoAnalyzer

        analyzer = AIVideoAnalyzer()
        return analyzer.available
    except Exception:
        return False


def extract_embedding_auto(
    analysis_data: dict[str, Any], video_path: str | None = None, prefer_ai: bool = False
) -> np.ndarray:
    """
    Extrahiert Embedding mit automatischem Fallback.

    Versucht zuerst KI-Embedding (512D), fällt auf manuelle Features (27D) zurück.

    Args:
        analysis_data: Dict mit Analyse-Daten (für manuelle Features)
        video_path: Pfad zur Video-Datei (für KI-Features, optional)
        prefer_ai: True = versuche zuerst KI, False = nutze nur manuelle Features

    Returns:
        np.ndarray shape (512,) wenn KI erfolgreich, sonst (27,)

    Example:
        >>> # KI bevorzugt
        >>> emb = extract_embedding_auto(data, "video.mp4", prefer_ai=True)
        >>> print(emb.shape)  # (512,) wenn KI verfügbar, sonst (27,)
        >>>
        >>> # Nur manuelle Features
        >>> emb = extract_embedding_auto(data, prefer_ai=False)
        >>> print(emb.shape)  # Immer (27,)
    """
    # KI-Pfad wenn gewünscht und video_path vorhanden
    if prefer_ai and video_path:
        ai_embedding = extract_ai_video_embedding(video_path)
        if ai_embedding is not None:
            logger.debug(f"Using AI embedding (512D) for {video_path}")
            return ai_embedding
        logger.debug("AI embedding failed, falling back to manual features")

    # Fallback zu manuellen Features (27D)
    return extract_motion_embedding(analysis_data)
