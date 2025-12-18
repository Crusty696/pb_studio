"""
Zentrale Konstanten für PB_studio

QUAL-02 FIX: Magic Numbers in zentrale Konstanten extrahiert.

Diese Datei enthält alle häufig verwendeten Konstanten,
um Magic Numbers im Code zu vermeiden.
"""

# =============================================================================
# VIDEO DEFAULTS
# =============================================================================

# Auflösungen
DEFAULT_VIDEO_WIDTH = 1920
DEFAULT_VIDEO_HEIGHT = 1080
DEFAULT_RESOLUTION = (DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_HEIGHT)

# Framerates
DEFAULT_FPS = 30.0
HIGH_FPS = 60.0

# Thumbnail
DEFAULT_THUMBNAIL_WIDTH = 160
DEFAULT_THUMBNAIL_HEIGHT = 90
DEFAULT_THUMBNAIL_SIZE = (DEFAULT_THUMBNAIL_WIDTH, DEFAULT_THUMBNAIL_HEIGHT)
DEFAULT_THUMBNAIL_QUALITY = 85

# =============================================================================
# AUDIO DEFAULTS
# =============================================================================

# Sample Rates
DEFAULT_SAMPLE_RATE = 22050  # Standard für librosa
LOW_SAMPLE_RATE = 11025  # Optimiert für Beat-Tracking

# Bitrates
DEFAULT_AUDIO_BITRATE = "128k"
HIGH_AUDIO_BITRATE = "320k"

# =============================================================================
# PACING / TIMING
# =============================================================================

# Minimale/Maximale Schnittintervalle (Sekunden)
MIN_CUT_INTERVAL = 0.5
MAX_CUT_INTERVAL = 8.0
DEFAULT_CUT_INTERVAL = 2.0

# Beat-basierte Schnitte
BEATS_PER_BAR = 4
DEFAULT_BPM = 120.0

# =============================================================================
# TIMEOUTS (Sekunden)
# =============================================================================

# FFmpeg Timeouts
FFMPEG_TIMEOUT_SEGMENT = 120  # 2 min pro Segment
FFMPEG_TIMEOUT_CONCAT = 1800  # 30 min für Konkatenierung
FFMPEG_TIMEOUT_PROBE = 30  # 30 sec für Metadaten
FFMPEG_TIMEOUT_THUMBNAIL = 10  # 10 sec für Thumbnail

# Render Timeouts
RENDER_TIMEOUT_SHORT = 60  # 1 min
RENDER_TIMEOUT_MEDIUM = 300  # 5 min
RENDER_TIMEOUT_LONG = 1800  # 30 min

# =============================================================================
# UI / DISPLAY
# =============================================================================

# Waveform
DEFAULT_WAVEFORM_WIDTH = 1920
WAVEFORM_PEAK_SCALE = 0.9  # 90% der verfügbaren Höhe

# Progress-Reporting
PROGRESS_START = 0.0
PROGRESS_HALF = 0.5
PROGRESS_ALMOST_DONE = 0.8
PROGRESS_COMPLETE = 1.0

# Pagination
DEFAULT_PAGE_SIZE = 50
MAX_INITIAL_RENDER = 50

# =============================================================================
# CACHING
# =============================================================================

# Cache Größen
DEFAULT_CACHE_SIZE = 50
MAX_MEMORY_CACHE_MB = 512

# Cache TTL (Sekunden)
CACHE_TTL_SHORT = 300  # 5 min
CACHE_TTL_MEDIUM = 3600  # 1 hour
CACHE_TTL_LONG = 86400  # 24 hours

# =============================================================================
# ANALYSIS
# =============================================================================

# Frame-Sampling
DEFAULT_MAX_FRAMES = 30
MIN_FRAMES_FOR_ANALYSIS = 10

# Similarity Thresholds
SIMILARITY_THRESHOLD_HIGH = 0.9
SIMILARITY_THRESHOLD_MEDIUM = 0.7
SIMILARITY_THRESHOLD_LOW = 0.5

# Motion Scores
MOTION_SCORE_STATIC = 0.2
MOTION_SCORE_MODERATE = 0.5
MOTION_SCORE_FAST = 0.8

# =============================================================================
# VECTOR SEARCH
# =============================================================================

# FAISS/Qdrant
DEFAULT_EMBEDDING_DIM = 128
TOP_K_RESULTS = 10
MIN_DISTANCE_THRESHOLD = 0.1

# =============================================================================
# FILE FORMATS
# =============================================================================

SUPPORTED_VIDEO_FORMATS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
SUPPORTED_AUDIO_FORMATS = {".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a"}
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

# =============================================================================
# ENCODING
# =============================================================================

# Video Codecs
DEFAULT_VIDEO_CODEC = "libx264"
DEFAULT_AUDIO_CODEC = "aac"

# CRF (Constant Rate Factor) - Lower = Better Quality
CRF_LOSSLESS = 0
CRF_HIGH_QUALITY = 18
CRF_BALANCED = 23
CRF_LOW_QUALITY = 28
CRF_DRAFT = 35

# Presets (Speed vs Quality)
PRESET_ULTRAFAST = "ultrafast"
PRESET_SUPERFAST = "superfast"
PRESET_VERYFAST = "veryfast"
PRESET_FASTER = "faster"
PRESET_FAST = "fast"
PRESET_MEDIUM = "medium"
PRESET_SLOW = "slow"
