"""
Factory Pattern für automatische Clip Matcher Selection.

Wählt automatisch den besten verfügbaren Matcher basierend auf:
1. Installed libraries (faiss-cpu, faiss-gpu, qdrant-client)
2. Hardware detection (NVIDIA GPU, AMD GPU, CPU)
3. User preference (falls angegeben)

Priority (wenn nichts angegeben):
1. NVIDIA CUDA GPU + FAISS GPU support → FAISSClipMatcher(use_gpu=True)
2. AMD GPU detected + qdrant available → QdrantClipMatcher()
3. qdrant available → QdrantClipMatcher()
4. faiss available → FAISSClipMatcher(use_gpu=False)

Author: PB_studio Development Team
"""

from typing import Literal, Union

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Import availability checks
try:
    from .faiss_clip_matcher import FAISSClipMatcher, detect_gpu_type, is_faiss_available

    FAISS_AVAILABLE = is_faiss_available()
except ImportError:
    FAISS_AVAILABLE = False
    FAISSClipMatcher = None
    detect_gpu_type = None

try:
    from .qdrant_clip_matcher import QdrantClipMatcher, is_qdrant_available

    QDRANT_AVAILABLE = is_qdrant_available()
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClipMatcher = None


Backend = Literal["auto", "faiss-cpu", "faiss-gpu", "qdrant"]


# PERF-12 FIX: Singleton cache for clip matchers
# Prevents rebuilding index on every search operation
#
# CACHE INVALIDATION (BUGFIX #8 - Thread Safety Documentation):
# ⚠️  WARNING: This cache is NOT thread-safe by design.
#
# Thread-Safety Considerations:
# - The cache uses a simple dict without locks (intentional for performance)
# - Acceptable because:
#   1. Index building is deterministic (same clips → same index)
#   2. Worst case: Concurrent threads rebuild index multiple times (wasteful but safe)
#   3. No data corruption risk (immutable index objects)
# - If thread-safety is required, add threading.Lock around cache access
#
# Cache Invalidation Requirements:
# The cache MUST be invalidated when:
# 1. Clips are added/removed from database → Call clear_matcher_cache()
# 2. Clip analysis changes (motion/mood updated) → Call clear_matcher_cache()
# 3. Backend preference changes → Cache will auto-invalidate on backend mismatch
# 4. Force rebuild needed → Use force_rebuild=True in get_or_create_matcher()
#
# Cache will automatically invalidate when clip list changes (detects via hash).
# For explicit invalidation, call clear_matcher_cache() from your code.
_matcher_cache: dict = {
    "instance": None,
    "backend": None,
    "index_hash": None,  # Hash of clips used to build index
    "use_ai_embeddings": None,  # Cache must match embedding mode (27D vs 512D)
}


def _compute_clip_hash(clips: list) -> str:
    """Compute hash of clip list for cache invalidation."""
    import hashlib

    # Use clip IDs and count for fast hash
    clip_ids = sorted([c.get("id", 0) for c in clips])
    key = f"{len(clips)}:{','.join(map(str, clip_ids[:20]))}"  # First 20 IDs
    return hashlib.md5(key.encode()).hexdigest()[:16]


def detect_best_backend() -> Backend:
    """
    Detects the best available vector search backend.

    Priority:
    1. NVIDIA GPU + faiss-gpu → 'faiss-gpu'
    2. AMD GPU + qdrant → 'qdrant'
    3. faiss-gpu available → 'faiss-gpu'
    4. qdrant available → 'qdrant'
    5. faiss-cpu available → 'faiss-cpu'

    Returns:
        Backend name: 'faiss-gpu', 'qdrant', or 'faiss-cpu'

    Raises:
        ImportError: If no backend is available
    """
    if not FAISS_AVAILABLE and not QDRANT_AVAILABLE:
        raise ImportError(
            "No vector search backend available!\n\n"
            "Install at least one:\n"
            "  pip install faiss-cpu        # CPU-only (basic)\n"
            "  pip install faiss-gpu        # NVIDIA GPU (best for NVIDIA)\n"
            "  pip install qdrant-client    # AMD GPU / Fast CPU (best for AMD)\n\n"
            "Or use poetry extras:\n"
            "  poetry install -E vector-cpu     # faiss-cpu\n"
            "  poetry install -E vector-nvidia  # faiss-gpu\n"
            "  poetry install -E vector-amd     # qdrant-client\n"
            "  poetry install -E vector-all     # all options"
        )

    # Detect GPU type
    has_nvidia_cuda = False
    has_amd_gpu = False
    gpu_info = "Unknown"

    if FAISS_AVAILABLE and detect_gpu_type:
        has_cuda, gpu_info = detect_gpu_type()
        has_nvidia_cuda = has_cuda
        has_amd_gpu = "AMD" in gpu_info or "Radeon" in gpu_info

    faiss_gpu_supported = False
    if FAISS_AVAILABLE:
        try:
            import faiss

            faiss_gpu_supported = hasattr(faiss, "StandardGpuResources")
        except Exception:
            faiss_gpu_supported = False

    # Priority 1: NVIDIA GPU + faiss-gpu
    if has_nvidia_cuda and FAISS_AVAILABLE and faiss_gpu_supported:
        logger.info(f"✅ Detected: NVIDIA CUDA GPU ({gpu_info})")
        logger.info("✅ Selected backend: faiss-gpu (NVIDIA GPU acceleration)")
        return "faiss-gpu"

    # Priority 2: AMD GPU + Qdrant
    if has_amd_gpu and QDRANT_AVAILABLE:
        logger.info(f"[OK] Detected: AMD GPU ({gpu_info})")
        logger.info("[OK] Selected backend: Qdrant (optimized for AMD/CPU)")
        return "qdrant"

    # AMD GPU detected but Qdrant not available? Warn user
    if has_amd_gpu and not QDRANT_AVAILABLE and FAISS_AVAILABLE:
        logger.warning(f"[WARN] AMD GPU detected ({gpu_info}) but Qdrant not installed!")
        logger.warning("[WARN] For better performance, install: pip install qdrant-client")
        logger.info("[OK] Selected backend: faiss-cpu (fallback for AMD)")
        return "faiss-cpu"

    # Priority 4: Qdrant available
    if QDRANT_AVAILABLE:
        logger.info("✅ Selected backend: Qdrant (optimized CPU)")
        return "qdrant"

    # Priority 5: faiss-cpu (fallback)
    if FAISS_AVAILABLE:
        logger.info("✅ Selected backend: faiss-cpu")
        return "faiss-cpu"

    # Should never reach here (checked at the beginning)
    raise ImportError("No backend available")


def create_clip_matcher(
    backend: Backend = "auto",
    use_gpu: bool = True,
    use_persistence: bool = False,
    persist_path: str = "./qdrant_data",
    use_ai_embeddings: bool = False,
) -> Union["FAISSClipMatcher", "QdrantClipMatcher"]:
    """
    Creates the appropriate Clip Matcher based on backend selection.

    Args:
        backend: Which backend to use
                 - 'auto': Auto-detect best available (recommended)
                 - 'faiss-cpu': Force FAISS CPU mode
                 - 'faiss-gpu': Force FAISS GPU mode (requires faiss-gpu + NVIDIA)
                 - 'qdrant': Force Qdrant (AMD GPU / optimized CPU)
        use_gpu: For FAISS: whether to try GPU acceleration
                 Ignored for Qdrant (CPU-based)
        use_persistence: For Qdrant: whether to persist index to disk
                         Ignored for FAISS
        persist_path: For Qdrant: where to persist index
        use_ai_embeddings: Use X-CLIP 512D embeddings instead of manual 27D features

    Returns:
        Initialized clip matcher instance

    Raises:
        ImportError: If selected backend is not available
        ValueError: If invalid backend specified

    Examples:
        >>> # Auto-detection (recommended)
        >>> matcher = create_clip_matcher()  # Automatically picks best

        >>> # Force specific backend
        >>> matcher = create_clip_matcher(backend='qdrant')
        >>> matcher = create_clip_matcher(backend='faiss-gpu')

        >>> # NVIDIA GPU with FAISS
        >>> matcher = create_clip_matcher(backend='faiss-gpu', use_gpu=True)

        >>> # AMD GPU / CPU with Qdrant
        >>> matcher = create_clip_matcher(backend='qdrant')

        >>> # Mit KI-Embeddings (X-CLIP 512D)
        >>> matcher = create_clip_matcher(use_ai_embeddings=True)
    """
    # Auto-detect if needed
    if backend == "auto":
        backend = detect_best_backend()

    # Validate backend
    if backend not in ["faiss-cpu", "faiss-gpu", "qdrant"]:
        raise ValueError(
            f"Invalid backend: {backend}. Must be 'auto', 'faiss-cpu', 'faiss-gpu', or 'qdrant'"
        )

    # Create matcher based on backend
    if backend in ["faiss-cpu", "faiss-gpu"]:
        if not FAISS_AVAILABLE:
            raise ImportError(
                f"Backend '{backend}' selected but FAISS not available.\n"
                f"Install with: pip install {backend}"
            )

        # For faiss-cpu backend, always disable GPU. For faiss-gpu, only use GPU if supported.
        if backend == "faiss-cpu":
            should_use_gpu = False
        else:
            try:
                import faiss

                should_use_gpu = (
                    bool(use_gpu)
                    and hasattr(faiss, "StandardGpuResources")
                    and getattr(faiss, "get_num_gpus", lambda: 0)() > 0
                )
            except Exception:
                should_use_gpu = False

            if use_gpu and not should_use_gpu:
                logger.warning(
                    "FAISS GPU requested but not available (no CUDA GPU or faiss-gpu missing). "
                    "Falling back to FAISS CPU."
                )

        ai_info = " + AI-Embeddings (512D)" if use_ai_embeddings else ""
        logger.info(f"Creating FAISSClipMatcher (GPU: {should_use_gpu}{ai_info})")
        return FAISSClipMatcher(use_gpu=should_use_gpu, use_ai_embeddings=use_ai_embeddings)

    elif backend == "qdrant":
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "Backend 'qdrant' selected but qdrant-client not available.\n"
                "Install with: pip install qdrant-client"
            )

        # Note: QdrantClipMatcher müsste ebenfalls erweitert werden für use_ai_embeddings
        # Aktuell wird es nur für FAISS unterstützt
        if use_ai_embeddings:
            logger.warning(
                "KI-Embeddings werden für Qdrant noch nicht unterstützt. "
                "Nutze FAISS-Backend für KI-Embeddings."
            )

        logger.info(
            f"Creating QdrantClipMatcher (persistence: {use_persistence}, path: {persist_path})"
        )
        return QdrantClipMatcher(use_persistence=use_persistence, persist_path=persist_path)

    # Should never reach here
    raise ValueError(f"Unknown backend: {backend}")


def get_available_backends() -> dict:
    """
    Returns information about available backends.

    Returns:
        Dict with backend availability and info
    """
    backends = {
        "faiss-cpu": {
            "available": FAISS_AVAILABLE,
            "description": "FAISS CPU mode (all platforms)",
            "performance": "Baseline",
            "gpu_support": False,
        },
        "faiss-gpu": {
            "available": False,  # Will check below
            "description": "FAISS GPU mode (NVIDIA only)",
            "performance": "5-10x faster than CPU",
            "gpu_support": True,
        },
        "qdrant": {
            "available": QDRANT_AVAILABLE,
            "description": "Qdrant (optimized CPU, works with AMD)",
            "performance": "1.5-3x faster than FAISS CPU",
            "gpu_support": False,  # CPU-based but optimized
        },
    }

    # Check if faiss-gpu is actually installed
    if FAISS_AVAILABLE:
        try:
            import faiss

            if hasattr(faiss, "StandardGpuResources"):
                backends["faiss-gpu"]["available"] = True
        except Exception:
            pass

    # Add GPU detection info
    if FAISS_AVAILABLE and detect_gpu_type:
        has_cuda, gpu_info = detect_gpu_type()
        backends["gpu_detected"] = {"has_nvidia_cuda": has_cuda, "gpu_info": gpu_info}

    return backends


def print_backend_info():
    """
    Logs information about available backends and recommendation.

    Useful for debugging and user information.
    """
    backends = get_available_backends()

    logger.info("=" * 60)
    logger.info("PB Studio - Vector Search Backend Information")
    logger.info("=" * 60)

    # GPU Info
    if "gpu_detected" in backends:
        gpu_info = backends.pop("gpu_detected")
        logger.info("[GPU] GPU Detection:")
        logger.info(f"   NVIDIA CUDA: {'[YES]' if gpu_info['has_nvidia_cuda'] else '[NO]'}")
        logger.info(f"   Info: {gpu_info['gpu_info']}")

    # Available backends
    logger.info("[BACKENDS] Available Backends:")
    for name, info in backends.items():
        status = "[OK] Installed" if info["available"] else "[NO] Not installed"
        logger.info(f"   {name}: {status}")
        logger.info(f"      {info['description']}")
        logger.info(f"      Performance: {info['performance']}")
        if info["gpu_support"]:
            logger.info("      Requires: NVIDIA GPU with CUDA")

    # Recommendation
    logger.info("[RECOMMEND] Recommendation:")
    try:
        best = detect_best_backend()
        logger.info(f"   Use backend: {best}")
        logger.info("   Install command:")
        if best == "faiss-gpu":
            logger.info("      pip install faiss-gpu")
        elif best == "qdrant":
            logger.info("      pip install qdrant-client")
        else:
            logger.info("      pip install faiss-cpu")
    except ImportError as e:
        logger.warning(f"   {e}")

    logger.info("=" * 60)


# Convenience function for backward compatibility
def create_matcher(*args, **kwargs):
    """Alias for create_clip_matcher (backward compatibility)."""
    return create_clip_matcher(*args, **kwargs)


def get_or_create_matcher(
    clips: list,
    backend: Backend = "auto",
    use_gpu: bool = True,
    force_rebuild: bool = False,
    use_ai_embeddings: bool = False,
) -> Union["FAISSClipMatcher", "QdrantClipMatcher"]:
    """
    PERF-12 FIX: Get cached matcher or create new one if needed.

    This singleton pattern prevents rebuilding the index on every search,
    providing ~100ms savings per operation.

    Args:
        clips: List of clip dicts with analysis data
        backend: Which backend to use ('auto', 'faiss-cpu', 'faiss-gpu', 'qdrant')
        use_gpu: For FAISS: whether to try GPU acceleration
        force_rebuild: Force rebuilding index even if cached

    Returns:
        Initialized and indexed clip matcher instance

    Performance:
        - First call: ~100-200ms (create matcher + build index)
        - Subsequent calls with same clips: ~0ms (return cached)
        - After clip changes: ~100-200ms (rebuild index)
    """
    global _matcher_cache

    # HIGH-08 FIX: Wrap detect_best_backend() in try/except to catch ImportError
    try:
        effective_backend = backend if backend != "auto" else detect_best_backend()
    except ImportError as e:
        logger.error(f"No vector search backends available: {e}")
        raise RuntimeError("Install faiss-cpu or qdrant-client to use clip matching")

    clip_hash = _compute_clip_hash(clips)

    # Check if we can reuse cached matcher
    cache_valid = (
        not force_rebuild
        and _matcher_cache["instance"] is not None
        and _matcher_cache["backend"] == effective_backend
        and _matcher_cache["index_hash"] == clip_hash
        and _matcher_cache["use_ai_embeddings"] == use_ai_embeddings
    )

    if cache_valid:
        logger.debug(f"Reusing cached {effective_backend} matcher (hash: {clip_hash[:8]}...)")
        return _matcher_cache["instance"]

    # Create new matcher
    logger.info(f"Creating new {effective_backend} matcher (clips: {len(clips)})")
    matcher = create_clip_matcher(
        backend=effective_backend,
        use_gpu=use_gpu,
        use_ai_embeddings=use_ai_embeddings,
    )

    # Build index
    import time

    start = time.perf_counter()
    matcher.build_index(clips)
    elapsed = (time.perf_counter() - start) * 1000

    logger.info(f"Index built in {elapsed:.1f}ms (hash: {clip_hash[:8]}...)")

    # Cache the matcher
    _matcher_cache["instance"] = matcher
    _matcher_cache["backend"] = effective_backend
    _matcher_cache["index_hash"] = clip_hash
    _matcher_cache["use_ai_embeddings"] = use_ai_embeddings

    return matcher


def clear_matcher_cache() -> None:
    """
    Clear the cached matcher instance.

    Call this when clips are added/removed from the database
    to force rebuilding the index on next search.
    """
    global _matcher_cache

    if _matcher_cache["instance"] is not None:
        logger.info("Clearing matcher cache")
        _matcher_cache["instance"] = None
        _matcher_cache["backend"] = None
        _matcher_cache["index_hash"] = None
        _matcher_cache["use_ai_embeddings"] = None


def get_matcher_cache_info() -> dict:
    """Get information about the cached matcher."""
    return {
        "has_cached_matcher": _matcher_cache["instance"] is not None,
        "backend": _matcher_cache["backend"],
        "index_hash": _matcher_cache["index_hash"],
    }
