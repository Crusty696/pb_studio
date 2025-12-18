"""
Semantic Clip Matcher using CLIP embeddings and FAISS.
Enables "Text-to-Video" search functionality.
"""

from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from ..analysis.analyzers.semantic_analyzer import SemanticAnalyzer
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SemanticClipMatcher:
    """
    Matcher for semantic video search using CLIP embeddings.
    Allows finding clips based on text descriptions (e.g. "party", "sunset")
    or visual similarity to other clips.
    """

    def __init__(self, semantic_analyzer: SemanticAnalyzer | None = None):
        """
        Initialize matcher.

        Args:
            semantic_analyzer: Optional existing analyzer instance.
        """
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available. Semantic matching will be disabled.")

        self.semantic_analyzer = semantic_analyzer or SemanticAnalyzer()
        self.index = None
        self.clip_ids: list[int] = []
        self.clip_paths: dict[int, str] = {}
        self.dimension = 512  # Standard for CLIP ViT-B/32

    def build_index(self, clips: list[dict[str, Any]]) -> None:
        """
        Build FAISS index from clip embeddings.

        Args:
            clips: List of clip dictionaries.
                   Must contain reference to embedding file (.npy).
                   Looks in:
                   - clip['embedding_path']
                   - clip['analysis']['embedding_path']
                   - clip['fingerprint']['vector_file']
        """
        if not FAISS_AVAILABLE:
            return

        logger.info(f"Building semantic index for {len(clips)} clips...")

        embeddings_list = []
        valid_clip_ids = []

        for clip in clips:
            clip_id = clip.get("id")
            if not clip_id:
                continue

            # Find embedding path
            embedding_path = None

            # Strategy 1: Direct key
            if "embedding_path" in clip:
                embedding_path = clip["embedding_path"]
            # Strategy 2: Analysis dict
            elif "analysis" in clip and isinstance(clip["analysis"], dict):
                embedding_path = clip["analysis"].get("embedding_path")
            # Strategy 3: Fingerprint dict
            elif "fingerprint" in clip and isinstance(clip["fingerprint"], dict):
                embedding_path = clip["fingerprint"].get("vector_file")

            if not embedding_path or not Path(embedding_path).exists():
                continue

            try:
                # Load embedding
                vector = np.load(embedding_path)

                # Check dimension
                if vector.shape[0] != self.dimension:
                    logger.warning(
                        f"Clip {clip_id}: Dimension mismatch {vector.shape[0]} != {self.dimension}"
                    )
                    continue

                embeddings_list.append(vector)
                valid_clip_ids.append(clip_id)
                self.clip_paths[clip_id] = clip.get("file_path", "")

            except Exception as e:
                logger.debug(f"Failed to load embedding for clip {clip_id}: {e}")
                continue

        if not embeddings_list:
            logger.warning("No valid embeddings found. Semantic search disabled.")
            return

        # Convert to numpy
        embeddings_np = np.array(embeddings_list).astype("float32")
        faiss.normalize_L2(embeddings_np)  # Cosine similarity = L2 normalized dot product

        # Build Index
        # Inner Product (dot product) on normalized vectors = Cosine Similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings_np)

        self.clip_ids = valid_clip_ids
        logger.info(f"Semantic index built with {len(self.clip_ids)} vectors.")

    def find_best_match(
        self, text_query: str, k: int = 5, exclude_ids: list[int] | None = None
    ) -> list[tuple[int, float]]:
        """
        Find best matching clips for a text query.

        Args:
            text_query: Search text (e.g. "energetic dancing")
            k: Top K results
            exclude_ids: Optional list of clip IDs to exclude

        Returns:
            List of (clip_id, score) tuples. Score is cosine similarity (0-1).
        """
        if self.index is None or not self.index.is_trained:
            return []

        # 1. Get text embedding
        query_vector = self.semantic_analyzer.get_text_embedding(text_query)
        if query_vector is None:
            return []

        query_np = np.array([query_vector]).astype("float32")
        faiss.normalize_L2(query_np)

        # 2. Search FAISS
        # Search for more than k to handle exclusions
        search_k = min(k + len(exclude_ids or []) + 5, len(self.clip_ids))
        scores, indices = self.index.search(query_np, search_k)

        results = []
        exclude_set = set(exclude_ids or [])

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            clip_id = self.clip_ids[idx]
            if clip_id in exclude_set:
                continue

            results.append((clip_id, float(score)))
            if len(results) >= k:
                break

        return results

    def is_ready(self) -> bool:
        return self.index is not None and self.index.ntotal > 0
