"""
embeddings.py
-------------
Thin wrapper around a Sentence Transformer model used to embed document
chunks and user/system queries for semantic retrieval. Cached via
Streamlit's resource cache so the model loads only once per session.
"""

from __future__ import annotations

from typing import List
import numpy as np

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None  # lazy singleton


def get_embedder():
    """Lazily load and cache the sentence-transformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of strings -> (n, dim) float32 numpy array, L2-normalized."""
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    model = get_embedder()
    vectors = model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=32,
    )
    return vectors.astype("float32")


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string -> (1, dim) float32 numpy array."""
    return embed_texts([query])
