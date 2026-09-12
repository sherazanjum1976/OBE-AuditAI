"""
vector_store.py
---------------
A dynamic, in-memory FAISS index built fresh for every user session.
No external/paid vector database is required, keeping the app suitable
for free Streamlit Cloud deployment.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
import faiss

from modules.pdf_processor import DocChunk


class FaissVectorStore:
    """Simple wrapper around a FAISS flat inner-product (cosine, since
    embeddings are L2-normalized) index, with parallel chunk metadata."""

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: List[DocChunk] = []

    def add(self, vectors: np.ndarray, chunks: List[DocChunk]):
        if vectors.shape[0] == 0:
            return
        assert vectors.shape[0] == len(chunks), "Vector/chunk count mismatch"
        self.index.add(vectors)
        self.chunks.extend(chunks)

    @property
    def size(self) -> int:
        return len(self.chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[DocChunk, float]]:
        if self.size == 0:
            return []
        top_k = min(top_k, self.size)
        scores, indices = self.index.search(query_vector, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results


def build_vector_store(chunks: List[DocChunk], vectors: np.ndarray) -> FaissVectorStore:
    dim = vectors.shape[1] if vectors.shape[0] > 0 else 384
    store = FaissVectorStore(dim=dim)
    store.add(vectors, chunks)
    return store
