# dj_inbox/rag/search.py
"""Step 3 of RAG: find the chunks whose meaning is closest to the question.

Every chunk and every question is a vector (a list of ~768 numbers) from the embedding model.
Similar meaning -> vectors pointing in a similar direction. Cosine similarity measures that direction:

    cos(a, b) = (a . b) / (|a| * |b|)          1 = same direction, 0 = unrelated, -1 = opposite

With a matrix of all chunk vectors (one row each), one matrix-vector product scores EVERY chunk at once.
That's all a "vector database" does at its core; pgvector does the same thing inside Postgres later.

tests/test_search.py is the spec.
"""
from __future__ import annotations

import numpy as np


def cosine_top_k(query: np.ndarray, matrix: np.ndarray, k: int) -> list[int]:
    """Row indices of the k rows of `matrix` most cosine-similar to `query`, best first.

    query:  shape (d,)       one vector
    matrix: shape (n, d)     one row per chunk
    The vectors are NOT guaranteed to be normalized, so divide by the lengths (np.linalg.norm).
    If k > n, return all n indices.

    Hints: matrix @ query gives n dot products. np.linalg.norm(matrix, axis=1) gives n row lengths.
           np.argsort(scores)[::-1] sorts indices from highest score to lowest.
    """
    dots = matrix @ query                                           # n dot products, one per chunk
    lengths = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)  # |chunk| * |query| for each chunk
    scores = dots / (lengths + 1e-12)                               # cosine; 1e-12 avoids dividing by zero
    best_first = np.argsort(scores)[::-1]                           # indices from highest score to lowest
    return [int(i) for i in best_first[:k]]                         # slicing past the end just returns all n
