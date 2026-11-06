# dj_inbox/rag/retrieve.py
"""Search small, return big: find the best-matching EMAILS, return the first k different THREADS they belong to.

Why (measured Nov 5): a question matches the client's email (names, event), but the answer usually lives in
OUR reply (prices, confirmed dates). Returning single emails: recall@5 41%. Returning their threads: 98%.
"""
from __future__ import annotations

import numpy as np

from dj_inbox.rag.chunk import Chunk
from dj_inbox.rag.search import cosine_top_k

POOL = 50  # emails to look through; several top emails often belong to the same thread


def top_threads(qvec: np.ndarray, matrix: np.ndarray, chunks: list[Chunk], k: int, pool: int = POOL) -> list[str]:
    """Thread ids of the k best threads, best first. matrix row i is the vector of chunks[i]."""
    thread_ids: list[str] = []
    for i in cosine_top_k(qvec, matrix, pool):
        tid = chunks[i].thread_id
        if tid not in thread_ids:          # keep each thread the first time it appears
            thread_ids.append(tid)
        if len(thread_ids) == k:           # stop once we have k different threads
            break
    return thread_ids
