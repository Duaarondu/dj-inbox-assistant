# tests/test_search.py
"""The spec for cosine_top_k. Make these pass; don't edit them to make them pass."""
import numpy as np

from dj_inbox.rag.search import cosine_top_k

M = np.array([
    [1.0, 0.0, 0.0],     # 0: points along x
    [0.0, 1.0, 0.0],     # 1: along y
    [0.9, 0.1, 0.0],     # 2: almost x
    [-1.0, 0.0, 0.0],    # 3: opposite of x
], dtype=np.float32)


def test_best_first():
    assert cosine_top_k(np.array([1.0, 0.0, 0.0]), M, k=3) == [0, 2, 1]


def test_uses_direction_not_length():
    """A long vector isn't 'more similar'. With raw dot products, row 1 = [0, 100, 0] would score 10 and win;
    by direction it's nearly unrelated to the query, and row 2 = [0.9, 0.1, 0] points almost exactly the same way."""
    long = M.copy()
    long[1] = [0.0, 100.0, 0.0]
    assert cosine_top_k(np.array([1.0, 0.1, 0.0]), long, k=1) == [2]


def test_k_larger_than_rows_returns_all():
    assert sorted(cosine_top_k(np.array([0.0, 0.0, 1.0]), M, k=10)) == [0, 1, 2, 3]
    assert len(cosine_top_k(np.array([1.0, 0.0, 0.0]), M, k=10)) == 4


def test_returns_plain_ints():
    result = cosine_top_k(np.array([1.0, 0.0, 0.0]), M, k=2)
    assert isinstance(result, list) and all(isinstance(i, int) for i in result)
