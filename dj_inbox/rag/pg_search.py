# dj_inbox/rag/pg_search.py
"""Step 3 of RAG, inside Postgres: pgvector does what cosine_top_k (search.py) does with NumPy.

    cosine_top_k:   scores = (matrix @ q) / (|rows| * |q|),  sort high -> low,  take the top `pool`
    pgvector:       ORDER BY embedding <=> q  LIMIT pool

`<=>` is cosine DISTANCE = 1 - cosine similarity, so the closest rows come first with ORDER BY ... ASC.
The HNSW index (db/migrations/001_email_chunks.sql) makes that ORDER BY fast on millions of rows by
looking at an approximate neighbourhood instead of every row.

Small-to-big stays the same as dj_inbox/rag/retrieve.py: take the best `pool` emails, keep the first k
DIFFERENT threads they belong to.

tests/test_pgvector.py is the spec (it needs a database: see that file).
"""
from __future__ import annotations

import numpy as np
import psycopg

POOL = 50          # emails to look through, same as retrieve.POOL
EF_SEARCH = 100    # how many candidates the HNSW index examines. Must be >= POOL: an index scan never returns
                   # more rows than ef_search (default 40), so LIMIT 50 would silently give back only 40.

# YOUR PART: one SELECT that returns the thread_id of this company's `pool` closest chunks, closest first.
#   - parameters are passed by name: %(company_id)s, %(qvec)s, %(pool)s
#   - filter on company_id (non-negotiable: another company's emails must never come back)
#   - order by cosine distance to the question vector, nearest first
#   - limit to pool rows
SEARCH_SQL = """
    select thread_id
      from email_chunks
     where company_id = %(company_id)s
     order by embedding <=> %(qvec)s
     limit %(pool)s
"""


def top_threads_pg(conn: psycopg.Connection, company_id: str, qvec: np.ndarray, k: int, pool: int = POOL) -> list[str]:
    """Thread ids of the k best threads for this company, best first. Same contract as retrieve.top_threads."""
    with conn.transaction():
        conn.execute(f"set local hnsw.ef_search = {int(EF_SEARCH)}")      # only for this transaction
        rows = conn.execute(SEARCH_SQL, {"company_id": company_id, "qvec": qvec, "pool": pool}).fetchall()
    thread_ids: list[str] = []
    for (thread_id,) in rows:
        if thread_id not in thread_ids:          # keep each thread the first time it appears
            thread_ids.append(thread_id)
        if len(thread_ids) == k:
            break
    return thread_ids
