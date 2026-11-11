# dj_inbox/db/load.py
"""Copy one company's email chunks and their (cached) embeddings into Postgres.

    python -m dj_inbox.db.load

Uses the vectors already cached in data/embeddings (no embedding calls if the cache is complete).
Safe to run again: rows are upserted on (company_id, chunk_id).
"""
from __future__ import annotations

import time

import numpy as np
import psycopg

from dj_inbox.rag.chunk import Chunk

UPSERT = """
    insert into email_chunks (company_id, chunk_id, thread_id, message_id, sent_at, text, embedding)
    values (%s, %s, %s, %s, %s, %s, %s)
    on conflict (company_id, chunk_id) do update
       set thread_id = excluded.thread_id, message_id = excluded.message_id, sent_at = excluded.sent_at,
           text = excluded.text, embedding = excluded.embedding
"""


def load_chunks(conn: psycopg.Connection, company_id: str, chunks: list[Chunk], vectors: np.ndarray) -> int:
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
    rows = [(company_id, c.chunk_id, c.thread_id, c.message_id, c.sent_at, c.text, v)
            for c, v in zip(chunks, vectors, strict=True)]
    with conn.cursor() as cur:
        cur.executemany(UPSERT, rows)
    return len(rows)


def main() -> None:
    from dj_inbox.agent.store import INBOX, NORTHSHORE, VECTORS
    from dj_inbox.db import connect
    from dj_inbox.db.migrate import migrate
    from dj_inbox.gcp import gemini_client
    from dj_inbox.ingest import read_mbox
    from dj_inbox.rag.chunk import chunk_inbox
    from dj_inbox.rag.embed import Embedder

    start = time.perf_counter()
    chunks = chunk_inbox(read_mbox(INBOX, NORTHSHORE))
    embedder = Embedder(gemini_client(), cache_dir=VECTORS)
    vectors = embedder.embed([c.text for c in chunks], "RETRIEVAL_DOCUMENT")
    with connect(vectors=False) as conn:
        migrate(conn)
    with connect() as conn:
        n = load_chunks(conn, "northshore", chunks, vectors)
        total = conn.execute("select count(*) from email_chunks where company_id = %s", ("northshore",)).fetchone()[0]
    print(f"upserted {n} chunks ({embedder.calls} embedding calls); northshore now has {total} rows "
          f"[{time.perf_counter() - start:.1f}s]")


if __name__ == "__main__":
    main()
