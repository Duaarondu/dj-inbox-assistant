# tests/test_pgvector.py
"""pgvector search against a real Postgres with the vector extension: the spec for dj_inbox/rag/pg_search.py.

Runs only when TEST_DATABASE_URL is set (CI's pgvector job sets it). Each test works inside its own temporary
schema and drops it afterwards, so it never touches the real tables, but still: point it at a throwaway
database, not at production.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime

import numpy as np
import pytest

from dj_inbox.db import CONNECT_KW, safe_connect
from dj_inbox.db.load import load_chunks
from dj_inbox.db.migrate import migrate
from dj_inbox.rag.chunk import Chunk
from dj_inbox.rag.pg_search import top_threads_pg
from dj_inbox.rag.retrieve import top_threads

URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="needs TEST_DATABASE_URL (a throwaway Postgres with pgvector)")
DIM = 768


@pytest.fixture
def conn():
    schema = f"test_{uuid.uuid4().hex[:10]}"
    with safe_connect(URL, **CONNECT_KW) as c:
        c.execute("create extension if not exists vector")
        c.execute(f"create schema {schema}")
        c.execute(f"set search_path = {schema}, public, extensions")   # Supabase keeps pgvector in `extensions`
        from pgvector.psycopg import register_vector
        register_vector(c)
        migrate(c)
        try:
            yield c
        finally:
            c.execute(f"drop schema {schema} cascade")


def make_chunks(prefix: str, threads: int, per_thread: int) -> list[Chunk]:
    return [Chunk(chunk_id=f"{prefix}{t}-{i}", thread_id=f"{prefix}{t}", message_id=f"<{prefix}{t}-{i}@x>",
                  sent_at=datetime(2026, 5, 1, 10, 0), text=f"email {i} of thread {prefix}{t}")
            for t in range(threads) for i in range(per_thread)]


def test_migrate_creates_table_with_rls_and_is_idempotent(conn):
    assert migrate(conn) == []                                                 # second run: nothing to do
    rls = conn.execute("select relrowsecurity from pg_class where oid = 'email_chunks'::regclass").fetchone()[0]
    assert rls is True                                                         # Supabase's REST API sees nothing
    indexes = {r[0] for r in conn.execute("select indexname from pg_indexes where tablename = 'email_chunks'")}
    assert "email_chunks_embedding_hnsw" in indexes


def test_load_is_an_upsert(conn):
    chunks = make_chunks("a", threads=3, per_thread=2)
    vecs = np.random.default_rng(0).normal(size=(len(chunks), DIM)).astype(np.float32)
    assert load_chunks(conn, "co-a", chunks, vecs) == 6
    chunks[0] = chunks[0].model_copy(update={"text": "edited"})
    load_chunks(conn, "co-a", chunks, vecs)
    assert conn.execute("select count(*) from email_chunks").fetchone()[0] == 6
    assert conn.execute("select text from email_chunks where chunk_id = 'a0-0'").fetchone()[0] == "edited"


def test_same_threads_as_the_numpy_search(conn):
    """pgvector must reproduce the NumPy small-to-big search exactly (same data, same question vectors)."""
    rng = np.random.default_rng(7)
    chunks = make_chunks("a", threads=40, per_thread=3)
    vecs = rng.normal(size=(len(chunks), DIM)).astype(np.float32)
    load_chunks(conn, "co-a", chunks, vecs)
    for _ in range(10):
        q = rng.normal(size=DIM).astype(np.float32)
        assert top_threads_pg(conn, "co-a", q, k=5) == top_threads(q, vecs, chunks, 5)


def test_never_returns_another_companys_emails(conn):
    """Company B's chunks are planted EXACTLY on the question vector, so they'd win if the filter were missing."""
    rng = np.random.default_rng(3)
    a_chunks = make_chunks("a", threads=10, per_thread=2)
    load_chunks(conn, "co-a", a_chunks, rng.normal(size=(len(a_chunks), DIM)).astype(np.float32))
    q = rng.normal(size=DIM).astype(np.float32)
    b_chunks = make_chunks("b", threads=10, per_thread=2)
    load_chunks(conn, "co-b", b_chunks, np.tile(q, (len(b_chunks), 1)))
    got = top_threads_pg(conn, "co-a", q, k=5)
    assert len(got) == 5 and all(t.startswith("a") for t in got)
    assert top_threads_pg(conn, "co-b", q, k=3)[0].startswith("b")
    assert top_threads_pg(conn, "nobody", q, k=5) == []


def test_k_larger_than_the_number_of_threads(conn):
    chunks = make_chunks("a", threads=2, per_thread=3)
    load_chunks(conn, "co-a", chunks, np.random.default_rng(1).normal(size=(6, DIM)).astype(np.float32))
    assert sorted(top_threads_pg(conn, "co-a", np.ones(DIM, dtype=np.float32), k=5)) == ["a0", "a1"]
