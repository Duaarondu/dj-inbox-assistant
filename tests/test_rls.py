# tests/test_rls.py
"""The database-level company lock (db/migrations/002_companies_rls.sql), tested as the API's own role.

Every test switches to `app_api` (the login the API will use) and checks what Postgres lets it see.
The point: even a query with NO "where company_id = ..." returns only the company set for this
transaction, and a request that sets nothing sees nothing.

Runs only when TEST_DATABASE_URL is set (CI's pgvector job). Each test works in its own temporary schema.
Never point it at production: it needs a superuser to switch roles.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime

import numpy as np
import psycopg
import pytest

from dj_inbox.db import CONNECT_KW, safe_connect
from dj_inbox.db.load import load_chunks
from dj_inbox.db.migrate import migrate
from dj_inbox.rag.chunk import Chunk
from dj_inbox.rag.pg_search import top_threads_pg

URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="needs TEST_DATABASE_URL (a throwaway Postgres with pgvector)")
DIM = 768
USER_A, USER_B = uuid.uuid4(), uuid.uuid4()


def make_chunks(prefix: str, threads: int, per_thread: int) -> list[Chunk]:
    return [Chunk(chunk_id=f"{prefix}{t}-{i}", thread_id=f"{prefix}{t}", message_id=f"<{prefix}{t}-{i}@x>",
                  sent_at=datetime(2026, 5, 1, 10, 0), text=f"email {i} of thread {prefix}{t}")
            for t in range(threads) for i in range(per_thread)]


@pytest.fixture
def conn():
    """Two companies with 6 emails each and one owner each, loaded as the superuser; tests then act as app_api."""
    schema = f"test_{uuid.uuid4().hex[:10]}"
    with safe_connect(URL, **CONNECT_KW) as c:
        c.execute("create extension if not exists vector")
        c.execute(f"create schema {schema}")
        c.execute(f"set search_path = {schema}, public, extensions")
        from pgvector.psycopg import register_vector
        register_vector(c)
        migrate(c)
        rng = np.random.default_rng(3)
        for company, prefix in (("co-a", "a"), ("co-b", "b")):
            chunks = make_chunks(prefix, threads=3, per_thread=2)
            load_chunks(c, company, chunks, rng.normal(size=(len(chunks), DIM)).astype(np.float32))
        c.execute("insert into companies (company_id, name) values ('co-a', 'A'), ('co-b', 'B')")
        c.execute("insert into company_members (user_id, company_id, role) values (%s, 'co-a', 'owner'), "
                  "(%s, 'co-b', 'owner')", (USER_A, USER_B))
        c.execute("set role app_api")
        try:
            yield c
        finally:
            c.execute("reset role")
            c.execute(f"drop schema {schema} cascade")


def scalars(conn, sql: str, params=None) -> list:
    return [row[0] for row in conn.execute(sql, params).fetchall()]


def test_app_api_cannot_skip_the_policies(conn):
    row = conn.execute("select rolsuper, rolbypassrls from pg_roles where rolname = 'app_api'").fetchone()
    assert row == (False, False)
    assert conn.execute("select current_user").fetchone()[0] == "app_api"


def test_nothing_is_visible_when_no_company_is_set(conn):
    """Fail closed: a request that forgets set_config sees no rows at all."""
    assert scalars(conn, "select count(*) from email_chunks") == [0]
    assert scalars(conn, "select count(*) from companies") == [0]


def test_only_the_set_company_even_without_a_where_clause(conn):
    with conn.transaction():
        conn.execute("select set_config('app.company_id', 'co-a', true)")
        assert scalars(conn, "select distinct company_id from email_chunks") == ["co-a"]
        assert scalars(conn, "select count(*) from email_chunks") == [6]
        assert scalars(conn, "select company_id from companies") == ["co-a"]


def test_an_unknown_company_sees_nothing(conn):
    with conn.transaction():
        conn.execute("select set_config('app.company_id', 'co-z', true)")
        assert scalars(conn, "select count(*) from email_chunks") == [0]


def test_the_setting_ends_with_the_transaction(conn):
    """Why set_config(..., true): the pooler reuses this connection for the next request."""
    with conn.transaction():
        conn.execute("select set_config('app.company_id', 'co-a', true)")
        assert scalars(conn, "select count(*) from email_chunks") == [6]
    assert scalars(conn, "select count(*) from email_chunks") == [0]       # next "request" on the same connection


def test_a_query_that_forgets_the_filter_still_gets_one_company(conn):
    q = np.random.default_rng(9).normal(size=DIM).astype(np.float32)
    with conn.transaction():
        conn.execute("select set_config('app.company_id', 'co-b', true)")
        rows = scalars(conn, "select thread_id from email_chunks order by embedding <=> %s limit 50", (q,))
    assert rows and all(t.startswith("b") for t in rows)


def test_search_as_app_api_returns_only_that_companys_threads(conn):
    q = np.random.default_rng(11).normal(size=DIM).astype(np.float32)
    assert all(t.startswith("a") for t in top_threads_pg(conn, "co-a", q, k=3))
    assert all(t.startswith("b") for t in top_threads_pg(conn, "co-b", q, k=3))
    assert len(top_threads_pg(conn, "co-a", q, k=3)) == 3


def test_app_api_cannot_write(conn):
    with conn.transaction():
        conn.execute("select set_config('app.company_id', 'co-a', true)")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            with conn.transaction():
                conn.execute("delete from email_chunks")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            with conn.transaction():
                conn.execute("insert into company_members (user_id, company_id, role) values (%s, 'co-b', 'owner')",
                             (USER_A,))


def test_a_user_sees_only_their_own_memberships(conn):
    with conn.transaction():
        conn.execute("select set_config('app.user_id', %s, true)", (str(USER_A),))
        assert scalars(conn, "select company_id from company_members") == ["co-a"]
    with conn.transaction():
        conn.execute("select set_config('app.user_id', %s, true)", (str(USER_B),))
        assert scalars(conn, "select company_id from company_members") == ["co-b"]


def test_no_user_set_sees_no_memberships_and_does_not_crash(conn):
    """After a transaction used app.user_id, Postgres reads it back as '' (not NULL); ''::uuid would error."""
    with conn.transaction():
        conn.execute("select set_config('app.user_id', %s, true)", (str(USER_A),))
    assert scalars(conn, "select current_setting('app.user_id', true)") == [""]
    assert scalars(conn, "select count(*) from company_members") == [0]
