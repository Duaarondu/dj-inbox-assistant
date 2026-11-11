# dj_inbox/db/__init__.py
"""Postgres connections (Supabase in production, a throwaway local Postgres in tests).

DATABASE_URL comes from .env locally and from Secret Manager on Cloud Run. Never print or log it: it contains
the database password.
"""
from __future__ import annotations

import os

import psycopg
from pgvector.psycopg import register_vector

from dj_inbox.gcp import ROOT

# prepare_threshold=None: no server-side prepared statements, which connection poolers (Supabase's pooler)
# can't always route back to the same database session.
CONNECT_KW = {"autocommit": True, "prepare_threshold": None}


def database_url(env: str = "DATABASE_URL") -> str:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    url = os.environ.get(env)
    if not url:
        raise SystemExit(f"Set {env} in .env (Supabase: Connect -> Connection string -> Session pooler).")
    return url


class DatabaseConnectError(RuntimeError):
    """Connection failed. The driver's own message is dropped on purpose: for a malformed URL it quotes the
    password (seen Nov 11), and this error can end up in logs."""


def safe_connect(url: str, **kwargs) -> psycopg.Connection:
    try:
        return psycopg.connect(url, **kwargs)
    except psycopg.Error as e:
        hint = ("DATABASE_URL could not be parsed: remove [brackets] around the password, and use only letters "
                "and digits in it (or percent-encode the special characters)"
                if isinstance(e, psycopg.ProgrammingError) else "check the host, port and password in DATABASE_URL")
        raise DatabaseConnectError(f"Could not connect to the database ({type(e).__name__}): {hint}.") from None


def connect(url: str | None = None, *, vectors: bool = True) -> psycopg.Connection:
    """vectors=True teaches the connection the `vector` type (needs the extension to exist already)."""
    conn = safe_connect(url or database_url(), **CONNECT_KW)
    if vectors:
        register_vector(conn)
    return conn


def make_pool(url: str | None = None, max_size: int = 4):
    """A small connection pool for the API: opening a TLS connection per question would add ~100 ms."""
    from psycopg_pool import ConnectionPool

    url = url or database_url()
    safe_connect(url, **CONNECT_KW).close()     # fail fast with a safe message; the pool would log the raw one
    return ConnectionPool(url, min_size=1, max_size=max_size, kwargs=CONNECT_KW,
                          configure=register_vector, open=True)
