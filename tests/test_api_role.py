# tests/test_api_role.py
"""dj_inbox/db/api_role.py: the SCRAM hash, the app_api URL, and .env writing."""
from __future__ import annotations

import os
import uuid
from urllib.parse import unquote, urlsplit

import psycopg
import pytest
from psycopg import sql

from dj_inbox.db import CONNECT_KW, safe_connect
from dj_inbox.db.api_role import api_url, scram_sha256, write_env_var

URL = os.environ.get("TEST_DATABASE_URL")


def test_scram_format_and_random_salt():
    a, b = scram_sha256("same-password"), scram_sha256("same-password")
    assert a.startswith("SCRAM-SHA-256$4096:") and a != b       # new salt every time


def test_api_url_swaps_only_user_and_password_for_the_pooler():
    owner = "postgresql://postgres.abcdefgh:ownerpw@aws-1-us-east-1.pooler.supabase.com:5432/postgres?sslmode=require"
    url = api_url(owner, "new/pw+x")
    parts = urlsplit(url)
    assert parts.username == "app_api.abcdefgh"
    assert unquote(parts.password) == "new/pw+x"
    assert (parts.hostname, parts.port, parts.path, parts.query) == (
        "aws-1-us-east-1.pooler.supabase.com", 5432, "/postgres", "sslmode=require")
    assert "ownerpw" not in url


def test_api_url_plain_postgres_user():
    assert urlsplit(api_url("postgresql://postgres:pw@localhost:5432/db", "x")).username == "app_api"


def test_write_env_var_replaces_and_keeps_other_lines(tmp_path):
    env = tmp_path / ".env"
    env.write_text("DATABASE_URL=owner\nDATABASE_API_URL=old\nOTHER=1\n")
    write_env_var(env, "DATABASE_API_URL", "new")
    lines = env.read_text().splitlines()
    assert lines.count("DATABASE_API_URL=new") == 1 and "DATABASE_API_URL=old" not in lines
    assert "DATABASE_URL=owner" in lines and "OTHER=1" in lines


@pytest.mark.skipif(not URL, reason="needs TEST_DATABASE_URL (a throwaway Postgres that checks passwords)")
def test_postgres_accepts_the_hashed_password_and_rejects_a_wrong_one():
    """End to end: the verifier we compute in Python must let the real password log in, and only it."""
    role, password = f"test_login_{uuid.uuid4().hex[:8]}", "correct-horse-battery-staple"
    with safe_connect(URL, **CONNECT_KW) as admin:
        admin.execute(sql.SQL("create role {} login password {}").format(
            sql.Identifier(role), sql.Literal(scram_sha256(password))))
        try:
            stored = admin.execute("select rolpassword from pg_authid where rolname = %s", (role,)).fetchone()[0]
            assert stored.startswith("SCRAM-SHA-256$4096:")          # stored as sent, not re-hashed
            parts = urlsplit(URL)
            base = f"postgresql://{role}:{{pw}}@{parts.hostname}:{parts.port or 5432}{parts.path or '/postgres'}"
            with psycopg.connect(base.format(pw=password)) as c:
                assert c.execute("select current_user").fetchone()[0] == role
            with pytest.raises(psycopg.OperationalError):
                psycopg.connect(base.format(pw="wrong-password")).close()
        finally:
            admin.execute(sql.SQL("drop role {}").format(sql.Identifier(role)))
