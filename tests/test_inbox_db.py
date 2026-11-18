# tests/test_inbox_db.py
"""The per-company inbox tables (db/migrations/005_inbox_tables.sql), tested as the roles that use them.

app_ingest (the import worker) may write, but only rows of the company set for the transaction.
app_api (the public API) may only read, and only its company's rows.

Runs only when TEST_DATABASE_URL is set (CI's pgvector job). Each test works in its own temporary schema.
Never point it at production: it needs a superuser to switch roles.
"""
from __future__ import annotations

import json
import os
import uuid

import psycopg
import pytest

from dj_inbox.db import CONNECT_KW, safe_connect
from dj_inbox.db.migrate import migrate

URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="needs TEST_DATABASE_URL (a throwaway Postgres with pgvector)")

LEAD = json.dumps({"event_type": "wedding", "outcome": "open"})


def add_thread(c, company: str, thread: str, message: str, gmail_id: str | None = None) -> None:
    c.execute("insert into threads (company_id, thread_id, subject) values (%s, %s, 'Wedding inquiry')",
              (company, thread))
    c.execute("""insert into emails (company_id, message_id, gmail_id, thread_id, sent_at, sender_role, sender_name,
                                     sender_email, subject, body_raw, body_clean)
                 values (%s, %s, %s, %s, now(), 'client', 'Kim', 'kim@x.example', 'Wedding inquiry', 'hi', 'hi')""",
              (company, message, gmail_id, thread))
    c.execute("""insert into leads (company_id, thread_id, lead_index, fields, source, model, prompt_version)
                 values (%s, %s, 0, %s, 'email', 'm', 'v2')""", (company, thread, LEAD))


@pytest.fixture
def conn():
    """Companies co-a and co-b with one thread each, loaded as the superuser; tests then switch roles."""
    schema = f"test_{uuid.uuid4().hex[:10]}"
    with safe_connect(URL, **CONNECT_KW) as c:
        c.execute("create extension if not exists vector")
        c.execute(f"create schema {schema}")
        c.execute(f"set search_path = {schema}, public, extensions")
        migrate(c)
        c.execute("insert into companies (company_id, name) values ('co-a', 'A'), ('co-b', 'B')")
        add_thread(c, "co-a", "ta", "<a1@x>", "ga1")
        add_thread(c, "co-b", "tb", "<b1@x>", "gb1")
        try:
            yield c
        finally:
            c.execute("reset role")
            c.execute(f"drop schema {schema} cascade")


def act_as(c, role: str, company: str | None) -> None:
    """Switch role and set app.company_id for the session (tests run in autocommit, one statement each)."""
    c.execute(f"set role {role}")
    c.execute("select set_config('app.company_id', %s, false)", (company or "",))


def count(c, table: str) -> int:
    return c.execute(f"select count(*) from {table}").fetchone()[0]


def test_ingest_role_cannot_skip_the_policies(conn):
    row = conn.execute("select rolsuper, rolbypassrls from pg_roles where rolname = 'app_ingest'").fetchone()
    assert row == (False, False)


def test_ingest_writes_and_reads_its_own_company(conn):
    act_as(conn, "app_ingest", "co-a")
    add_thread(conn, "co-a", "ta2", "<a2@x>", "ga2")
    assert count(conn, "threads") == 2 and count(conn, "emails") == 2 and count(conn, "leads") == 2


def test_ingest_cannot_write_into_another_company(conn):
    act_as(conn, "app_ingest", "co-a")
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        conn.execute("insert into threads (company_id, thread_id, subject) values ('co-b', 'evil', 's')")


def test_ingest_cannot_move_its_rows_to_another_company(conn):
    act_as(conn, "app_ingest", "co-a")
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        conn.execute("update leads set company_id = 'co-b'")


def test_ingest_cannot_see_change_or_delete_another_company(conn):
    act_as(conn, "app_ingest", "co-a")
    assert conn.execute("select count(*) from emails where company_id = 'co-b'").fetchone()[0] == 0
    assert conn.execute("update emails set body_raw = 'x' where company_id = 'co-b'").rowcount == 0
    assert conn.execute("delete from threads where company_id = 'co-b'").rowcount == 0
    conn.execute("reset role")
    assert conn.execute("select body_raw from emails where company_id = 'co-b'").fetchone()[0] == "hi"


def test_ingest_with_no_company_set_sees_and_writes_nothing(conn):
    """Fail closed: a worker that forgot set_config can't read or write anything."""
    act_as(conn, "app_ingest", None)
    assert count(conn, "threads") == 0
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        conn.execute("insert into threads (company_id, thread_id, subject) values ('co-a', 'x', 's')")


def test_api_reads_only_its_company_and_cannot_write(conn):
    act_as(conn, "app_api", "co-b")
    assert [r[0] for r in conn.execute("select thread_id from threads")] == ["tb"]
    assert [r[0] for r in conn.execute("select message_id from emails")] == ["<b1@x>"]
    assert count(conn, "leads") == 1
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        conn.execute("insert into threads (company_id, thread_id, subject) values ('co-b', 'x', 's')")


def test_the_same_email_cannot_be_imported_twice(conn):
    """Dedupe on both ids: a replayed Gmail notification or a re-run import must not create a duplicate."""
    with pytest.raises(psycopg.errors.UniqueViolation):
        add_thread(conn, "co-a", "ta", "<a1@x>")                       # same thread / Message-ID
    with pytest.raises(psycopg.errors.UniqueViolation):
        conn.execute("""insert into emails (company_id, message_id, gmail_id, thread_id, sent_at, sender_role,
                        sender_name, sender_email, subject, body_raw, body_clean)
                        values ('co-a', '<other@x>', 'ga1', 'ta', now(), 'client', 'K', 'k@x', 's', 'b', 'b')""")


def test_deleting_a_company_deletes_its_inbox(conn):
    """Disconnect = delete data: removing the company removes its threads, emails and leads."""
    conn.execute("delete from companies where company_id = 'co-a'")
    assert conn.execute("select count(*) from emails where company_id = 'co-a'").fetchone()[0] == 0
    assert conn.execute("select count(*) from leads where company_id = 'co-a'").fetchone()[0] == 0
    assert conn.execute("select count(*) from emails where company_id = 'co-b'").fetchone()[0] == 1
