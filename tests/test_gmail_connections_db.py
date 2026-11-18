# tests/test_gmail_connections_db.py
"""db/migrations/006_gmail_connections.sql, tested as the roles that use it.

The point: only a company's OWNER can connect Gmail, only to their own company; the public API can never
read the encrypted token; the worker can read it, but only for the company of the current transaction.

Runs only when TEST_DATABASE_URL is set (CI's pgvector job). Each test works in its own temporary schema.
"""
from __future__ import annotations

import os
import uuid

import psycopg
import pytest

from dj_inbox.db import CONNECT_KW, safe_connect
from dj_inbox.db.migrate import migrate

URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="needs TEST_DATABASE_URL (a throwaway Postgres with pgvector)")
OWNER_A, STAFF_A, OWNER_B = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()


@pytest.fixture
def conn():
    schema = f"test_{uuid.uuid4().hex[:10]}"
    with safe_connect(URL, **CONNECT_KW) as c:
        c.execute("create extension if not exists vector")
        c.execute(f"create schema {schema}")
        c.execute(f"set search_path = {schema}, public, extensions")
        migrate(c)
        c.execute("insert into companies (company_id, name) values ('co-a', 'A'), ('co-b', 'B')")
        c.execute("""insert into company_members (user_id, company_id, role)
                     values (%s, 'co-a', 'owner'), (%s, 'co-a', 'staff'), (%s, 'co-b', 'owner')""",
                  (OWNER_A, STAFF_A, OWNER_B))
        try:
            yield c
        finally:
            c.execute("reset role")
            c.execute(f"drop schema {schema} cascade")


def as_api(c, user, company) -> None:
    c.execute("reset role")
    c.execute("set role app_api")
    c.execute("select set_config('app.user_id', %s, false)", (str(user) if user else "",))
    c.execute("select set_config('app.company_id', %s, false)", (company or "",))


def connect(c, address="owner@north.example", token=b"\x01ciphertext") -> None:
    c.execute("select save_gmail_connection(%s, %s, 'projects/p/keys/k', '123')", (address, token))


def test_owner_connects_their_own_company(conn):
    as_api(conn, OWNER_A, "co-a")
    connect(conn)
    rows = conn.execute("select company_id, gmail_address, status from gmail_connections").fetchall()
    assert rows == [("co-a", "owner@north.example", "importing")]


def test_staff_cannot_connect(conn):
    as_api(conn, STAFF_A, "co-a")
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        connect(conn)


def test_owner_of_b_cannot_connect_into_a(conn):
    """Even with app.company_id pointed at someone else's company, the membership check refuses."""
    as_api(conn, OWNER_B, "co-a")
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        connect(conn)


def test_not_signed_in_cannot_connect(conn):
    as_api(conn, None, "co-a")
    with pytest.raises(psycopg.errors.InvalidAuthorizationSpecification):
        connect(conn)


def test_api_can_never_read_the_token_or_write_directly(conn):
    as_api(conn, OWNER_A, "co-a")
    connect(conn)
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        conn.execute("select token_ciphertext from gmail_connections")
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        conn.execute("select * from gmail_connections")
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        conn.execute("update gmail_connections set status = 'ready'")
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        conn.execute("""insert into gmail_connections (company_id, gmail_address, token_ciphertext, kms_key, status,
                        connected_by) values ('co-a', 'x@y', '\\x00', 'k', 'ready', %s)""", (OWNER_A,))


def test_api_sees_only_its_own_companys_status(conn):
    as_api(conn, OWNER_A, "co-a")
    connect(conn, "a@north.example")
    as_api(conn, OWNER_B, "co-b")
    connect(conn, "b@south.example")
    assert conn.execute("select gmail_address from gmail_connections").fetchall() == [("b@south.example",)]


def test_one_mailbox_cannot_be_connected_to_two_companies(conn):
    as_api(conn, OWNER_A, "co-a")
    connect(conn, "shared@gmail.com")
    as_api(conn, OWNER_B, "co-b")
    with pytest.raises(psycopg.errors.UniqueViolation):
        connect(conn, "Shared@Gmail.com")


def test_reconnecting_replaces_the_token_and_restarts_the_import(conn):
    as_api(conn, OWNER_A, "co-a")
    connect(conn, token=b"old")
    conn.execute("reset role")
    conn.execute("update gmail_connections set status = 'error', error = 'token expired'")
    as_api(conn, OWNER_A, "co-a")
    connect(conn, token=b"new")
    conn.execute("reset role")
    assert conn.execute("select token_ciphertext, status, error from gmail_connections").fetchone() == (
        b"new", "importing", None)


def test_worker_reads_the_token_of_its_company_only(conn):
    as_api(conn, OWNER_A, "co-a")
    connect(conn, "a@north.example", b"token-a")
    as_api(conn, OWNER_B, "co-b")
    connect(conn, "b@south.example", b"token-b")
    conn.execute("reset role")
    conn.execute("set role app_ingest")
    conn.execute("select set_config('app.company_id', 'co-a', false)")
    assert conn.execute("select token_ciphertext from gmail_connections").fetchall() == [(b"token-a",)]
    assert conn.execute("update gmail_connections set status = 'ready' where company_id = 'co-b'").rowcount == 0
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        conn.execute("update gmail_connections set company_id = 'co-b'")


def test_deleting_the_company_deletes_the_connection(conn):
    as_api(conn, OWNER_A, "co-a")
    connect(conn)
    conn.execute("reset role")
    conn.execute("delete from companies where company_id = 'co-a'")
    assert conn.execute("select count(*) from gmail_connections").fetchone()[0] == 0


def test_the_api_helpers_map_database_refusals(conn):
    """save_connection_in_db / connection_status, as app_api, the way the API calls them."""
    from dj_inbox.gmail.connect import MailboxTaken, NotOwner, connection_status, save_connection_in_db

    as_api(conn, None, None)
    save_connection_in_db(conn, str(OWNER_A), "co-a", "a@north.example", b"x", "k", "1")
    assert connection_status(conn, "co-a")["status"] == "importing"
    assert connection_status(conn, "co-b") is None
    with pytest.raises(NotOwner):
        save_connection_in_db(conn, str(STAFF_A), "co-a", "a@north.example", b"x", "k", "1")
    with pytest.raises(MailboxTaken):
        save_connection_in_db(conn, str(OWNER_B), "co-b", "a@north.example", b"x", "k", "1")
