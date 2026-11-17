# tests/test_gmail_import.py
"""dj_inbox/gmail/import_inbox.py + dj_inbox/db/inbox.py on a real (throwaway) Postgres, as app_ingest.

The synthetic inbox stands in for Gmail: each mbox email becomes (message, thread id, gmail id) the way the
Gmail reader returns them. Embeddings and extraction are fakes (no model calls); they record what they got.

Runs only when TEST_DATABASE_URL is set (CI's pgvector job).
"""
from __future__ import annotations

import mailbox
import os
import uuid
from pathlib import Path

import numpy as np
import pytest

from dj_inbox.db import CONNECT_KW, safe_connect
from dj_inbox.db.inbox import load_inbox
from dj_inbox.db.migrate import migrate
from dj_inbox.extract.extract import ExtractedLead, ThreadLeads
from dj_inbox.gmail.import_inbox import import_messages
from dj_inbox.ingest import CompanyConfig, parse_raw, read_mbox

URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="needs TEST_DATABASE_URL (a throwaway Postgres with pgvector)")
MBOX = Path(__file__).resolve().parents[1] / "synthetic_inbox" / "data" / "inbox.mbox"
COMPANY = CompanyConfig(domains=["northshoresound.example"])
THREADS = {"t001", "t002", "t003", "t004", "t005"}


def as_gmail(threads: set[str]):
    """Synthetic emails of these threads as the Gmail reader returns them: (message, 'G'+thread, 'g'+n)."""
    items = []
    for n, msg in enumerate(mailbox.mbox(str(MBOX))):
        thread = str(msg["X-GM-THRID"]).strip()
        if thread in threads:
            items.append((parse_raw(msg.as_bytes()), "G" + thread, f"g{n}"))
    return items


class Fakes:
    def __init__(self):
        self.embedded: list[str] = []
        self.extracted: list[str] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        self.embedded += texts
        return np.random.default_rng(0).normal(size=(len(texts), 768)).astype(np.float32)

    def extract(self, threads) -> dict[str, ThreadLeads]:
        self.extracted += [t.thread_id for t in threads]
        return {t.thread_id: ThreadLeads(leads=[ExtractedLead(event_type="wedding", outcome="open")])
                for t in threads}


@pytest.fixture
def conn():
    schema = f"test_{uuid.uuid4().hex[:10]}"
    with safe_connect(URL, **CONNECT_KW) as c:
        c.execute("create extension if not exists vector")
        c.execute(f"create schema {schema}")
        c.execute(f"set search_path = {schema}, public, extensions")
        from pgvector.psycopg import register_vector
        register_vector(c)
        migrate(c)
        c.execute("insert into companies (company_id, name) values ('co-a', 'Northshore (Gmail)'), ('co-b', 'B')")
        c.execute("set role app_ingest")          # the import runs as the write role, under RLS
        try:
            yield c
        finally:
            c.execute("reset role")
            c.execute(f"drop schema {schema} cascade")


def run(conn, items, fakes, company_id="co-a"):
    return import_messages(conn, company_id, items, COMPANY, fakes.embed, fakes.extract, model="fake")


def test_import_stores_the_same_threads_the_file_reader_builds(conn):
    fakes = Fakes()
    stats = run(conn, as_gmail(THREADS), fakes)
    name, stored, leads = load_inbox(conn, "co-a")
    expected = {"G" + t.thread_id: t for t in read_mbox(MBOX, COMPANY) if t.thread_id in THREADS}
    assert name == "Northshore (Gmail)"
    assert {t.thread_id for t in stored} == set(expected)
    for t in stored:
        want = expected[t.thread_id]
        assert [m.body_clean for m in t.messages] == [m.body_clean for m in want.messages]
        assert [m.sender_role for m in t.messages] == [m.sender_role for m in want.messages]
        assert t.clean_text() == want.clean_text()          # same text the model reads (local send times kept)
    assert stats.chunks == sum(len(t.messages) for t in stored) == len(fakes.embedded)
    assert len(leads) == stats.leads == len(THREADS)


def test_a_rerun_imports_nothing_and_calls_no_model(conn):
    run(conn, as_gmail(THREADS), Fakes())
    again = Fakes()
    stats = run(conn, as_gmail(THREADS), again)
    assert stats.changed_threads == 0 and again.embedded == [] and again.extracted == []
    assert conn.execute("select count(*) from emails").fetchone()[0] == 0     # outside a company transaction: nothing
    _, stored, _ = load_inbox(conn, "co-a")
    assert sum(len(t.messages) for t in stored) == len(as_gmail(THREADS))    # no duplicates


def test_a_new_reply_reprocesses_only_its_thread_in_full(conn):
    items = as_gmail(THREADS)
    last = max(i for i, (_, thread, _) in enumerate(items) if thread == "Gt003")
    run(conn, items[:last] + items[last + 1:], Fakes())         # everything except t003's last email
    later = Fakes()
    stats = run(conn, [items[last]], later)                      # then that email arrives on its own
    assert later.extracted == ["Gt003"] and stats.changed_threads == 1
    n_t003 = sum(1 for _, thread, _ in items if thread == "Gt003")
    assert len(later.embedded) == n_t003                         # the whole thread, not just the new email


def test_newest_emails_first_still_gives_each_thread_its_first_subject(conn):
    """A partial import (--limit) brings the newest emails, so "Re: ..." arrives before the original subject."""
    items = as_gmail(THREADS)
    run(conn, items[-6:], Fakes())                               # the newest few first
    run(conn, items, Fakes())                                    # then everything
    _, stored, _ = load_inbox(conn, "co-a")
    expected = {"G" + t.thread_id: t.subject for t in read_mbox(MBOX, COMPANY) if t.thread_id in THREADS}
    assert {t.thread_id: t.subject for t in stored} == expected


def test_import_only_writes_the_given_company(conn):
    run(conn, as_gmail(THREADS), Fakes(), company_id="co-a")
    conn.execute("reset role")
    rows = conn.execute("select company_id, count(*) from emails group by company_id").fetchall()
    assert rows == [("co-a", len(as_gmail(THREADS)))]
    assert conn.execute("select count(*) from email_chunks where company_id <> 'co-a'").fetchone()[0] == 0


class OnePool:
    """The bit of a psycopg pool that load_company uses, over the test's connection."""
    def __init__(self, conn):
        self.conn = conn

    def connection(self):
        from contextlib import nullcontext
        return nullcontext(self.conn)


def test_the_agent_store_loads_an_imported_company_from_postgres(conn):
    from dj_inbox.agent.store import STATUS_TODAY, load_company

    run(conn, as_gmail(THREADS), Fakes())
    conn.execute("reset role")
    conn.execute("set role app_api")                     # the API reads as app_api
    vec = np.random.default_rng(1).normal(size=768).astype(np.float32)
    store = load_company(OnePool(conn), "co-a", STATUS_TODAY, lambda q: vec)
    assert store.company_name == "Northshore (Gmail)"
    assert set(store.threads) == {"G" + t for t in THREADS}
    assert {r["thread_id"] for r in store.leads} == set(store.threads)
    assert all(r["status"] in {"awaiting_reply", "waiting_on_client", "cold"} for r in store.leads)   # all "open"
    assert set(store.search("karaoke?", 3)) <= set(store.threads)


def test_a_company_with_no_imported_emails_has_no_store(conn):
    from dj_inbox.agent.store import STATUS_TODAY, load_company

    conn.execute("reset role")
    conn.execute("set role app_api")
    with pytest.raises(PermissionError):
        load_company(OnePool(conn), "co-b", STATUS_TODAY, lambda q: None)
