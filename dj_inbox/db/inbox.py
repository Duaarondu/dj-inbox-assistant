# dj_inbox/db/inbox.py
"""Write and read one company's inbox in Postgres (db/migrations/005_inbox_tables.sql).

Every function runs in ONE transaction that starts with set_config('app.company_id', ..., true). Connected
as app_ingest (writes) or app_api (reads), row-level security then limits the transaction to that company's
rows, on top of the WHERE company_id = ... in every query (two locks, as in pg_search.py).

Writes are idempotent: re-importing the same emails adds nothing (dedupe on Message-ID and Gmail id).
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import numpy as np
import psycopg
from psycopg.types.json import Jsonb

from dj_inbox.db.load import load_chunks
from dj_inbox.extract.extract import ExtractedLead
from dj_inbox.ingest import Contacts, Email, EmailThread
from dj_inbox.rag.chunk import Chunk


@contextmanager
def company_tx(conn: psycopg.Connection, company_id: str) -> Iterator[psycopg.Connection]:
    """A transaction locked to one company (RLS). `true` = this transaction only: it can't leak to the next."""
    if not company_id:
        raise ValueError("company_id is required")
    with conn.transaction():
        conn.execute("select set_config('app.company_id', %s, true)", (company_id,))
        yield conn


def _offset_minutes(dt: datetime) -> int:
    off = dt.utcoffset()
    return int(off.total_seconds() // 60) if off is not None else 0


def save_threads(conn: psycopg.Connection, company_id: str, threads: list[EmailThread],
                 gmail_ids: dict[str, str] | None = None) -> set[str]:
    """Insert threads and emails that aren't stored yet. Returns the ids of threads that got a new email."""
    gmail_ids = gmail_ids or {}
    changed: set[str] = set()
    with company_tx(conn, company_id), conn.cursor() as cur:
        cur.executemany("""insert into threads (company_id, thread_id, subject) values (%s, %s, %s)
                           on conflict (company_id, thread_id) do nothing""",
                        [(company_id, t.thread_id, t.subject) for t in threads])
        for t in threads:
            for m in t.messages:
                cur.execute("""insert into emails (company_id, message_id, gmail_id, thread_id, sent_at, sent_offset,
                                                   sender_role, sender_name, sender_email, subject, body_raw,
                                                   body_clean, contacts)
                               values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                               on conflict do nothing""",
                            (company_id, m.message_id, gmail_ids.get(m.message_id), t.thread_id, m.sent_at,
                             _offset_minutes(m.sent_at),
                             m.sender_role, m.sender_name, m.sender_email, m.subject, m.body_raw, m.body_clean,
                             Jsonb(m.contacts.model_dump())))
                if cur.rowcount:
                    changed.add(t.thread_id)
    return changed


def save_chunks(conn: psycopg.Connection, company_id: str, chunks: list[Chunk], vectors: np.ndarray) -> int:
    with company_tx(conn, company_id):
        return load_chunks(conn, company_id, chunks, vectors)


def save_leads(conn: psycopg.Connection, company_id: str, leads: dict[str, list[ExtractedLead]],
               sources: dict[str, str], model: str, prompt_version: str) -> int:
    """Replace the leads of these threads (a thread's lead count can change when a new email arrives)."""
    rows = [(company_id, thread_id, i, Jsonb(lead.model_dump()), sources[thread_id], model, prompt_version)
            for thread_id, items in leads.items() for i, lead in enumerate(items)]
    with company_tx(conn, company_id), conn.cursor() as cur:
        cur.execute("delete from leads where company_id = %s and thread_id = any(%s)", (company_id, list(leads)))
        cur.executemany("""insert into leads (company_id, thread_id, lead_index, fields, source, model, prompt_version)
                           values (%s, %s, %s, %s, %s, %s, %s)""", rows)
    return len(rows)


def load_inbox(conn: psycopg.Connection, company_id: str) -> tuple[str, list[EmailThread], list[dict]]:
    """(company name, threads oldest first, lead rows) for one company.

    Lead rows are the stored fields plus thread_id, lead_index and source. Status is NOT here: it depends on
    today's date and is computed by the caller (dj_inbox.agent.store).
    """
    with company_tx(conn, company_id):
        name_row = conn.execute("select name from companies where company_id = %s", (company_id,)).fetchone()
        subjects = dict(conn.execute("select thread_id, subject from threads where company_id = %s", (company_id,)))
        emails = conn.execute("""select message_id, thread_id, sent_at, sent_offset, sender_role, sender_name,
                                        sender_email, subject, body_raw, body_clean, contacts
                                   from emails where company_id = %s order by sent_at, message_id""",
                              (company_id,)).fetchall()
        lead_rows = conn.execute("""select thread_id, lead_index, fields, source from leads where company_id = %s
                                    order by thread_id, lead_index""", (company_id,)).fetchall()
    if name_row is None:
        raise PermissionError(f"No company {company_id!r}")
    by_thread: dict[str, list[Email]] = {}
    for mid, tid, sent_at, offset, role, sname, semail, subject, raw, clean, contacts in emails:
        by_thread.setdefault(tid, []).append(Email(
            message_id=mid, thread_id=tid, sent_at=sent_at.astimezone(timezone(timedelta(minutes=offset))),
            sender_role=role, sender_name=sname, sender_email=semail,
            subject=subject, body_raw=raw, body_clean=clean, contacts=Contacts.model_validate(contacts)))
    threads = [EmailThread(thread_id=tid, subject=subjects.get(tid, msgs[0].subject), messages=msgs)
               for tid, msgs in by_thread.items()]
    threads.sort(key=lambda t: t.messages[0].sent_at)
    leads = [{"thread_id": tid, "lead_index": i, **ExtractedLead.model_validate(fields).model_dump(), "source": src}
             for tid, i, fields, src in lead_rows]
    return name_row[0], threads, leads
