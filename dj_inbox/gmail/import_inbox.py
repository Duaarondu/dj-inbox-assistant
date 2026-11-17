# dj_inbox/gmail/import_inbox.py
"""Import a company's Gmail into Postgres: the same pipeline the file inbox went through.

    Gmail (filtered) -> parse + thread + clean/redact (dj_inbox.ingest) -> threads/emails rows
                     -> chunks + embeddings (email_chunks) -> lead extraction (leads rows)

Only threads that got a NEW email are chunked, embedded and extracted again, so a re-run costs nothing and
the same function serves automatic updates later (feed it just the new messages).

Local run (before the Connect Gmail page exists), as the write role app_ingest:
    python -m dj_inbox.db.api_role --role app_ingest              (once: writes DATABASE_INGEST_URL to .env)
    python -m dj_inbox.gmail.import_inbox --company c_... --client-secrets ~/Downloads/demo-loader.json \
        --account northshore.sound.demo@gmail.com --company-domain northshoresound.example --limit 20
    (then without --limit)
The OAuth token stays in memory; nothing about it is printed or written.
"""
from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from email.message import EmailMessage

import psycopg

from dj_inbox.db.inbox import load_inbox, save_chunks, save_leads, save_threads, stored_gmail_ids
from dj_inbox.extract.extract import PROMPT_VERSION, ThreadLeads, source
from dj_inbox.ingest import CompanyConfig, EmailThread, read_messages
from dj_inbox.rag.chunk import chunk_inbox

SCOPES = ["openid", "https://www.googleapis.com/auth/userinfo.email",
          "https://www.googleapis.com/auth/gmail.readonly"]
USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"

Extract = Callable[[list[EmailThread]], dict[str, ThreadLeads]]


@dataclass
class ImportStats:
    messages: int          # messages handed in
    changed_threads: int     # threads that got at least one new email
    threads_total: int
    chunks: int
    leads: int


def import_messages(conn: psycopg.Connection, company_id: str,
                    items: Iterable[tuple[EmailMessage, str | None, str | None]], company: CompanyConfig,
                    embed, extract: Extract, model: str) -> ImportStats:
    """embed(texts) -> vectors (RETRIEVAL_DOCUMENT); extract(threads) -> {thread_id: ThreadLeads}."""
    items = list(items)
    threads, gmail_ids = read_messages(items, company)
    changed = save_threads(conn, company_id, threads, gmail_ids)
    if not changed:
        _, stored, _ = load_inbox(conn, company_id)
        return ImportStats(len(items), 0, len(stored), 0, 0)

    # Work on the FULL stored thread (a new reply's thread may hold older emails not in this batch).
    _, stored, _ = load_inbox(conn, company_id)
    todo = [t for t in stored if t.thread_id in changed]
    chunks = chunk_inbox(todo)
    save_chunks(conn, company_id, chunks, embed([c.text for c in chunks]))
    extracted = extract(todo)
    n_leads = save_leads(conn, company_id, {t.thread_id: extracted[t.thread_id].leads for t in todo},
                         {t.thread_id: source(t, company) for t in todo}, model, PROMPT_VERSION)
    return ImportStats(len(items), len(todo), len(stored), len(chunks), n_leads)


def main(argv: list[str] | None = None) -> int:
    from pathlib import Path

    from google.auth.transport.requests import AuthorizedSession  # google-auth: already a dependency
    from google_auth_oauthlib.flow import InstalledAppFlow  # local dev only: pip install google-auth-oauthlib

    from dj_inbox.db import connect, database_url
    from dj_inbox.extract.extract import extract_all
    from dj_inbox.gcp import ROOT, gemini_client
    from dj_inbox.gmail.reader import IMPORT_QUERY, GmailReader
    from dj_inbox.rag.embed import Embedder

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--company", required=True, help="company_id to import into (from /me after onboarding)")
    ap.add_argument("--client-secrets", required=True, type=Path, help="OAuth client JSON (Desktop app)")
    ap.add_argument("--account", required=True, help="the Gmail account that must sign in")
    ap.add_argument("--company-domain", action="append", default=[], help="email domain(s) of the company")
    ap.add_argument("--limit", type=int, help="only the newest N messages")
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    args = ap.parse_args(argv)
    account = args.account.lower()

    flow = InstalledAppFlow.from_client_secrets_file(str(args.client_secrets.expanduser()), SCOPES)
    creds = flow.run_local_server(port=0, login_hint=account, prompt="select_account consent")
    session = AuthorizedSession(creds)
    who = session.get(USERINFO_URL, timeout=30).json().get("email", "").lower()
    if who != account:
        raise SystemExit(f"Signed in as {who or 'unknown'}, not {account}. Nothing was read.")

    start = time.perf_counter()
    reader = GmailReader(session, progress=print)
    ids = reader.list_ids(IMPORT_QUERY, limit=args.limit)
    client = gemini_client()
    embedder = Embedder(client, cache_dir=ROOT / "data" / "embeddings")         # cache: same text = no new call
    company = CompanyConfig(domains=args.company_domain, addresses=[account])
    with connect(database_url("DATABASE_INGEST_URL")) as conn:
        have = stored_gmail_ids(conn, args.company)
        todo = [i for i in ids if i not in have]
        print(f"{len(ids)} messages match `{IMPORT_QUERY}`; {len(ids) - len(todo)} already imported; "
              f"fetching {len(todo)}...")
        items = reader.fetch(todo)
        print(f"fetched {len(items)} in {time.perf_counter() - start:.0f}s ({reader.calls} Gmail calls)")
        stats = import_messages(
            conn, args.company, items, company,
            embed=lambda texts: embedder.embed(texts, "RETRIEVAL_DOCUMENT"),
            extract=lambda threads: extract_all(client, args.model, threads,
                                                cache_dir=ROOT / "data" / "extractions" / args.model)[0],
            model=args.model)
    print(f"{stats.threads_total} threads stored; {stats.changed_threads} with new emails -> "
          f"{stats.chunks} chunks ({embedder.calls} embedding calls), {stats.leads} leads "
          f"[{time.perf_counter() - start:.0f}s total]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
