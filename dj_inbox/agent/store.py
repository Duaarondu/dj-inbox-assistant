# dj_inbox/agent/store.py
"""Where the agent's tools get their data, one InboxStore per company.

Tools never take a company_id from the model. They call store_for(tool_context), which looks up the
company in the SESSION state (set by the server, not the model) and returns only that company's data.
Two kinds of store:
  - load_northshore(): the public demo, from files (leads.jsonl + the mbox + cached vectors), loaded at startup.
  - load_company(): a signed-in company's imported inbox, from Postgres (db/migrations/005), read through
    dj_inbox.db.inbox (WHERE company_id = ... plus row-level security). get_store() loads these on demand and
    reloads them after STORE_TTL_S, so newly imported emails show up without a restart.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dj_inbox.ingest import CompanyConfig, EmailThread

ROOT = Path(__file__).resolve().parents[2]
INBOX = ROOT / "synthetic_inbox" / "data" / "inbox.mbox"
LEADS = ROOT / "data" / "processed" / "leads.jsonl"
VECTORS = ROOT / "data" / "embeddings"
NORTHSHORE = CompanyConfig(domains=["northshoresound.example"])

SearchFn = Callable[[str, int], list[str]]      # (query, k) -> thread ids, best first

# "Today" for lead statuses (cold / awaiting reply). The demo data (file AND Gmail-imported) is synthetic and
# its "now" is Oct 28, 2026 9:00 New York time (= evals TODAY), so every company uses it for now.
# TODO before any real inbox: use the real date for real companies.
STATUS_TODAY = datetime(2026, 10, 28, 9, 0, tzinfo=timezone(timedelta(hours=-4)))


@dataclass
class InboxStore:
    company_id: str
    company_name: str
    leads: list[dict[str, Any]]                 # rows of leads.jsonl; status already computed as of "today"
    threads: dict[str, EmailThread]             # thread_id -> thread
    search: SearchFn                            # small-to-big retrieval


_STORES: dict[str, InboxStore] = {}
_LOADED_AT: dict[str, float] = {}                  # only for stores that came from the loader (they expire)
_LOADER: Callable[[str], InboxStore] | None = None
STORE_TTL_S = 15.0


def register(store: InboxStore) -> None:
    _STORES[store.company_id] = store


def set_loader(loader: Callable[[str], InboxStore] | None, ttl_s: float = STORE_TTL_S) -> None:
    """How get_store() finds a company that isn't registered (production: load_company from Postgres)."""
    global _LOADER, STORE_TTL_S
    _LOADER, STORE_TTL_S = loader, ttl_s


def clear() -> None:
    """For tests."""
    _STORES.clear()
    _LOADED_AT.clear()
    set_loader(None)


def get_store(company_id: str | None, now: Callable[[], float] = time.monotonic) -> InboxStore:
    """One company's data. The caller must get company_id from the session or the signed-in user, never
    from model output or a request body."""
    if company_id in _STORES and (company_id not in _LOADED_AT or now() - _LOADED_AT[company_id] < STORE_TTL_S):
        return _STORES[company_id]
    if _LOADER is None or not company_id:
        raise PermissionError(f"No data loaded for company {company_id!r}")
    store = _LOADER(company_id)                    # raises PermissionError when the company has no inbox yet
    _STORES[company_id], _LOADED_AT[company_id] = store, now()
    return store


def store_for(tool_context: Any) -> InboxStore:
    """The data of the company in this session, and nothing else."""
    return get_store(tool_context.state.get("company_id"))


_QUERY_EMBEDDER: Callable[[str], object] | None = None


def query_embedder() -> Callable[[str], object]:
    """query -> vector (RETRIEVAL_QUERY), one shared client. Questions are cached in memory only: the disk cache
    is ~10 MB and is rewritten on every new text, which would add a full-file write to every question."""
    global _QUERY_EMBEDDER
    if _QUERY_EMBEDDER is None:
        from dj_inbox.gcp import gemini_client
        from dj_inbox.rag.embed import Embedder
        embedder = Embedder(gemini_client(), cache_dir=None)
        _QUERY_EMBEDDER = lambda query: embedder.embed([query], "RETRIEVAL_QUERY")[0]   # noqa: E731
    return _QUERY_EMBEDDER


def load_company(pool, company_id: str, today: datetime, embed_query: Callable[[str], object]) -> InboxStore:
    """A company's imported inbox from Postgres. `pool` gives connections (app_api in production).
    Lead status is computed here, as of `today` (never stored: it changes with the date)."""
    from dj_inbox.db.inbox import load_inbox
    from dj_inbox.extract.extract import ExtractedLead, status
    from dj_inbox.rag.pg_search import top_threads_pg

    with pool.connection() as conn:
        name, threads, rows = load_inbox(conn, company_id)
    if not threads:
        raise PermissionError(f"Company {company_id!r} has no imported emails yet")
    by_id = {t.thread_id: t for t in threads}
    fields = set(ExtractedLead.model_fields)
    leads = []
    for r in rows:
        if r["thread_id"] in by_id:
            lead = ExtractedLead.model_validate({k: r[k] for k in fields})
            leads.append({**r, "status": status(lead, by_id[r["thread_id"]], today)})

    def search(query: str, k: int) -> list[str]:
        with pool.connection() as conn:
            return top_threads_pg(conn, company_id, embed_query(query), k)     # filtered on company_id + RLS

    return InboxStore(company_id, name, leads, by_id, search)


def load_northshore() -> InboxStore:
    """The synthetic inbox with real retrieval. Embeds only what isn't cached yet (query embeddings: 1 call each).

    VECTOR_BACKEND=pgvector searches Postgres (DATABASE_URL; load it first: python -m dj_inbox.db.load).
    Default: the NumPy search over the cached vectors. Same small-to-big contract either way.
    """
    import numpy as np

    from dj_inbox.gcp import gemini_client
    from dj_inbox.ingest import read_mbox
    from dj_inbox.rag.chunk import chunk_inbox
    from dj_inbox.rag.embed import Embedder
    from dj_inbox.rag.retrieve import top_threads

    if not LEADS.exists():
        raise SystemExit(f"{LEADS} is missing. Create it with: python -m evals.run_extraction")
    threads = read_mbox(INBOX, NORTHSHORE)
    with LEADS.open(encoding="utf-8") as fh:
        leads = [json.loads(line) for line in fh]

    client = gemini_client()
    embed_query = query_embedder()
    company_id = "northshore"

    if os.environ.get("VECTOR_BACKEND", "numpy") == "pgvector":
        from dj_inbox.db import make_pool
        from dj_inbox.rag.pg_search import top_threads_pg

        pool = make_pool()

        def search(query: str, k: int) -> list[str]:
            qvec = embed_query(query)
            with pool.connection() as conn:
                return top_threads_pg(conn, company_id, qvec, k)    # filtered on company_id in SQL
    else:
        chunks = chunk_inbox(threads)
        matrix: np.ndarray = Embedder(client, cache_dir=VECTORS).embed([c.text for c in chunks], "RETRIEVAL_DOCUMENT")

        def search(query: str, k: int) -> list[str]:
            qvec = embed_query(query)
            return top_threads(qvec, matrix, chunks, k)

    store = InboxStore(company_id, "Northshore Sound Co.", leads, {t.thread_id: t for t in threads}, search)
    register(store)
    return store
