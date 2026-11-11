# dj_inbox/agent/store.py
"""Where the agent's tools get their data, one InboxStore per company.

Tools never take a company_id from the model. They call store_for(tool_context), which looks up the
company in the SESSION state (set by the server, not the model) and returns only that company's data.
Today the store is in memory (leads.jsonl + the mbox + cached vectors). Later it becomes Postgres, and
store_for becomes "every query has WHERE company_id = :company_id" plus row-level security.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dj_inbox.ingest import CompanyConfig, EmailThread

ROOT = Path(__file__).resolve().parents[2]
INBOX = ROOT / "synthetic_inbox" / "data" / "inbox.mbox"
LEADS = ROOT / "data" / "processed" / "leads.jsonl"
VECTORS = ROOT / "data" / "embeddings"
NORTHSHORE = CompanyConfig(domains=["northshoresound.example"])

SearchFn = Callable[[str, int], list[str]]      # (query, k) -> thread ids, best first


@dataclass
class InboxStore:
    company_id: str
    company_name: str
    leads: list[dict[str, Any]]                 # rows of leads.jsonl; status already computed as of "today"
    threads: dict[str, EmailThread]             # thread_id -> thread
    search: SearchFn                            # small-to-big retrieval


_STORES: dict[str, InboxStore] = {}


def register(store: InboxStore) -> None:
    _STORES[store.company_id] = store


def clear() -> None:
    """For tests."""
    _STORES.clear()


def get_store(company_id: str | None) -> InboxStore:
    """One company's data. The caller must get company_id from the session or the signed-in user, never
    from model output or a request body."""
    if company_id not in _STORES:
        raise PermissionError(f"No data loaded for company {company_id!r}")
    return _STORES[company_id]


def store_for(tool_context: Any) -> InboxStore:
    """The data of the company in this session, and nothing else."""
    return get_store(tool_context.state.get("company_id"))


def load_northshore() -> InboxStore:
    """The synthetic inbox with real retrieval. Embeds only what isn't cached yet (query embeddings: 1 call each)."""
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
    chunks = chunk_inbox(threads)
    matrix: np.ndarray = Embedder(client, cache_dir=VECTORS).embed([c.text for c in chunks], "RETRIEVAL_DOCUMENT")
    # Questions are cached in memory only: the disk cache is ~10 MB and is rewritten on every new text,
    # which would add a full-file write to every question the API answers.
    query_embedder = Embedder(client, cache_dir=None)

    def search(query: str, k: int) -> list[str]:
        qvec = query_embedder.embed([query], "RETRIEVAL_QUERY")[0]
        return top_threads(qvec, matrix, chunks, k)

    store = InboxStore("northshore", "Northshore Sound Co.", leads, {t.thread_id: t for t in threads}, search)
    register(store)
    return store
