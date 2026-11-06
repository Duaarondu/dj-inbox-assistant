# evals/run_retrieval.py
"""Rung 3 of the ladder: measure RETRIEVAL ALONE. No model answers anything yet.

    python -m evals.run_retrieval            # recall@5 over every question that has a gold email
    python -m evals.run_retrieval --k 10     # how much does looking at 10 instead of 5 help?

For each question: embed it, find the most similar emails, and keep the first k DIFFERENT threads they belong to
("small-to-big": search small units, return the whole conversation). Then check whether the gold email is in them.
v1 returned single emails: 41% recall@5, but the right thread was found 95% of the time (Nov 5).
If retrieval misses, the answering model never even sees the right email, so this number caps the whole system.
Count and not-in-inbox questions have no gold email and are skipped here.

Uses the same Google Cloud setup as run_baseline (.env + gcloud sign-in).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dj_inbox.ingest import CompanyConfig, read_mbox
from dj_inbox.rag.chunk import chunk_inbox
from dj_inbox.rag.embed import Embedder
from dj_inbox.rag.search import cosine_top_k
from evals.score import Question, recall_at_k

ROOT = Path(__file__).resolve().parents[1]
INBOX = ROOT / "synthetic_inbox" / "data" / "inbox.mbox"
QUESTIONS = Path(__file__).resolve().parent / "questions.jsonl"
RESULTS = Path(__file__).resolve().parent / "results"
CACHE = ROOT / "data" / "embeddings"           # git-ignored (data/ is generated)


def evaluate(embedder: Embedder, chunks, threads: dict, questions: list[Question], k: int) -> list[dict]:
    """threads: {thread_id: EmailThread}, used to return whole conversations."""
    matrix = embedder.embed([c.text for c in chunks], "RETRIEVAL_DOCUMENT")
    scored = [q for q in questions if q.gold_evidence]
    qvecs = embedder.embed([q.question for q in scored], "RETRIEVAL_QUERY")
    rows = []
    for q, qvec in zip(scored, qvecs, strict=True):
        top = cosine_top_k(qvec, matrix, 50)
        thread_ids = []
        for i in top:
            tid = chunks[i].thread_id
            if tid not in thread_ids:
                thread_ids.append(tid)
            if len(thread_ids) == k:
                break
        retrieved = [(tid, threads[tid].clean_text()) for tid in thread_ids]
        rows.append({"id": q.id, "type": q.type, "question": q.question,
                     "recall": recall_at_k(retrieved, q.gold_evidence, k),
                     "gold_threads": sorted({g.thread_id for g in q.gold_evidence}),
                     "retrieved": thread_ids})
    return rows


def report(rows: list[dict], k: int) -> str:
    by_type: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r["recall"])
    overall = sum(r["recall"] for r in rows) / len(rows)
    lines = [f"recall@{k}: {overall:.0%} over {len(rows)} questions",
             "by type: " + ", ".join(f"{t} {sum(v) / len(v):.0%}" for t, v in sorted(by_type.items()))]
    misses = [r for r in rows if r["recall"] < 1]
    for r in misses:
        lines.append(f"  MISS {r['id']:<20} gold {r['gold_threads']}  got {r['retrieved']}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    from dotenv import load_dotenv
    from google import genai
    load_dotenv(ROOT / ".env")
    client = genai.Client(enterprise=True, project=os.environ["GOOGLE_CLOUD_PROJECT"],
                          location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))

    threads = read_mbox(INBOX, CompanyConfig(domains=["northshoresound.example"]))
    chunks = chunk_inbox(threads)
    with QUESTIONS.open(encoding="utf-8") as fh:
        questions = [Question.model_validate_json(line) for line in fh]

    start = time.perf_counter()
    embedder = Embedder(client, cache_dir=CACHE)
    threads_by_id = {t.thread_id: t for t in threads}
    rows = evaluate(embedder, chunks, threads_by_id, questions, args.k)
    print(f"{len(chunks)} chunks, {embedder.calls} embedding API calls, {time.perf_counter() - start:.1f}s")
    print(report(rows, args.k))

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"retrieval-k{args.k}-{datetime.now():%Y%m%d-%H%M}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"details -> {out}")


if __name__ == "__main__":
    main()
