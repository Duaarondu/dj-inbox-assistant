# evals/run_rag.py
"""Rung 4: full RAG. Retrieve the top 5 threads, then the model answers from ONLY those.

    python -m evals.run_rag --limit 5                              # Gemini 3.5 Flash-Lite, 5 questions
    python -m evals.run_rag                                        # all 50
    python -m evals.run_rag --model gemini-3.8-flash

Same 50 questions, same scorer and same instructions as the no-RAG baseline (evals/run_baseline.py).
The only change is what the model reads: ~5 relevant threads instead of the whole inbox.
At the end it re-scores the latest baseline run for the same model with today's scorer, so the comparison is fair.

Setup: the same .env + gcloud sign-in as run_baseline. Vectors come from data/embeddings (run run_retrieval first,
or this embeds the inbox once, ~2 cents).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from dj_inbox.gcp import gemini_client
from dj_inbox.ingest import CompanyConfig, EmailThread, read_mbox
from dj_inbox.rag.chunk import chunk_inbox
from dj_inbox.rag.embed import Embedder
from dj_inbox.rag.retrieve import top_threads
from evals.run_baseline import PRICES, ask_with_retry, cost, inbox_text, make_backend, summarize
from evals.score import Question, recall_at_k, score_answer

ROOT = Path(__file__).resolve().parents[1]
INBOX = ROOT / "synthetic_inbox" / "data" / "inbox.mbox"
QUESTIONS = Path(__file__).resolve().parent / "questions.jsonl"
RESULTS = Path(__file__).resolve().parent / "results"
CACHE = ROOT / "data" / "embeddings"


def run(backend, embedder: Embedder, threads: list[EmailThread], questions: list[Question], k: int,
        max_cost: float, out: Path | None = None, sleep=time.sleep) -> list[dict]:
    by_id = {t.thread_id: t for t in threads}
    chunks = chunk_inbox(threads)
    matrix = embedder.embed([c.text for c in chunks], "RETRIEVAL_DOCUMENT")
    qvecs = embedder.embed([q.question for q in questions], "RETRIEVAL_QUERY")

    rows, spent = [], 0.0
    fh = out.open("a", encoding="utf-8") if out else None
    try:
        for q, qvec in zip(questions, qvecs, strict=True):
            if spent >= max_cost:
                print(f"Stopping: spent ${spent:.2f}, limit ${max_cost:.2f}")
                break
            start = time.perf_counter()
            thread_ids = top_threads(qvec, matrix, chunks, k)                  # 1. retrieve
            context = inbox_text([by_id[t] for t in thread_ids])               # 2. only those threads
            answer, u = ask_with_retry(backend, context, q.question, sleep=sleep)  # 3. answer
            latency = time.perf_counter() - start
            spent += (c := cost(backend.model, u))
            s = score_answer(answer, q)
            recall = recall_at_k([(t, by_id[t].clean_text()) for t in thread_ids], q.gold_evidence, k)
            row = {"id": q.id, "type": q.type, "question": q.question, "expected": q.expected_answer,
                   "answer": answer, "correct": s.correct, "facts_found": s.facts_found,
                   "facts_expected": s.facts_expected, "deferred": s.deferred, "forbidden_hits": s.forbidden_hits,
                   "retrieved": thread_ids, "recall": recall, "model": backend.model, "latency_s": round(latency, 2),
                   "cost_usd": round(c, 6), "input_tokens": u.input, "cache_write_tokens": u.cache_write,
                   "cache_read_tokens": u.cache_read, "output_tokens": u.output}
            rows.append(row)
            if fh:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
            flag = "OK " if s.correct else ("XR " if recall < 1 else "XX ")   # XR = retrieval missed the email
            print(f"{flag} {q.id:<20} ${c:.5f} {latency:5.1f}s  {answer[:70]!r}")
    finally:
        if fh:
            fh.close()
    return rows


def rescored_baseline(model: str, questions: list[Question]) -> dict | None:
    """The latest no-RAG run for this model, re-scored with today's scorer (it has been fixed since)."""
    runs = sorted(RESULTS.glob(f"baseline-{model}-*.jsonl"))
    if not runs:
        return None
    by_id = {q.id: q for q in questions}
    with runs[-1].open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    rows = [r for r in rows if r["id"] in by_id]
    for r in rows:
        r["correct"] = score_answer(r["answer"], by_id[r["id"]]).correct
    return {"file": runs[-1].name, **summarize(rows)} if rows else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gemini-3.5-flash-lite", choices=sorted(PRICES))
    ap.add_argument("--k", type=int, default=5, help="threads given to the model")
    ap.add_argument("--limit", type=int, help="only the first N questions")
    ap.add_argument("--max-cost", type=float, default=1.00)
    args = ap.parse_args()

    backend = make_backend(args.model)        # loads .env; Gemini or Claude
    embed_client = gemini_client()

    threads = read_mbox(INBOX, CompanyConfig(domains=["northshoresound.example"]))
    with QUESTIONS.open(encoding="utf-8") as fh:
        questions = [Question.model_validate_json(line) for line in fh][: args.limit]

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"rag-{args.model}-k{args.k}-{datetime.now():%Y%m%d-%H%M}.jsonl"
    rows = run(backend, Embedder(embed_client, cache_dir=CACHE), threads, questions, args.k, args.max_cost, out=out)

    s = summarize(rows)
    avg_in = sum(r["input_tokens"] + r["cache_read_tokens"] for r in rows) / len(rows)
    print(f"\nRAG  {s['correct']}/{s['questions']} correct ({s['accuracy']:.0%}) | ${s['total_cost_usd']:.4f} total, "
          f"${s['cost_per_question_usd']:.5f}/question | p50 {s['p50_latency_s']}s, p95 {s['p95_latency_s']}s | "
          f"~{avg_in:,.0f} input tokens/question")
    print("by type:", s["by_type"])
    wrong = [r for r in rows if not r["correct"]]
    print("wrong:", dict(Counter(r["type"] for r in wrong)),
          f"| retrieval missed the email in {sum(r['recall'] < 1 for r in wrong)} of them")

    base = rescored_baseline(args.model, questions)
    if base:
        print(f"\nNo-RAG {base['correct']}/{base['questions']} correct ({base['accuracy']:.0%}) | "
              f"${base['cost_per_question_usd']:.5f}/question | p50 {base['p50_latency_s']}s "
              f"(re-scored with today's scorer: {base['file']})")
        print("by type:", base["by_type"])
    print(f"details -> {out}")


if __name__ == "__main__":
    main()
