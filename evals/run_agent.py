# evals/run_agent.py
"""Rung 5: the ADK agent. The model picks a tool (count_leads / find_events / find_leads / search_emails), then answers.

    python -m evals.run_agent --limit 5                          # Gemini 3.5 Flash-Lite, 5 questions
    python -m evals.run_agent                                    # all 50
    python -m evals.run_agent --model gemini-3.8-flash           # compare with RAG Flash: 44/50 (88%)
    python -m evals.run_agent --questions v2                     # eval v2 (evals/questions_v2.jsonl)

Same 50 questions, same scorer, same prices as run_baseline / run_rag. New measurements:
  - routing: count questions must call a counting tool (count_leads or find_events); no other v1 question may
  - tool calls and model calls per question (each model call costs time and tokens)
At the end it re-scores the latest plain-RAG run for the same model, so the comparison is fair.

Setup: .env + gcloud sign-in as before, and data/processed/leads.jsonl (python -m evals.run_extraction).
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from dj_inbox.agent.answer import build_workflow
from dj_inbox.agent.runtime import run_workflow_async
from evals.run_baseline import PRICES, TODAY, Usage, cost, summarize
from evals.score import Question, score_answer

QUESTION_SETS = {"v1": Path(__file__).resolve().parent / "questions.jsonl",
                 "v2": Path(__file__).resolve().parent / "questions_v2.jsonl"}
RESULTS = Path(__file__).resolve().parent / "results"
COMPANY_ID = "northshore"


COUNTING_TOOLS = {"count_leads", "find_events"}     # code over the lead table (find_events since Nov 11)


def routed_ok(question_type: str, tools: list[str]) -> bool:
    """Counting questions must be answered by a counting tool (never by reading emails); no other v1 question
    should touch one. ("How many events have we booked?" fits either: events = inquiries when no 2nd event is
    booked.)"""
    return bool(COUNTING_TOOLS & set(tools)) == (question_type == "count")


def run(workflow, model: str, questions: list[Question], max_cost: float, out: Path | None = None,
        company_id: str = COMPANY_ID) -> list[dict]:
    rows, spent = [], 0.0
    fh = out.open("a", encoding="utf-8") if out else None
    # ONE event loop for the whole run: ADK keeps one Gemini connection per loop. A new loop per question
    # (asyncio.run) left old connections to be closed on dead loops: "RuntimeError: Event loop is closed".
    loop = asyncio.Runner()
    try:
        for q in questions:
            if spent >= max_cost:
                print(f"Stopping: spent ${spent:.2f}, limit ${max_cost:.2f}")
                break
            try:
                result = loop.run(run_workflow_async(workflow, q.question, company_id))
                answer, error = str(result.output or ""), None
            except Exception as e:                       # one bad question shouldn't lose the whole run
                answer, error, result = "", f"{type(e).__name__}: {e}", None
            u = Usage(input=result.tokens_in, output=result.tokens_out) if result else Usage()
            spent += (c := cost(model, u))
            s = score_answer(answer, q)
            tools = [call.name for call in result.tool_calls] if result else []
            row = {"id": q.id, "type": q.type, "question": q.question, "expected": q.expected_answer,
                   "answer": answer, "correct": s.correct, "facts_found": s.facts_found,
                   "facts_expected": s.facts_expected, "deferred": s.deferred, "forbidden_hits": s.forbidden_hits,
                   "tools": tools, "tool_args": [call.args for call in result.tool_calls] if result else [],
                   "routed_ok": routed_ok(q.type, tools), "model_calls": result.model_calls if result else 0,
                   "model": model, "latency_s": round(result.latency_s, 2) if result else 0.0,
                   "cost_usd": round(c, 6), "input_tokens": u.input, "cache_write_tokens": 0,
                   "cache_read_tokens": 0, "output_tokens": u.output, "error": error}
            rows.append(row)
            if fh:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
            flag = "OK " if s.correct else ("XE " if error else ("XR " if not row["routed_ok"] else "XX "))
            print(f"{flag} {q.id:<20} {','.join(tools) or '-':<28} ${c:.5f} {row['latency_s']:5.1f}s  "
                  f"{(error or answer)[:60]!r}")
    finally:
        loop.close()
        if fh:
            fh.close()
    return rows


def routing_summary(rows: list[dict]) -> dict:
    counts = [r for r in rows if r["type"] == "count"]
    others = [r for r in rows if r["type"] != "count"]
    return {
        "count_to_counting_tool": f"{sum(r['routed_ok'] for r in counts)}/{len(counts)}",
        "others_kept_off_counting_tools": f"{sum(r['routed_ok'] for r in others)}/{len(others)}",
        "tool_use": dict(Counter(t for r in rows for t in r["tools"])),
        "avg_model_calls": round(sum(r["model_calls"] for r in rows) / len(rows), 2),
        "errors": sum(r["error"] is not None for r in rows),
    }


def rescored_rag(model: str, questions: list[Question]) -> dict | None:
    """The latest plain-RAG run for this model, re-scored with today's scorer."""
    runs = sorted(RESULTS.glob(f"rag-{model}-k5-*.jsonl"))
    if not runs:
        return None
    by_id = {q.id: q for q in questions}
    with runs[-1].open(encoding="utf-8") as fh:
        rows = [r for r in (json.loads(line) for line in fh) if r["id"] in by_id]
    for r in rows:
        r["correct"] = score_answer(r["answer"], by_id[r["id"]]).correct
    return {"file": runs[-1].name, **summarize(rows)} if rows else None


def main() -> None:
    from dj_inbox.agent.store import load_northshore

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gemini-3.5-flash-lite", choices=[m for m in sorted(PRICES) if "gemini" in m])
    ap.add_argument("--limit", type=int, help="only the first N questions")
    ap.add_argument("--max-cost", type=float, default=1.00)
    ap.add_argument("--questions", choices=sorted(QUESTION_SETS), default="v1")
    args = ap.parse_args()

    store = load_northshore()                             # registers the company; tools find it via the session
    with QUESTION_SETS[args.questions].open(encoding="utf-8") as fh:
        questions = [Question.model_validate_json(line) for line in fh][: args.limit]

    RESULTS.mkdir(exist_ok=True)
    tag = "" if args.questions == "v1" else f"-{args.questions}"
    out = RESULTS / f"agent{tag}-{args.model}-{datetime.now():%Y%m%d-%H%M}.jsonl"
    workflow = build_workflow(args.model, store.company_name, TODAY)
    rows = run(workflow, args.model, questions, args.max_cost, out=out, company_id=store.company_id)

    s, r = summarize(rows), routing_summary(rows)
    print(f"\nAGENT {s['correct']}/{s['questions']} correct ({s['accuracy']:.0%}) | ${s['total_cost_usd']:.4f} total, "
          f"${s['cost_per_question_usd']:.5f}/question | p50 {s['p50_latency_s']}s, p95 {s['p95_latency_s']}s | "
          f"{r['avg_model_calls']} model calls/question")
    print("by type:", s["by_type"])
    print(f"routing: count -> counting tool {r['count_to_counting_tool']}, "
          f"others kept off counting tools {r['others_kept_off_counting_tools']} | tool use {r['tool_use']} | "
          f"errors {r['errors']}")
    rag = rescored_rag(args.model, questions) if args.questions == "v1" else None
    if rag:
        print(f"\nRAG   {rag['correct']}/{rag['questions']} correct ({rag['accuracy']:.0%}) | "
              f"${rag['cost_per_question_usd']:.5f}/question | p50 {rag['p50_latency_s']}s "
              f"(re-scored: {rag['file']})")
        print("by type:", rag["by_type"])
    print(f"details -> {out}")


if __name__ == "__main__":
    main()
