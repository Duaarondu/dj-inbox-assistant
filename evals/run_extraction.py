# evals/run_extraction.py
"""Measure lead extraction field by field against the synthetic inbox's known answers.

    python -m evals.run_extraction                           # Gemini 3.5 Flash-Lite, all 200 threads (~5 cents)
    python -m evals.run_extraction --model gemini-3.8-flash

Prints accuracy per field (date, venue, guests, price, ...), status accuracy, and the 5 counting questions
answered from the extracted leads with plain code: the "route counting to SQL" idea, measured.
Extracted leads are saved to data/processed/leads.jsonl (git-ignored) for the lead tracker later.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dj_inbox.extract.extract import PROMPT_VERSION, ThreadLeads, extract_all, source, status
from dj_inbox.gcp import gemini_client
from dj_inbox.ingest import CompanyConfig, EmailThread, read_mbox
from evals.run_baseline import PRICES

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "synthetic_inbox" / "data"
RESULTS = Path(__file__).resolve().parent / "results"
CACHE = ROOT / "data" / "extractions"
LEADS_OUT = ROOT / "data" / "processed" / "leads.jsonl"
COMPANY = CompanyConfig(domains=["northshoresound.example"])
TODAY = datetime(2026, 10, 28, 9, 0, tzinfo=ZoneInfo("America/New_York"))  # the synthetic inbox's "now"
FIELDS = ["event_type", "event_date", "venue", "venue_town", "guest_count", "hours", "budget",
          "package_quoted", "addons_quoted", "quoted_price", "status"]

# The 5 counting questions, answered with code over the leads table (how they'll be answered in the product).
COUNTS = {
    "count-01": lambda r: r["lead_index"] == 0 and r["status"] == "awaiting_reply",
    "count-02": lambda r: r["lead_index"] == 0 and r["status"] == "cold",
    "count-03": lambda r: r["lead_index"] == 0 and r["status"] == "booked",
    "count-04": lambda r: r["lead_index"] == 0 and r["source"] == "website_form",
    "count-05": lambda r: r["lead_index"] == 0 and r["status"] == "booked" and r["event_type"] == "wedding",
}


def to_rows(threads: list[EmailThread], extracted: dict[str, ThreadLeads]) -> list[dict]:
    """Flatten into one row per lead, the shape of the future `leads` table."""
    rows = []
    for t in threads:
        for i, lead in enumerate(extracted[t.thread_id].leads):
            rows.append({"thread_id": t.thread_id, "lead_index": i, **lead.model_dump(),
                         "status": status(lead, t, TODAY), "source": source(t, COMPANY)})
    return rows


def same(field: str, got, want) -> bool:
    if field == "addons_quoted":
        return sorted(a.lower() for a in got or []) == sorted(a.lower() for a in want or [])
    if isinstance(got, str) and isinstance(want, str):
        return got.strip().lower() == want.strip().lower()
    return got == want


def unstated_towns(threads: list[EmailThread], truth: list[dict]) -> set[tuple[str, int]]:
    """Leads whose true town never appears in their emails (website forms give only the venue name).
    Nobody could extract those, so they're left out of the venue_town score (found Nov 6: 48 'misses')."""
    text = {t.thread_id: t.clean_text() for t in threads}
    return {(r["thread_id"], r["lead_index"]) for r in truth
            if r["venue_town"] and r["venue_town"] not in text[r["thread_id"]]}


def compare(rows: list[dict], truth: list[dict], unstated: set[tuple[str, int]] = frozenset()) -> dict:
    """Per-field accuracy over leads matched by (thread, position); plus lead-count accuracy per thread.
    `unstated` leads are skipped for venue_town only."""
    got = {(r["thread_id"], r["lead_index"]): r for r in rows}
    want = {(r["thread_id"], r["lead_index"]): r for r in truth}
    hits, misses, scored = defaultdict(int), defaultdict(list), defaultdict(int)
    for key, w in want.items():
        g = got.get(key)
        for f in FIELDS:
            if f == "venue_town" and key in unstated:
                continue
            scored[f] += 1
            if g is not None and same(f, g.get(f), w.get(f)):
                hits[f] += 1
            else:
                misses[f].append({"lead": f"{key[0]}#{key[1]}", "got": None if g is None else g.get(f),
                                  "want": w.get(f)})
    per_thread = defaultdict(lambda: [0, 0])
    for t, _ in got:
        per_thread[t][0] += 1
    for t, _ in want:
        per_thread[t][1] += 1
    count_ok = sum(a == b for a, b in per_thread.values())
    return {"leads": len(want), "accuracy": {f: hits[f] / scored[f] for f in FIELDS if scored[f]},
            "scored": dict(scored), "misses": dict(misses),
            "lead_count_correct": f"{count_ok}/{len(per_thread)}"}


def counting(rows: list[dict], truth: list[dict]) -> list[tuple[str, int, int]]:
    return [(qid, sum(map(f, rows)), sum(map(f, truth))) for qid, f in COUNTS.items()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gemini-3.5-flash-lite", choices=[m for m in PRICES if m.startswith("gemini")])
    ap.add_argument("--limit", type=int, help="only the first N threads")
    args = ap.parse_args()

    client = gemini_client()

    threads = read_mbox(DATA / "inbox.mbox", COMPANY)[: args.limit]
    ids = {t.thread_id for t in threads}
    with (DATA / "ground_truth_leads.jsonl").open(encoding="utf-8") as fh:
        truth = [r for r in map(json.loads, fh) if r["thread_id"] in ids]

    start = time.perf_counter()
    extracted, usage, calls = extract_all(client, args.model, threads, cache_dir=CACHE / args.model)
    secs = time.perf_counter() - start
    p = PRICES[args.model]
    spent = (usage.input * p["input"] + usage.output * p["output"]) / 1e6
    rows = to_rows(threads, extracted)
    LEADS_OUT.parent.mkdir(parents=True, exist_ok=True)
    LEADS_OUT.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    result = compare(rows, truth, unstated_towns(threads, truth))
    print(f"{len(threads)} threads, {calls} API calls, {secs:.1f}s, ${spent:.4f} "
          f"({usage.input:,} in / {usage.output:,} out tokens) | "
          f"leads per thread right: {result['lead_count_correct']}")
    print("field accuracy over", result["leads"], "leads:")
    for f, a in result["accuracy"].items():
        print(f"  {f:<15} {a:6.1%}   {len(result['misses'].get(f, []))} wrong of {result['scored'][f]}")
    print("counting questions answered from the extracted leads (code, not the model):")
    for qid, got, want in counting(rows, truth):
        print(f"  {'OK' if got == want else 'XX'} {qid}  got {got}, truth {want}")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"extraction-{args.model}-{datetime.now():%Y%m%d-%H%M}.json"
    summary = {"model": args.model, "prompt_version": PROMPT_VERSION, "cost_usd": spent, "seconds": secs, **result}
    out.write_text(json.dumps(summary, indent=1,
                              default=str), encoding="utf-8")
    print(f"details (every wrong field) -> {out}")


if __name__ == "__main__":
    main()
