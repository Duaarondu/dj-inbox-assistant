# evals/rescore.py
"""Re-score every saved answer run with TODAY's questions and scorer. No API calls, free.

    python -m evals.rescore

Answer keys and the scorer get fixed over time (e.g. Nov 6: "not booked" instead of "cold"). Re-scoring the
saved answers keeps old runs comparable to new ones without paying to run them again.
Runs with fewer than all questions (stopped early, or --limit) are marked "partial".
"""
from __future__ import annotations

import json
from pathlib import Path

from evals.run_baseline import summarize
from evals.score import Question, score_answer

RESULTS = Path(__file__).resolve().parent / "results"
QUESTIONS = Path(__file__).resolve().parent / "questions.jsonl"


def rescore_file(path: Path, questions: dict[str, Question]) -> tuple[dict, int]:
    """Summary of the file re-scored, and how many answers changed verdict."""
    with path.open(encoding="utf-8") as fh:
        rows = [r for r in map(json.loads, fh) if r["id"] in questions]
    changed = 0
    for r in rows:
        new = score_answer(r["answer"], questions[r["id"]]).correct
        changed += new != r["correct"]
        r["correct"] = new
    return summarize(rows), changed


def main() -> None:
    with QUESTIONS.open(encoding="utf-8") as fh:
        questions = {q.id: q for q in (Question.model_validate_json(line) for line in fh)}
    files = sorted(list(RESULTS.glob("baseline-*.jsonl")) + list(RESULTS.glob("rag-*.jsonl")))
    print(f"{'run':<52} {'correct':>9} {'acc':>5} {'$/question':>11} {'p50':>6}  changed")
    for f in files:
        s, changed = rescore_file(f, questions)
        if not s["questions"]:
            continue
        partial = "" if s["questions"] == len(questions) else "  (partial)"
        print(f"{f.stem:<52} {s['correct']:>4}/{s['questions']:<4} {s['accuracy']:>5.0%} "
              f"{s['cost_per_question_usd']:>11.5f} {s['p50_latency_s']:>5.1f}s  {changed:>3}{partial}")


if __name__ == "__main__":
    main()
