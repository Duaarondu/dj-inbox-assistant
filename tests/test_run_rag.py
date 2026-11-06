# tests/test_run_rag.py
"""Full RAG with fakes: word-count embeddings + a fake model. Checks the plumbing, not answer quality."""
import json

import numpy as np
from test_run_retrieval import FakeGenai

from dj_inbox.ingest import Contacts, Email, EmailThread
from dj_inbox.rag.chunk import chunk_inbox
from dj_inbox.rag.embed import Embedder
from dj_inbox.rag.retrieve import top_threads
from evals import run_rag
from evals.run_baseline import Usage
from evals.score import Question


def thread(tid, client, bodies):
    msgs = [Email(message_id=f"<{tid}.{i}>", thread_id=tid, sent_at=f"2025-11-2{i}T10:00:00+00:00",
                  sender_role=role, sender_name=client if role == "client" else "Dana Kim", sender_email="x@y.example",
                  subject=f"DJ for {client}", body_raw=b, body_clean=b, contacts=Contacts())
            for i, (role, b) in enumerate(bodies)]
    return EmailThread(thread_id=tid, subject=f"DJ for {client}", messages=msgs)


THREADS = [
    thread("t1", "Nicole Chen", [("client", "Nicole Chen birthday party pricing please"),
                                 ("company", "Hi Nicole, Signature package: $1,895")]),
    thread("t2", "Omar Santos", [("client", "Omar Santos wedding at Willow Estate"),
                                 ("company", "Hi Omar, Premier package: $2,990")]),
    thread("t3", "Ryan Russo", [("client", "Ryan Russo holiday party"), ("company", "Hi Ryan, Essential: $1,295")]),
]
Q = Question(id="price-01", type="price", question="What did we quote for Nicole Chen birthday party?",
             expected_answer="$1,895", expected_facts=[{"kind": "money", "value": 1895}],
             gold_evidence=[{"thread_id": "t1", "quote": "Signature package: $1,895"}], should_defer=False,
             must_not_contain=[])


class FakeModel:
    """Answers with the first price in whatever context it was given, and records that context."""
    model = "gemini-3.5-flash-lite"

    def __init__(self):
        self.contexts = []

    def ask(self, inbox, question):
        self.contexts.append(inbox)
        price = "$1,895" if "$1,895" in inbox else "Not found in your emails."
        return f"We quoted {price}.", Usage(input=len(inbox) // 4, output=10)


def test_top_threads_dedupes_and_keeps_order():
    chunks = chunk_inbox(THREADS)                       # t1-0, t1-1, t2-0, t2-1, t3-0, t3-1
    matrix = np.eye(6, dtype=np.float32)
    q = np.array([0.9, 1.0, 0.0, 0.0, 0.5, 0.0], dtype=np.float32)   # best: t1-1, t1-0, then t3-0
    assert top_threads(q, matrix, chunks, k=2) == ["t1", "t3"]


def test_model_only_sees_retrieved_threads(tmp_path):
    model = FakeModel()
    rows = run_rag.run(model, Embedder(FakeGenai(), cache_dir=tmp_path), THREADS, [Q], k=1, max_cost=1)
    assert rows[0]["retrieved"] == ["t1"] and rows[0]["recall"] == 1.0 and rows[0]["correct"]
    assert "Nicole" in model.contexts[0] and "Omar" not in model.contexts[0]   # the whole thread, nothing else


def test_rows_saved_and_baseline_rescored(tmp_path, monkeypatch):
    out = tmp_path / "rag.jsonl"
    run_rag.run(FakeModel(), Embedder(FakeGenai(), cache_dir=tmp_path), THREADS, [Q], k=2, max_cost=1, out=out)
    assert json.loads(out.read_text().splitlines()[0])["id"] == "price-01"

    old = {"id": "price-01", "type": "price", "answer": "We quoted $1,895.", "correct": False,
           "latency_s": 2.0, "cost_usd": 0.02, "cache_read_tokens": 0}       # scored wrong by an older scorer
    monkeypatch.setattr(run_rag, "RESULTS", tmp_path)
    (tmp_path / "baseline-gemini-3.5-flash-lite-20261105-1322.jsonl").write_text(json.dumps(old) + "\n")
    base = run_rag.rescored_baseline("gemini-3.5-flash-lite", [Q])
    assert base["correct"] == 1 and base["file"].startswith("baseline-")
    assert run_rag.rescored_baseline("gemini-3.8-flash", [Q]) is None


def test_budget_guard(tmp_path):
    rows = run_rag.run(FakeModel(), Embedder(FakeGenai(), cache_dir=tmp_path), THREADS, [Q, Q], k=1, max_cost=0)
    assert rows == []


def test_rescore_uses_todays_key(tmp_path):
    from evals.rescore import rescore_file
    f = tmp_path / "rag-x.jsonl"
    f.write_text(json.dumps({"id": "price-01", "type": "price", "answer": "We quoted $1,895.", "correct": False,
                             "latency_s": 1.0, "cost_usd": 0.001, "cache_read_tokens": 0}) + "\n")
    summary, changed = rescore_file(f, {"price-01": Q})
    assert summary["correct"] == 1 and changed == 1
