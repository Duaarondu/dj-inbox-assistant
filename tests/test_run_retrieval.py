# tests/test_run_retrieval.py
"""The retrieval pipeline with a fake embedder: word-count vectors instead of Gemini. Offline, free, runs in CI.

Word counts are a crude stand-in for meaning (it's basically keyword search), but the plumbing is identical:
chunks -> vectors -> top k -> recall@k.
"""
import json
import re
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dj_inbox.ingest import CompanyConfig, read_mbox
from dj_inbox.rag.chunk import chunk_inbox
from dj_inbox.rag.embed import DIM, Embedder
from evals.run_retrieval import evaluate, report
from evals.score import Question

ROOT = Path(__file__).resolve().parents[1]


def word_vector(text: str) -> list[float]:
    v = np.zeros(DIM, dtype=np.float32)
    for word in re.findall(r"[a-z0-9$,]+", text.lower()):
        v[zlib.crc32(word.encode()) % DIM] += 1
    return v.tolist()


class FakeGenai:
    """Like Gemini Embedding 2 on Google Cloud: one text per request."""

    def __init__(self, vectors_per_call=1, fail_first=0):
        self.requests, self.vectors_per_call, self.fail_first = [], vectors_per_call, fail_first
        self.models = SimpleNamespace(embed_content=self._embed)

    def _embed(self, model, contents, config):
        assert isinstance(contents, str), "send one text per request"
        self.requests.append((contents, config.task_type))
        if len(self.requests) <= self.fail_first:
            raise RateLimited()
        return SimpleNamespace(embeddings=[SimpleNamespace(values=word_vector(contents))] * self.vectors_per_call)


class RateLimited(Exception):
    code = 429


def test_embedder_one_text_per_request_and_caches(tmp_path):
    client = FakeGenai()
    texts = [f"email {i}" for i in range(250)]
    first = Embedder(client, cache_dir=tmp_path).embed(texts, "RETRIEVAL_DOCUMENT")
    assert first.shape == (250, DIM) and len(client.requests) == 250
    again = Embedder(client, cache_dir=tmp_path).embed(texts, "RETRIEVAL_DOCUMENT")   # loaded from disk
    assert len(client.requests) == 250 and np.allclose(first, again)
    assert np.allclose(first[7], word_vector("email 7"))                              # order is kept


def test_wrong_vector_count_is_an_error(tmp_path):
    """Regression (Nov 5): 100 texts in one request came back as 1 merged vector."""
    with pytest.raises(ValueError, match="expected 1 vector"):
        Embedder(FakeGenai(vectors_per_call=2), cache_dir=tmp_path).embed(["a"], "RETRIEVAL_DOCUMENT")


def test_rate_limits_are_retried(tmp_path):
    waits = []
    client = FakeGenai(fail_first=2)
    Embedder(client, cache_dir=tmp_path, sleep=waits.append).embed(["a"], "RETRIEVAL_DOCUMENT")
    assert waits == [5, 10] and len(client.requests) == 3


def test_query_and_document_vectors_are_cached_separately(tmp_path):
    client = FakeGenai()
    emb = Embedder(client, cache_dir=tmp_path)
    emb.embed(["same text"], "RETRIEVAL_DOCUMENT")
    emb.embed(["same text"], "RETRIEVAL_QUERY")
    assert [task for _, task in client.requests] == ["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]


def test_end_to_end_on_the_synthetic_inbox(tmp_path):
    inbox = ROOT / "synthetic_inbox" / "data" / "inbox.mbox"
    threads = read_mbox(inbox, CompanyConfig(domains=["northshoresound.example"]))
    chunks = chunk_inbox(threads)
    with (ROOT / "evals" / "questions.jsonl").open(encoding="utf-8") as fh:
        questions = [Question.model_validate_json(line) for line in fh]
    by_id = {t.thread_id: t for t in threads}
    rows = evaluate(Embedder(FakeGenai(), cache_dir=tmp_path), chunks, by_id, questions, k=5)
    assert len(rows) == sum(bool(q.gold_evidence) for q in questions)      # count/not-in-inbox skipped
    assert all(0 <= r["recall"] <= 1 for r in rows)
    assert all(len(r["retrieved"]) == 5 == len(set(r["retrieved"])) for r in rows)   # 5 DIFFERENT threads
    assert report(rows, 5).startswith("recall@5: ")
    json.dumps(rows)                                                         # results file is writable
