# tests/test_run_baseline.py
"""The baseline runner, tested with fake Claude and Gemini clients: no keys, no cost, runs in CI."""
import json
from types import SimpleNamespace

import pytest

from evals.run_baseline import PRICES, ClaudeBackend, GeminiBackend, Usage, ask_with_retry, cost, run, summarize
from evals.score import Question

ANSWERS = {"Brennan quote?": "We quoted $2,395 (thread t055).", "Garcia quote?": "Not found in your emails."}


class FakeClaude:
    def __init__(self):
        self.calls = []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        first = len(self.calls) == 1
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=ANSWERS[kw["messages"][0]["content"]])],
            usage=SimpleNamespace(input_tokens=10, cache_creation_input_tokens=50_000 if first else 0,
                                  cache_read_input_tokens=0 if first else 50_000, output_tokens=20))


class FakeGemini:
    """Fake google-genai client: .models.generate_content and .caches.create/delete."""

    def __init__(self, cache_ok=True):
        self.calls, self.created, self.deleted, self.cache_ok = [], [], [], cache_ok
        self.models = SimpleNamespace(generate_content=self._generate)
        self.caches = SimpleNamespace(create=self._create, delete=self._delete)

    def _create(self, model, config):
        if not self.cache_ok:
            raise RuntimeError("explicit caching not supported")
        self.created.append(config)
        return SimpleNamespace(name="cachedContents/abc")

    def _delete(self, name):
        self.deleted.append(name)

    def _generate(self, model, contents, config):
        self.calls.append({"contents": contents, "config": config})
        question = contents[-1].removeprefix("Question: ")
        cached = 50_000 if getattr(config, "cached_content", None) else 0
        return SimpleNamespace(
            text=ANSWERS[question],
            usage_metadata=SimpleNamespace(prompt_token_count=50_010, cached_content_token_count=cached,
                                           candidates_token_count=20, thoughts_token_count=100))


def question(qid, text, money=None, defer=False):
    return Question(id=qid, type="price" if money else "not_in_inbox", question=text, expected_answer="",
                    expected_facts=[{"kind": "money", "value": money}] if money else [], gold_evidence=[],
                    should_defer=defer, must_not_contain=[])


QS = [question("q1", "Brennan quote?", money=2395), question("q2", "Garcia quote?", defer=True)]


def test_cost_uses_cache_prices():
    p = PRICES["claude-sonnet-5-5"]
    assert cost("claude-sonnet-5-5", Usage(input=1_000_000)) == p["input"]
    assert cost("claude-sonnet-5-5", Usage(cache_read=1_000_000)) == p["cache_read"]


def test_claude_scores_answers_and_caches_the_inbox():
    client = FakeClaude()
    rows = run(ClaudeBackend(client, "claude-sonnet-5-5"), QS, inbox="INBOX TEXT", max_cost=5)
    assert [r["correct"] for r in rows] == [True, True]
    inbox_block = client.calls[0]["system"][1]
    assert "INBOX TEXT" in inbox_block["text"] and inbox_block["cache_control"] == {"type": "ephemeral"}
    assert rows[1]["cache_read_tokens"] == 50_000


def test_gemini_explicit_cache_is_created_used_and_deleted():
    client = FakeGemini()
    backend = GeminiBackend(client, "gemini-3.8-flash")
    backend.prepare("INBOX TEXT")
    rows = run(backend, QS, inbox="INBOX TEXT", max_cost=5)
    backend.close()
    assert "INBOX TEXT" in client.created[0].contents[0]
    assert all(c["contents"] == [c["contents"][0]] and c["contents"][0].startswith("Question:") for c in client.calls)
    assert all(c["config"].cached_content == "cachedContents/abc" for c in client.calls)
    assert client.deleted == ["cachedContents/abc"]
    assert [r["correct"] for r in rows] == [True, True]
    assert (rows[0]["input_tokens"], rows[0]["cache_read_tokens"], rows[0]["output_tokens"]) == (10, 50_000, 120)


def test_gemini_falls_back_when_caching_unavailable():
    client = FakeGemini(cache_ok=False)
    backend = GeminiBackend(client, "gemini-4-argon")
    backend.prepare("INBOX TEXT")
    rows = run(backend, QS, inbox="INBOX TEXT", max_cost=5)
    backend.close()
    assert backend.cache is None and client.deleted == []
    assert "INBOX TEXT" in client.calls[0]["contents"][0]       # inbox sent with every question instead
    assert [r["correct"] for r in rows] == [True, True]


class RateLimited(Exception):
    code = 429


def test_rate_limit_is_retried_then_succeeds():
    calls, waits = [], []

    class Flaky:
        model = "gemini-3.8-flash"

        def ask(self, inbox, question):
            calls.append(1)
            if len(calls) < 3:
                raise RateLimited()
            return "ok", Usage()

    assert ask_with_retry(Flaky(), "x", "q", sleep=waits.append) == ("ok", Usage())
    assert waits == [15, 30]


def test_other_errors_are_not_retried():
    class Broken:
        def ask(self, inbox, question):
            raise ValueError("bad request")

    with pytest.raises(ValueError):
        ask_with_retry(Broken(), "x", "q", sleep=lambda s: None)


def test_rows_are_saved_as_they_arrive(tmp_path):
    out = tmp_path / "r.jsonl"

    class DiesOnSecond:
        model = "gemini-3.8-flash"

        def __init__(self):
            self.n = 0

        def ask(self, inbox, question):
            self.n += 1
            if self.n == 2:
                raise ValueError("crash")
            return ANSWERS[question], Usage()

    with pytest.raises(ValueError):
        run(DiesOnSecond(), QS, inbox="x", max_cost=5, out=out)
    assert [json.loads(line)["id"] for line in out.read_text().splitlines()] == ["q1"]


def test_budget_guard_stops_early():
    rows = run(ClaudeBackend(FakeClaude(), "claude-sonnet-5-5"), QS, inbox="x", max_cost=0.0001)
    assert len(rows) == 1


def test_summary():
    rows = [{"type": "price", "correct": c, "latency_s": s, "cost_usd": 0.01, "cache_read_tokens": r}
            for c, s, r in [(True, 1.0, 0), (False, 2.0, 5), (True, 3.0, 5)]]
    s = summarize(rows)
    assert s["accuracy"] == round(2 / 3, 3) and s["by_type"] == {"price": "2/3"} and s["cache_hits"] == 2
