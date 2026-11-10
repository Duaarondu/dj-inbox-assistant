# tests/test_run_agent.py
"""The agent eval runner, end to end with a scripted model and a tiny inbox: no credentials, no cost."""
from __future__ import annotations

import json
from datetime import datetime

import pytest

from dj_inbox.agent import store as store_mod
from dj_inbox.agent.answer import build_workflow
from dj_inbox.agent.store import InboxStore
from dj_inbox.agent.testing import FakeLlm, fake_text, fake_tool_call
from dj_inbox.ingest import Contacts, Email, EmailThread
from evals.run_agent import routed_ok, routing_summary, run
from evals.score import Question


def one_email_thread(tid: str, name: str, body: str) -> EmailThread:
    email = Email(message_id=f"<{tid}@x>", thread_id=tid, sent_at=datetime(2026, 5, 1, 10, 0), sender_role="client",
                  sender_name=name, sender_email=f"{tid}@mail.example", subject="Wedding DJ", body_raw=body,
                  body_clean=body, contacts=Contacts())
    return EmailThread(thread_id=tid, subject="Wedding DJ", messages=[email])


@pytest.fixture(autouse=True)
def tiny_store():
    store_mod.clear()
    threads = [one_email_thread("t1", "Marco Brennan", "Booked! Deposit sent for $2,395."),
               one_email_thread("t2", "Anna Lee", "Still thinking.")]
    leads = [{"thread_id": "t1", "lead_index": 0, "event_type": "wedding", "status": "booked", "source": "email"},
             {"thread_id": "t2", "lead_index": 0, "event_type": "wedding", "status": "cold", "source": "email"}]
    store_mod.register(InboxStore("northshore", "Northshore Sound Co.", leads, {t.thread_id: t for t in threads},
                                  lambda q, k: ["t1"]))
    yield
    store_mod.clear()


def question(qid: str, qtype: str, text: str, facts: list[dict]) -> Question:
    return Question(id=qid, type=qtype, question=text, expected_answer="", expected_facts=facts, gold_evidence=[],
                    should_defer=False, must_not_contain=[])


def test_run_scores_routes_and_prices_each_question(tmp_path):
    model = FakeLlm(script=[
        fake_tool_call("count_leads", status="booked"), fake_text("We have booked 1 event."),
        fake_tool_call("find_leads", client_name="Brennan"), fake_text("Marco Brennan paid $2,395. (sources: t1-0)"),
    ])
    questions = [question("count-01", "count", "How many events have we booked?", [{"kind": "number", "value": 1}]),
                 question("price-01", "price", "What did Brennan pay?", [{"kind": "money", "value": 2395}])]
    out = tmp_path / "agent.jsonl"
    rows = run(build_workflow(model), "gemini-3.5-flash-lite", questions, max_cost=1.0, out=out)

    assert [r["correct"] for r in rows] == [True, True]
    assert [r["tools"] for r in rows] == [["count_leads"], ["find_leads"]]
    assert rows[0]["tool_args"] == [{"status": "booked"}]
    assert [r["routed_ok"] for r in rows] == [True, True]
    assert rows[0]["model_calls"] == 2 and rows[0]["input_tokens"] == 200 and rows[0]["output_tokens"] == 40
    assert rows[0]["cost_usd"] == pytest.approx((200 * 0.30 + 40 * 2.50) / 1e6)
    assert [json.loads(line)["id"] for line in out.read_text().splitlines()] == ["count-01", "price-01"]

    summary = routing_summary(rows)
    assert summary["count_to_count_leads"] == "1/1" and summary["others_kept_off_count_leads"] == "1/1"


def test_an_error_on_one_question_is_recorded_and_the_run_continues():
    model = FakeLlm(script=[fake_text("We booked 1 event.")])          # 2nd question: model runs out of script
    questions = [question("count-01", "count", "How many booked?", [{"kind": "number", "value": 1}]),
                 question("count-02", "count", "How many cold?", [{"kind": "number", "value": 1}])]
    rows = run(build_workflow(model), "gemini-3.5-flash-lite", questions, max_cost=1.0)

    assert len(rows) == 2
    assert rows[0]["error"] is None and rows[0]["routed_ok"] is False      # answered a count without count_leads
    assert rows[1]["error"] and rows[1]["correct"] is False


def test_the_model_never_sees_curly_braces_or_a_company_id_parameter():
    model = FakeLlm(script=[fake_text("Not found in your emails.")])
    run(build_workflow(model), "gemini-3.5-flash-lite",
        [question("x", "not_in_inbox", "Karaoke?", [])], max_cost=1.0)
    request = model.requests[0]
    assert "{" not in str(request.config.system_instruction)
    assert sorted(model.tool_names_shown()) == ["count_leads", "find_leads", "search_emails"]
    for name in model.tool_names_shown():
        assert "company_id" not in model.tool_params_shown(name)


@pytest.mark.parametrize(("qtype", "tools", "ok"), [
    ("count", ["count_leads"], True), ("count", ["search_emails"], False),
    ("price", ["find_leads"], True), ("price", ["count_leads", "find_leads"], False),
])
def test_routed_ok(qtype, tools, ok):
    assert routed_ok(qtype, tools) is ok
