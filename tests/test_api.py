# tests/test_api.py
"""The HTTP API with a tiny inbox and a scripted model: no credentials, no cost."""
from __future__ import annotations

import json
import logging
from datetime import date, datetime

import pytest
from fastapi.testclient import TestClient

from dj_inbox.agent import store as store_mod
from dj_inbox.agent.answer import build_workflow
from dj_inbox.agent.store import InboxStore
from dj_inbox.agent.testing import FakeLlm, fake_text, fake_tool_call
from dj_inbox.api.app import DailyCap, client_label, create_app
from dj_inbox.ingest import Contacts, Email, EmailThread


def make_thread(tid: str, subject: str, *messages: tuple[str, str, str]) -> EmailThread:
    emails = [Email(message_id=f"<{tid}-{i}@x>", thread_id=tid, sent_at=datetime(2026, 5, 1 + i, 10, 0),
                    sender_role=role, sender_name=name, sender_email=f"{tid}{i}@mail.example", subject=subject,
                    body_raw=body, body_clean=body, contacts=Contacts())
              for i, (role, name, body) in enumerate(messages)]
    return EmailThread(thread_id=tid, subject=subject, messages=emails)


T1 = make_thread("t1", "Wedding DJ inquiry", ("client", "Marco Brennan", "Wedding June 20?"),
                 ("company", "Dana", "Hi Marco, Signature package: $2,395."))
T2 = make_thread("t2", "New website inquiry", ("client", "Website Form", "Name: Anna Lee\nEvent type: Wedding"))
T9 = make_thread("t9", "Other company", ("client", "Someone Else", "Secret other-company email"))
LEADS = [{"thread_id": "t1", "lead_index": 0, "event_type": "wedding", "status": "booked", "source": "email",
          "event_date": "2026-06-20"},
         {"thread_id": "t1", "lead_index": 1, "event_type": "engagement_party", "status": "booked",
          "source": "email", "event_date": "2026-03-14"},
         {"thread_id": "t2", "lead_index": 0, "event_type": "wedding", "status": "cold", "source": "website_form",
          "event_date": "2026-09-27"}]


@pytest.fixture
def northshore():
    store_mod.clear()
    s = InboxStore("northshore", "Northshore Sound Co.", LEADS, {"t1": T1, "t2": T2}, lambda q, k: ["t1"])
    store_mod.register(s)
    store_mod.register(InboxStore("other-co", "Other", [], {"t9": T9}, lambda q, k: ["t9"]))
    yield s
    store_mod.clear()


def client_for(store: InboxStore, model: FakeLlm, cap: DailyCap | None = None) -> TestClient:
    app = create_app(store=store, workflow=build_workflow(model), model="gemini-3.5-flash-lite", cap=cap)
    return TestClient(app)


def test_health(northshore):
    with client_for(northshore, FakeLlm()) as c:
        assert c.get("/healthz").status_code == 404     # reserved on Cloud Run, so we never use it
        assert c.get("/health").json() == {"ok": True, "company": "Northshore Sound Co.", "threads": 2, "leads": 3,
                                            "model": "gemini-3.5-flash-lite"}


def test_ask_returns_answer_sources_and_cost(northshore, caplog):
    model = FakeLlm(script=[fake_tool_call("find_leads", client_name="Brennan"),
                            fake_text("We quoted Marco Brennan $2,395. (sources: t1-1, t7-0)")])
    with client_for(northshore, model) as c, caplog.at_level(logging.INFO, logger="dj_inbox.api"):
        r = c.post("/ask", json={"question": "What did we quote the Brennans?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "We quoted Marco Brennan $2,395."                # citation tail removed
    assert [s["email_id"] for s in body["sources"]] == ["t1-1"]               # t7-0 was never returned by a tool
    assert body["sources"][0]["sender"] == "Dana" and body["sources"][0]["subject"] == "Wedding DJ inquiry"
    assert body["tools"] == ["find_leads"]
    assert body["tokens_in"] == 200 and body["cost_usd"] == pytest.approx((200 * 0.30 + 40 * 2.50) / 1e6, abs=1e-6)

    line = json.loads(next(r.message for r in caplog.records if '"event": "ask"' in r.message))
    assert line["email_ids"] == ["t1-1"] and line["tools"] == ["find_leads"] and "question" not in line


def test_count_answer_lists_matched_threads(northshore):
    model = FakeLlm(script=[fake_tool_call("count_leads", status="booked"), fake_text("1 event is booked.")])
    with client_for(northshore, model) as c:
        body = c.post("/ask", json={"question": "How many booked?"}).json()
    assert body["matched_threads"] == ["t1"] and body["sources"] == []


@pytest.mark.parametrize("question", ["", "   ", "x" * 501])
def test_bad_questions_are_rejected_before_any_model_call(northshore, question):
    model = FakeLlm()
    with client_for(northshore, model) as c:
        assert c.post("/ask", json={"question": question}).status_code == 422
    assert model.requests == []


def test_daily_cap_is_checked_before_the_model(northshore):
    model = FakeLlm(script=[fake_text("Not found in your emails.")])
    with client_for(northshore, model, cap=DailyCap(1)) as c:
        assert c.post("/ask", json={"question": "karaoke?"}).status_code == 200
        r = c.post("/ask", json={"question": "karaoke again?"})
    assert r.status_code == 429
    assert len(model.requests) == 1                                            # the 2nd question never reached it


def test_daily_cap_resets_on_a_new_day():
    days = iter([date(2026, 11, 10), date(2026, 11, 10), date(2026, 11, 10), date(2026, 11, 11)])
    cap = DailyCap(1, today=lambda: next(days))
    assert cap.try_take() is True
    assert cap.try_take() is False
    assert cap.try_take() is True


def test_company_id_in_the_request_body_is_ignored(northshore):
    model = FakeLlm(script=[fake_tool_call("search_emails", query="secret"), fake_text("Not found in your emails.")])
    with client_for(northshore, model) as c:
        r = c.post("/ask", json={"question": "secret?", "company_id": "other-co"})
    assert r.status_code == 200
    tool_results = [json.dumps(part.function_response.response) for content in model.requests[1].contents
                    for part in content.parts or [] if part.function_response]
    assert len(tool_results) == 1                                              # what the tool gave the model
    assert '"thread_id": "t1"' in tool_results[0] and "Secret other-company email" not in tool_results[0]


def test_model_failure_is_a_502_not_a_crash(northshore):
    with client_for(northshore, FakeLlm(script=[])) as c:                      # FakeLlm raises on the first call
        r = c.post("/ask", json={"question": "anything"})
    assert r.status_code == 502


def test_leads_filters_and_labels_clients(northshore):
    with client_for(northshore, FakeLlm()) as c:
        all_leads = c.get("/leads").json()
        cold = c.get("/leads", params={"status": "cold"}).json()
    assert all_leads["count"] == 2 and [x["client"] for x in all_leads["leads"]] == ["Marco Brennan", "Anna Lee"]
    assert cold["count"] == 1 and cold["leads"][0]["source"] == "website_form"


def test_threads_are_company_scoped(northshore):
    with client_for(northshore, FakeLlm()) as c:
        ok = c.get("/threads/t1")
        other = c.get("/threads/t9")
    assert ok.status_code == 200 and [e["email_id"] for e in ok.json()["emails"]] == ["t1-0", "t1-1"]
    assert other.status_code == 404


def test_client_label_for_a_website_form():
    assert client_label(T2) == "Anna Lee"
    assert client_label(T1) == "Marco Brennan"


def test_leads_count_inquiries_like_the_agent_unless_all_events(northshore):
    """The tracker and count_leads must agree: an answer of "1 booked" means 1 row, not 2 with the 2nd event."""
    with client_for(northshore, FakeLlm()) as c:
        booked = c.get("/leads", params={"status": "booked"}).json()
        every_event = c.get("/leads", params={"status": "booked", "all_events": "true"}).json()
    assert booked["count"] == 1
    assert every_event["count"] == 2
    assert [x["event_type"] for x in every_event["leads"]] == ["engagement_party", "wedding"]   # by event date
