# tests/test_eval_questions_v2.py
"""The v2 answer keys, checked against the synthetic data. Bad answer keys cost us twice in v1 (Bug Journal #6, #10):
a "model failure" that is really a wrong key wastes a whole experiment. Each check here is independent of the
generator's own logic where it can be."""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from dj_inbox.agent import store as store_mod
from dj_inbox.agent.store import InboxStore
from dj_inbox.agent.tools import find_leads
from dj_inbox.ingest import CompanyConfig, read_mbox
from evals.make_questions import long_date
from evals.score import Question

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "synthetic_inbox" / "data"
QUESTIONS = [Question.model_validate_json(line)
             for line in (ROOT / "evals" / "questions_v2.jsonl").read_text(encoding="utf-8").splitlines()]
THREADS = {t.thread_id: t for t in read_mbox(DATA / "inbox.mbox", CompanyConfig(domains=["northshoresound.example"]))}
LEADS = [json.loads(line) for line in (DATA / "ground_truth_leads.jsonl").read_text(encoding="utf-8").splitlines()]


def of_type(kind: str) -> list[Question]:
    return [q for q in QUESTIONS if q.type == kind]


def test_shape():
    assert len({q.id for q in QUESTIONS}) == len(QUESTIONS) == 38
    assert Counter(q.type for q in QUESTIONS) == {"sum": 4, "list": 4, "calendar": 4, "not_in_emails": 4,
                                                   "no_name": 5, "described": 4, "misspelled": 5, "compare": 4,
                                                   "repeat_client": 4}


def test_every_quote_is_in_its_thread():
    for q in QUESTIONS:
        for g in q.gold_evidence:
            assert " ".join(g.quote.split()) in " ".join(THREADS[g.thread_id].clean_text().split()), q.id


def test_money_facts_appear_in_the_quoted_email():
    for q in QUESTIONS:
        if q.gold_evidence and q.type not in ("sum", "list"):
            quotes = " ".join(g.quote for g in q.gold_evidence)
            for f in q.expected_facts:
                if f.kind == "money":
                    assert f"${f.value:,}" in quotes or f"${f.value}" in quotes, q.id


def booked_by_rule(lead: dict) -> bool:
    """Independent re-statement of the rule (Nov 11): a second event is booked only if a deposit covers it."""
    if lead["status"] != "booked":
        return False
    if lead["lead_index"] == 0:
        return True
    text = THREADS[lead["thread_id"]].clean_text()
    return f"30% of ${lead['quoted_price']:,})" in text          # its own deposit (none in this inbox)


def test_revenue_sums_recomputed_by_hand():
    by_year = Counter()
    for lead in LEADS:
        if booked_by_rule(lead) and lead["event_date"]:
            by_year[lead["event_date"][:4]] += lead["quoted_price"] or 0
    sums = {q.question: q.expected_facts[0].value for q in of_type("sum")}
    assert sums["How much revenue do we have from events booked for this year?"] == by_year["2026"] == 84580
    assert sums["How much revenue did we book for events in 2025?"] == by_year["2025"]


def test_second_events_whose_deposit_covered_only_the_main_event_are_not_booked():
    """t001: deposit "30% of $1,750" (the wedding), so the $1,195 rehearsal dinner is not booked."""
    t001_dinner = next(lead for lead in LEADS if lead["thread_id"] == "t001" and lead["lead_index"] == 1)
    assert t001_dinner["status"] == "booked"                       # what the synthetic ground truth says
    assert booked_by_rule(t001_dinner) is False                    # what the emails support


def test_a_misspelled_name_really_misses_exact_lookup_but_the_real_name_hits():
    """Otherwise the 'misspelled' type wouldn't test what it claims."""
    store_mod.clear()
    store_mod.register(InboxStore("t", "T", LEADS, THREADS, lambda q, k: []))
    ctx = SimpleNamespace(state={"company_id": "t"})
    try:
        for q in of_type("misspelled"):
            typo_name = re.search(r"quote (.+?) for their", q.question).group(1)
            real_name = re.search(r"the client is (.+)\)", q.expected_answer).group(1)
            assert find_leads(ctx, typo_name)["matches"] == [], q.id
            assert find_leads(ctx, real_name)["matches"], q.id
    finally:
        store_mod.clear()


def test_planning_details_are_absent_from_the_whole_inbox():
    alltext = " ".join(t.subject + " " + t.clean_text() for t in THREADS.values()).lower()
    for phrase in ["flower wall", "first dance", "grand entrance"]:
        assert phrase not in alltext
    ceremony = next(q for q in of_type("not_in_emails") if "ceremony start" in q.question)
    client = re.search(r"for (.+)'s wedding", ceremony.question).group(1)
    thread = next(THREADS[lead["thread_id"]] for lead in LEADS if lead["client_name"] == client)
    assert not re.search(r"\b\d{1,2}(:\d\d)?\s?(am|pm)\b", thread.clean_text().lower())
    assert all(q.should_defer and not q.expected_facts for q in of_type("not_in_emails"))


@pytest.mark.parametrize("q", of_type("no_name"), ids=lambda q: q.id)
def test_no_name_questions_have_exactly_one_answer(q):
    venue, when = re.search(r"at (.+) on (.+)\?", q.question).groups()
    hits = [lead for lead in LEADS if lead["venue"] == venue and lead["event_date"]
            and long_date(lead["event_date"]) == when]
    assert [h["client_name"] for h in hits] == [q.expected_facts[0].value]


@pytest.mark.parametrize("q", of_type("described"), ids=lambda q: q.id)
def test_described_questions_have_exactly_one_answer(q):
    label, guests, month, year = re.search(r"a (.+) for about (\d+) guests in (\w+) (\d{4})", q.question).groups()
    hits = [lead for lead in LEADS if lead["event_label"] == label and lead["guest_count"] == int(guests)
            and lead["event_date"] and date.fromisoformat(lead["event_date"]).strftime("%B %Y") == f"{month} {year}"]
    assert [h["client_name"] for h in hits] == [q.expected_facts[0].value]


def test_compare_and_repeat_client_keys():
    for q in of_type("compare"):
        a, b = (f.value for f in q.expected_facts)
        assert a != b, q.id
    for q in of_type("repeat_client"):
        assert q.must_not_contain and q.must_not_contain[0] not in q.expected_answer.split("(")[0], q.id
