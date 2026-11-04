# tests/test_eval_questions.py
"""The eval set is the measuring stick for everything else, so check the stick.

If a gold quote isn't really in the cleaned emails, recall@5 can never reach 100% and nobody would know why.
"""
import json
from collections import Counter
from pathlib import Path

import pytest

from dj_inbox.ingest import CompanyConfig, read_mbox
from evals.make_questions import OUT, build

DATA = Path(__file__).resolve().parents[1] / "synthetic_inbox" / "data"


@pytest.fixture(scope="module")
def questions():
    with OUT.open(encoding="utf-8") as fh:
        return [json.loads(row) for row in fh]


@pytest.fixture(scope="module")
def threads():
    return {t.thread_id: t for t in read_mbox(DATA / "inbox.mbox", CompanyConfig(domains=["northshoresound.example"]))}


@pytest.fixture(scope="module")
def leads():
    with (DATA / "ground_truth_leads.jsonl").open(encoding="utf-8") as fh:
        return [json.loads(row) for row in fh]


def test_file_is_up_to_date(questions):
    """questions.jsonl must match what the generator produces now. Rerun: python -m evals.make_questions"""
    assert questions == json.loads(json.dumps(build()))


def test_ids_unique_and_every_type_present(questions):
    assert len({q["id"] for q in questions}) == len(questions)
    assert {q["type"] for q in questions} == {"price", "stale_price", "venue", "event_date", "date_change", "guests",
                                              "status", "prompt_injection", "count", "not_in_inbox"}


def test_gold_quotes_are_in_the_cleaned_emails(questions, threads):
    for q in questions:
        for ev in q["gold_evidence"]:
            texts = [m.body_clean for m in threads[ev["thread_id"]].messages]
            assert any(ev["quote"] in text for text in texts), q["id"]


def test_questions_name_exactly_one_client(questions, leads):
    names = Counter(lead["client_name"] for lead in leads if lead["lead_index"] == 0)
    owner = {lead["thread_id"]: lead["client_name"] for lead in leads if lead["lead_index"] == 0}
    for q in questions:
        for ev in q["gold_evidence"]:
            name = owner[ev["thread_id"]]
            assert name in q["question"] and names[name] == 1, q["id"]


def test_expected_facts_match_ground_truth(questions, leads):
    by_thread = {lead["thread_id"]: lead for lead in leads if lead["lead_index"] == 0}
    field = {"money": "quoted_price", "date": "event_date", "text": None}
    for q in questions:
        if not q["gold_evidence"] or q["type"] == "venue":
            continue
        lead = by_thread[q["gold_evidence"][0]["thread_id"]]
        for fact in q["expected_facts"]:
            if fact["kind"] in ("money", "date"):
                assert fact["value"] == lead[field[fact["kind"]]], q["id"]
            elif fact["kind"] == "number":
                assert fact["value"] == lead["guest_count"], q["id"]
            else:
                assert fact["value"] == lead["status"], q["id"]


def test_not_in_inbox_really_is_not_in_the_inbox(questions, threads):
    everything = "\n".join(m.body_raw for t in threads.values() for m in t.messages).lower()
    for word in ["garcia", "karaoke", "henderson", "okafor"]:
        assert word not in everything, word
    assert sum(q["should_defer"] for q in questions) == 4
