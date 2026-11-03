"""Checks that the synthetic inbox and its ground-truth labels agree.

If these fail, the eval labels are wrong, and every metric built on them is wrong too.
Run from synthetic_inbox/:  python -m pytest -q
"""
import json
import mailbox
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import generate_inbox as gen  # noqa: E402


@pytest.fixture(scope="session")
def data(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    subprocess.run([sys.executable, str(ROOT / "generate_inbox.py"), "--threads", "200", "--seed", "7",
                    "--out", str(out)], check=True, capture_output=True)
    threads = [json.loads(line) for line in (out / "threads.jsonl").read_text(encoding="utf-8").splitlines()]
    leads = [json.loads(line) for line in (out / "ground_truth_leads.jsonl").read_text(encoding="utf-8").splitlines()]
    return out, threads, leads


def thread_text(thread: dict) -> str:
    return "\n".join(m["body"] for m in thread["messages"])


def test_mbox_has_every_message(data):
    out, threads, _ = data
    box = mailbox.mbox(str(out / "inbox.mbox"))
    by_thread = defaultdict(int)
    for msg in box:
        by_thread[msg["X-GM-THRID"]] += 1
    assert dict(by_thread) == {t["thread_id"]: len(t["messages"]) for t in threads}


def test_every_thread_has_a_lead(data):
    _, threads, leads = data
    assert {t["thread_id"] for t in threads} == {line["thread_id"] for line in leads}


def test_quoted_price_appears_in_thread(data):
    """The gold price must literally be in the emails, in one of the two formats the company uses."""
    _, threads, leads = data
    text = {t["thread_id"]: thread_text(t) for t in threads}
    for lead in leads:
        if lead["quoted_price"] is None:
            continue
        p = lead["quoted_price"]
        assert f"${p:,}" in text[lead["thread_id"]] or f"${p}" in text[lead["thread_id"]], lead["thread_id"]


def test_quoted_price_matches_price_sheet(data):
    """Recompute every quote from the price sheet in force when it was sent."""
    _, _, leads = data
    venues = {v.name: v for v in gen.VENUES}
    for lead in leads:
        if lead["quoted_price"] is None:
            continue
        venue = venues.get(lead["venue"]) or gen.Venue("Private residence", lead["venue_town"], "nassau", False,
                                                       frozenset())
        q = gen.make_quote(lead["package_quoted"], lead["hours"], lead["addons_quoted"], venue,
                           datetime.fromisoformat(lead["event_date"]).date(),
                           datetime.fromisoformat(lead["quote_sent_at"]))
        assert q.total == lead["quoted_price"], lead["thread_id"]


def test_status_follows_the_rules(data):
    _, threads, leads = data
    last = {t["thread_id"]: t["messages"][-1] for t in threads}
    for lead in leads:
        m = last[lead["thread_id"]]
        age_days = (gen.TODAY - datetime.fromisoformat(m["date"])).days
        if lead["status"] == "cold":
            assert lead["last_message_from"] == "company" and age_days >= gen.COLD_AFTER_DAYS
        if lead["status"] == "waiting_on_client":
            assert lead["last_message_from"] == "company" and age_days < gen.COLD_AFTER_DAYS
        if lead["status"] == "awaiting_reply":
            assert lead["last_message_from"] == "client"


def test_messages_are_in_order_and_in_the_past(data):
    _, threads, _ = data
    for t in threads:
        times = [datetime.fromisoformat(m["date"]) for m in t["messages"]]
        assert times == sorted(times)
        assert times[-1] <= gen.TODAY
        assert times[0] >= gen.MIN_START


def test_events_come_after_the_inquiry(data):
    _, _, leads = data
    for lead in leads:
        if lead["event_date"]:
            assert lead["event_date"] > lead["first_message_at"][:10], lead["thread_id"]


def test_messy_cases_are_present(data):
    """The eval set needs enough of each hard case to measure anything."""
    _, _, leads = data
    tags = defaultdict(int)
    for lead in leads:
        if lead["lead_index"] == 0:
            for tag in lead["tags"]:
                tags[tag] += 1
    for tag, minimum in {"date_changed": 5, "upgraded": 5, "two_events": 2, "quoted_on_2025_sheet": 20,
                         "website_form": 30, "venue_referral": 5, "vague": 5, "quoted_history": 100}.items():
        assert tags[tag] >= minimum, (tag, tags[tag])


def test_same_seed_same_inbox(tmp_path, data):
    out, _, _ = data
    subprocess.run([sys.executable, str(ROOT / "generate_inbox.py"), "--seed", "7", "--out", str(tmp_path)],
                   check=True, capture_output=True)
    for name in ["ground_truth_leads.jsonl", "threads.jsonl", "venue_leads.csv"]:
        assert (tmp_path / name).read_text(encoding="utf-8") == (out / name).read_text(encoding="utf-8"), name


def test_website_form_subject_matches_form_date(data):
    """Regression: after a date change the subject showed the new date, not the one on the form."""
    _, threads, _ = data
    for t in threads:
        if t["subject"].startswith("New website inquiry"):
            form_date = t["messages"][0]["body"].split("Event date: ")[1].split("\n")[0]
            assert t["subject"].endswith(form_date), t["thread_id"]


def test_rehearsal_dinner_is_the_night_before(data):
    _, _, leads = data
    by_thread = defaultdict(list)
    for lead in leads:
        by_thread[lead["thread_id"]].append(lead)
    for pair in by_thread.values():
        if len(pair) == 2 and pair[1]["event_type"] == "rehearsal_dinner" and "date_changed" not in pair[0]["tags"]:
            d0, d1 = (datetime.fromisoformat(line["event_date"]) for line in pair)
            assert (d0 - d1).days == 1, pair[0]["thread_id"]
