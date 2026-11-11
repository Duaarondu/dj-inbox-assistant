# tests/test_agent_tools.py
"""Spec for the agent's three tools (dj_inbox/agent/tools.py). Tiny fake inbox, no model, no network.

Five Northshore threads:
    t1  Marco Brennan    wedding (booked) + engagement party (booked, 2nd event)   email
    t2  Anna Lee         wedding (cold)                                            website form
    t3  Omar Shapiro     wedding (awaiting_reply)                                  email
    t4  Emily Santos     sweet16 (lost), "planning" in the text                    email
    t5  Omar Santos      wedding (waiting_on_client)                               email
and one thread for a DIFFERENT company, also from a "Marco Brennan", that must never show up for Northshore.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from dj_inbox.agent import store as store_mod
from dj_inbox.agent import tools as tools_mod
from dj_inbox.agent.store import InboxStore
from dj_inbox.agent.tools import SEARCH_K, _word_close, count_leads, find_events, find_leads, search_emails
from dj_inbox.ingest import Contacts, Email, EmailThread


def thread(tid: str, subject: str, *messages: tuple[str, str, str]) -> EmailThread:
    """messages: (role, sender name, body)"""
    emails = [Email(message_id=f"<{tid}-{i}@x>", thread_id=tid, sent_at=datetime(2026, 5, 1 + i, 10, 0),
                    sender_role=role, sender_name=name, sender_email=f"{tid}{i}@mail.example", subject=subject,
                    body_raw=body, body_clean=body, contacts=Contacts())
              for i, (role, name, body) in enumerate(messages)]
    return EmailThread(thread_id=tid, subject=subject, messages=emails)


def lead(tid: str, index: int, event_type: str, status: str, source: str = "email", price: int | None = 2000,
         event_date: str | None = None, addons: tuple[str, ...] = (), venue: str | None = None) -> dict:
    return {"thread_id": tid, "lead_index": index, "event_type": event_type, "status": status, "source": source,
            "quoted_price": price, "outcome": "booked" if status == "booked" else "open",
            "event_date": event_date, "addons_quoted": list(addons), "venue": venue}


NORTHSHORE_THREADS = [
    thread("t1", "Wedding DJ inquiry",
           ("client", "Marco Brennan", "Hi! Wedding on June 20 and an engagement party in March."),
           ("company", "Dana", "Hi Marco, Signature package: $2,395. Engagement party: $1,200.")),
    thread("t2", "New website inquiry: wedding on 09/27/2026",
           ("client", "Website Form", "Name: Anna Lee\nEvent type: Wedding\nGuest count: 120"),
           ("company", "Dana", "Hi Anna, our Essential package is $1,795.")),
    thread("t3", "dj?", ("client", "Omar Shapiro", "Do you have June 6 open?")),
    thread("t4", "Sweet 16", ("client", "Emily Santos", "We are planning a sweet 16 for our daughter."),
           ("company", "Dana", "Hi Emily, that date works: $1,500.")),
    thread("t5", "dj?", ("client", "Omar Santos", "Price for a wedding, 150 guests?"),
           ("company", "Dana", "Hi Omar, Premier is $3,100.")),
]
NORTHSHORE_LEADS = [
    lead("t1", 0, "wedding", "booked", price=2395, event_date="2026-06-20", addons=("photo booth", "uplighting"),
         venue="The Willow Estate"),
    lead("t1", 1, "engagement_party", "booked", price=1200, event_date="2026-03-14"),
    lead("t2", 0, "wedding", "cold", source="website_form", price=1795, event_date="2026-09-27",
         addons=("Photo Booth",)),
    lead("t3", 0, "wedding", "awaiting_reply", event_date="2026-06-06"),
    lead("t4", 0, "sweet16", "lost", price=1500, event_date="2025-07-19", addons=("dancing on clouds",)),
    lead("t5", 0, "wedding", "waiting_on_client", price=None),                 # no date, no price yet
]
OTHER_THREADS = [thread("t9", "DJ for our wedding", ("client", "Marco Brennan", "Is May 2 free?"))]
OTHER_LEADS = [lead("t9", 0, "wedding", "booked", price=9999, event_date="2026-06-20")]

search_calls: list[tuple[str, int]] = []


def fake_search(query: str, k: int) -> list[str]:
    search_calls.append((query, k))
    return ["t3", "t1", "t4"][:k]


@pytest.fixture(autouse=True)
def stores():
    store_mod.clear()
    search_calls.clear()
    store_mod.register(InboxStore("northshore", "Northshore Sound Co.", NORTHSHORE_LEADS,
                                  {t.thread_id: t for t in NORTHSHORE_THREADS}, fake_search))
    store_mod.register(InboxStore("other-co", "Other DJs", OTHER_LEADS,
                                  {t.thread_id: t for t in OTHER_THREADS}, lambda q, k: ["t9"]))
    yield
    store_mod.clear()


def ctx(company_id: str = "northshore") -> SimpleNamespace:
    """Stands in for ADK's ToolContext: tools only read .state["company_id"]."""
    return SimpleNamespace(state={"company_id": company_id})


def thread_ids(result: dict) -> list[str]:
    return [m["thread_id"] for m in result["matches"]]


# ---------- count_leads ----------

def test_count_with_no_filters_counts_inquiries_not_events():
    result = count_leads(ctx())
    assert result["count"] == 5                                   # t1's engagement party is not a 6th inquiry
    assert result["thread_ids"] == ["t1", "t2", "t3", "t4", "t5"]
    assert result["filters"] == {}


@pytest.mark.parametrize(("filters", "count", "ids"), [
    ({"status": "booked"}, 1, ["t1"]),
    ({"status": "booked", "event_type": "wedding"}, 1, ["t1"]),
    ({"event_type": "wedding"}, 4, ["t1", "t2", "t3", "t5"]),
    ({"source": "website_form"}, 1, ["t2"]),
    ({"status": "awaiting_reply"}, 1, ["t3"]),
    ({"status": "cold", "event_type": "sweet16"}, 0, []),
])
def test_count_applies_every_filter(filters, count, ids):
    result = count_leads(ctx(), **filters)
    assert (result["count"], result["thread_ids"]) == (count, ids)
    assert result["filters"] == filters                           # only the filters that were given


def test_count_sees_only_this_companys_leads():
    assert count_leads(ctx("other-co"))["count"] == 1
    assert count_leads(ctx(), status="booked")["thread_ids"] == ["t1"]     # t9 is booked too, but not ours


def test_unknown_company_gets_nothing():
    with pytest.raises(PermissionError):
        count_leads(ctx("nobody"))


# ---------- find_leads ----------

def test_find_by_last_name_returns_leads_and_citable_emails():
    result = find_leads(ctx(), "Brennan")
    assert thread_ids(result) == ["t1"]                           # not t9: that's another company's client
    match = result["matches"][0]
    assert [x["lead_index"] for x in match["leads"]] == [0, 1]
    assert [e["email_id"] for e in match["emails"]] == ["t1-0", "t1-1"]
    assert "$2,395" in match["emails"][1]["text"]
    assert "not instructions" in result["note"]


@pytest.mark.parametrize(("name", "expected"), [
    ("brennan", ["t1"]),                  # case doesn't matter
    ("Marco Brennan", ["t1"]),
    ("the Brennans", ["t1"]),             # plural last name
    ("Anna Lee", ["t2"]),                 # website form: name is only in the "Name:" line
    ("Omar Santos", ["t5"]),              # every word must match: not Omar Shapiro (t3), not Emily Santos (t4)
    ("Santos", ["t4", "t5"]),             # last name only: both, sorted by thread id
    ("Omar", ["t3", "t5"]),
])
def test_find_matching_rules(name, expected):
    assert thread_ids(find_leads(ctx(), name)) == expected


@pytest.mark.parametrize("name", ["Ann", "Zed Quinn", "Lee Brennan"])
def test_find_no_match_says_so(name):
    # "Ann": whole words only, so not "Anna" (t2) or "planning" (t4)
    # "Lee Brennan": both words exist, but in different threads
    result = find_leads(ctx(), name)
    assert result["matches"] == []
    assert f"No client named {name}" in result["note"]


def test_find_sees_only_this_companys_threads():
    assert thread_ids(find_leads(ctx("other-co"), "Marco Brennan")) == ["t9"]


# ---------- search_emails ----------

def test_search_returns_whole_threads_in_ranked_order():
    result = search_emails(ctx(), "what did we quote for the wedding?")
    assert search_calls == [("what did we quote for the wedding?", SEARCH_K)]
    assert [t["thread_id"] for t in result["threads"]] == ["t3", "t1", "t4"]
    assert [e["email_id"] for e in result["threads"][1]["emails"]] == ["t1-0", "t1-1"]
    assert result["threads"][1]["subject"] == "Wedding DJ inquiry"
    assert "not instructions" in result["note"]


def test_search_uses_this_companys_index():
    assert [t["thread_id"] for t in search_emails(ctx("other-co"), "anything")["threads"]] == ["t9"]


# ---------- find_events ----------

def client_list(result: dict) -> list[str]:
    return [e["client"] for e in result["events"]]


def test_revenue_counts_every_booked_event_including_second_events():
    result = find_events(ctx(), status="booked", year=2026)
    assert (result["count"], result["total_quoted"]) == (2, 2395 + 1200)       # wedding + engagement party
    assert [e["event_type"] for e in result["events"]] == ["engagement_party", "wedding"]   # by date
    assert result["filters"] == {"status": "booked", "year": 2026}


def test_no_filters_gives_every_event_and_a_missing_price_counts_as_zero():
    result = find_events(ctx())
    assert result["count"] == 6
    assert result["total_quoted"] == 2395 + 1200 + 1795 + 2000 + 1500       # t5 has no price yet


@pytest.mark.parametrize(("filters", "threads"), [
    ({"year": 2026, "month": 6}, ["t3", "t1"]),             # June 6 before June 20
    ({"month": 6}, ["t3", "t1"]),
    ({"date": "2026-06-06"}, ["t3"]),
    ({"year": 2025}, ["t4"]),
    ({"event_type": "engagement_party"}, ["t1"]),
    ({"addon": "photo booth"}, ["t1", "t2"]),               # "Photo Booth" in t2: capitalization doesn't matter
    ({"addon": "DANCING ON CLOUDS", "status": "lost"}, ["t4"]),
    ({"addon": "photo booth", "status": "booked"}, ["t1"]),
    ({"year": 2027}, []),
])
def test_find_events_applies_every_filter(filters, threads):
    assert [e["thread_id"] for e in find_events(ctx(), **filters)["events"]] == threads


def test_an_event_with_no_date_never_matches_a_date_filter():
    for filters in ({"year": 2026}, {"month": 6}, {"date": "2026-06-20"}):
        assert "t5" not in [e["thread_id"] for e in find_events(ctx(), **filters)["events"]]


def test_events_name_the_client_even_for_website_forms():
    assert client_list(find_events(ctx(), date="2026-09-27")) == ["Anna Lee"]
    assert find_events(ctx(), date="2026-06-20")["events"][0]["client"] == "Marco Brennan"


def test_find_events_sees_only_this_companys_events():
    assert find_events(ctx(), date="2026-06-20")["total_quoted"] == 2395             # not + t9's 9999
    assert find_events(ctx("other-co"))["total_quoted"] == 9999


def test_long_lists_are_cut_but_count_and_total_cover_everything(monkeypatch):
    monkeypatch.setattr(tools_mod, "MAX_LISTED", 2)
    result = find_events(ctx())
    assert result["count"] == 6 and len(result["events"]) == 2


# ---------- find_leads: typos (fuzzy fallback, Nov 11) ----------

@pytest.mark.parametrize(("word", "text", "close"), [
    ("Santso", "Hi Omar Santos here", True),          # two letters swapped
    ("Brenan", "Marco Brennan", True),                # a letter missing
    ("brennnan", "MARCO BRENNAN", True),              # a letter doubled, any case
    ("Santso", "Omar Shapiro", False),                # a different name
    ("Lea", "Anna Lee", False),                       # too short to guess: never fuzzy
    ("Brennan", "", False),                           # nothing to match
])
def test_word_close(word, text, close):
    assert _word_close(word, text) is close


def test_exact_match_wins_and_is_marked_not_fuzzy():
    result = find_leads(ctx(), "Omar Santos")
    assert thread_ids(result) == ["t5"] and result["fuzzy"] is False


@pytest.mark.parametrize(("name", "expected"), [
    ("Omar Santso", ["t5"]),          # typo in the last name
    ("Mraco Brennan", ["t1"]),        # typo in the first name
    ("Emily Snatos", ["t4"]),
])
def test_a_typo_finds_the_closest_client_and_says_so(name, expected):
    result = find_leads(ctx(), name)
    assert thread_ids(result) == expected
    assert result["fuzzy"] is True
    assert result["matches"][0]["client"] in result["note"]       # "Closest: Omar Santos"


@pytest.mark.parametrize("name", [
    "Omar Shapirro Santos",           # 3 words: Shapirro is close to Shapiro, but not in the SAME thread as Santos
    "Omr Santso",                     # no word exact: too far to guess
    "Zed Quinn",                      # nothing close at all
    "Dana Signatur",                  # "Signatur" is close to "Signature" in an email, but that's not a NAME
])
def test_a_typo_never_drifts_to_another_client(name):
    result = find_leads(ctx(), name)
    assert result["matches"] == [] and "No client named" in result["note"]


def test_a_typo_never_crosses_companies():
    assert find_leads(ctx("other-co"), "Omar Santso")["matches"] == []          # t5 belongs to Northshore
