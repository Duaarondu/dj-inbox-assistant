# tests/test_read_synthetic.py
"""Run the reader on the synthetic inbox and check it against the generator's known answers."""
import json
import mailbox
from collections import defaultdict
from pathlib import Path

import pytest

from dj_inbox.ingest import CompanyConfig, read_mbox
from dj_inbox.ingest.clean import strip_quoted

DATA = Path(__file__).resolve().parents[1] / "synthetic_inbox" / "data"
COMPANY = CompanyConfig(domains=["northshoresound.example"])


@pytest.fixture(scope="module")
def gold():
    threads = {json.loads(l)["thread_id"]: json.loads(l) for l in (DATA / "threads.jsonl").open(encoding="utf-8")}
    leads = [json.loads(l) for l in (DATA / "ground_truth_leads.jsonl").open(encoding="utf-8")]
    return threads, leads


@pytest.fixture(scope="module")
def read():
    return {t.thread_id: t for t in read_mbox(DATA / "inbox.mbox", COMPANY)}


def test_threads_match_the_generator(read, gold):
    threads, _ = gold
    assert {k: len(t.messages) for k, t in read.items()} == {k: len(t["messages"]) for k, t in threads.items()}


def test_threads_rebuild_without_gmail_ids(tmp_path, read):
    """Non-Gmail exports have no X-GM-THRID; References and subjects must still group correctly."""
    src, dst = mailbox.mbox(str(DATA / "inbox.mbox")), mailbox.mbox(str(tmp_path / "plain.mbox"))
    for msg in src:
        del msg["X-GM-THRID"]
        dst.add(msg)
    dst.flush()
    regrouped = read_mbox(tmp_path / "plain.mbox", COMPANY)
    groups = lambda ts: sorted(sorted(m.message_id for m in t.messages) for t in ts)
    assert groups(regrouped) == groups(read.values())


def test_reads_windows_line_endings(tmp_path, monkeypatch, read):
    """Regression: an .mbox written on Windows (\\r\\n, blank line after 'From ') parsed as header-less mail."""
    monkeypatch.setattr(mailbox, "linesep", b"\r\n")
    src, dst = mailbox.mbox(str(DATA / "inbox.mbox")), mailbox.mbox(str(tmp_path / "windows.mbox"))
    for msg in src:
        dst.add(msg)
    dst.flush()
    again = {t.thread_id: t for t in read_mbox(tmp_path / "windows.mbox", COMPANY)}
    assert again.keys() == read.keys()
    first = next(iter(read.values())).messages[0]
    assert again[first.thread_id].messages[0].body_clean == first.body_clean


def test_roles(read):
    for t in read.values():
        for m in t.messages:
            assert m.sender_role == ("company" if m.sender_email in {"mike@northshoresound.example",
                                                                     "dana@northshoresound.example"} else "client")


def test_no_quoted_history_left(read):
    for t in read.values():
        for m in t.messages:
            assert "wrote:" not in m.body_clean, m.message_id
            assert not any(l.lstrip().startswith(">") for l in m.body_clean.splitlines()), m.message_id


def test_no_company_signature_left(read):
    for t in read.values():
        for m in t.messages:
            assert "Northshore Sound Co.\n" not in m.body_clean and "Sent from my iPhone" not in m.body_clean


def test_no_client_contact_details_left(read, gold):
    _, leads = gold
    for lead in leads:
        text = read[lead["thread_id"]].clean_text()
        assert lead["client_email"] not in text, lead["thread_id"]
        assert lead["client_phone"] not in text, lead["thread_id"]


def test_gold_facts_survive_cleaning(read, gold):
    """Everything extraction needs must still be in the cleaned text."""
    _, leads = gold
    for lead in leads:
        text = read[lead["thread_id"]].clean_text()
        if lead["quoted_price"] is not None:
            p = lead["quoted_price"]
            assert f"${p:,}" in text or f"${p}" in text, (lead["thread_id"], p)
        if lead["guest_count"] is not None:
            assert str(lead["guest_count"]) in text, lead["thread_id"]
        if lead["venue"] and lead["venue"] != "Private residence":
            assert lead["venue"] in text, lead["thread_id"]


def test_client_phone_is_captured_when_present(read, gold):
    """If the client's phone was in the (unquoted) emails, the reader must have collected it."""
    threads, leads = gold
    for lead in leads:
        if lead["lead_index"]:
            continue
        digits = "".join(c for c in lead["client_phone"] if c.isdigit())
        raw = "\n".join(strip_quoted(m["body"]) for m in threads[lead["thread_id"]]["messages"])
        found = {p for m in read[lead["thread_id"]].messages for p in m.contacts.phones}
        assert (digits in found) == (lead["client_phone"] in raw), lead["thread_id"]


def test_cleaning_shrinks_the_text(read):
    msgs = [m for t in read.values() for m in t.messages]
    assert sum(len(m.body_clean) for m in msgs) < 0.6 * sum(len(m.body_raw) for m in msgs)
