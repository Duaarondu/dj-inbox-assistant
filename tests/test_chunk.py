# tests/test_chunk.py
"""The spec for chunk_thread. Make these pass; don't edit them to make them pass."""
from datetime import UTC, datetime

from dj_inbox.ingest import Contacts, Email, EmailThread
from dj_inbox.rag.chunk import chunk_inbox, chunk_thread


def email(i: int, role: str, name: str, body: str) -> Email:
    return Email(message_id=f"<m{i}@x>", thread_id="t055", sent_at=datetime(2025, 11, 21 + i, 16, 53, tzinfo=UTC),
                 sender_role=role, sender_name=name, sender_email="x@y.example", subject="Fwd: DJ? (Brennan wedding)",
                 body_raw="raw", body_clean=body, contacts=Contacts())


THREAD = EmailThread(thread_id="t055", subject="Fwd: DJ? (Brennan wedding)", messages=[
    email(0, "client", "Gina Romano", "Forwarding this couple. They're getting married August 23, 2026."),
    email(1, "company", "Dana Kim", "- Premier package (6 hours): $2,395\n\nThat comes to $2,395 all in."),
])


def test_one_chunk_per_email_in_order():
    chunks = chunk_thread(THREAD)
    assert [c.chunk_id for c in chunks] == ["t055-0", "t055-1"]
    assert [c.message_id for c in chunks] == ["<m0@x>", "<m1@x>"]
    assert all(c.thread_id == "t055" for c in chunks)


def test_header_then_body():
    first, second = chunk_thread(THREAD)
    assert first.text == ("Fwd: DJ? (Brennan wedding) | CLIENT Gina Romano | 2025-11-21\n"
                          "Forwarding this couple. They're getting married August 23, 2026.")
    assert second.text.startswith("Fwd: DJ? (Brennan wedding) | COMPANY Dana Kim | 2025-11-22\n")


def test_price_lines_stay_together():
    """Project rule: never split a quote across chunks."""
    assert "Premier package (6 hours): $2,395\n\nThat comes to $2,395 all in." in chunk_thread(THREAD)[1].text


def test_sent_at_is_kept():
    assert chunk_thread(THREAD)[1].sent_at == datetime(2025, 11, 22, 16, 53, tzinfo=UTC)


def test_chunk_inbox_flattens_threads():
    assert len(chunk_inbox([THREAD, THREAD])) == 4
