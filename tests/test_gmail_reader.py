# tests/test_gmail_reader.py
"""dj_inbox/gmail/reader.py against a fake Gmail API, and read_messages' thread-id rule."""
from __future__ import annotations

import base64
from email.message import EmailMessage

import pytest

from dj_inbox.gmail.reader import MESSAGES_URL, GmailError, GmailReader
from dj_inbox.ingest import CompanyConfig, read_messages

COMPANY = CompanyConfig(domains=["northshoresound.example"])


class Resp:
    def __init__(self, status: int, body: dict | None = None):
        self.status_code, self.body = status, body or {}

    def json(self) -> dict:
        return self.body


class FakeGmail:
    """Answers list and get calls from a script; records every call."""
    def __init__(self, script: dict[str, list[Resp]]):
        self.script, self.calls = script, []

    def get(self, url: str, params: dict | None = None) -> Resp:
        self.calls.append((url, dict(params or {})))
        key = url.rsplit("/", 1)[-1] + ("@" + params["pageToken"] if params and "pageToken" in params else "")
        return self.script[key].pop(0)


def raw_email(message_id: str, sender: str, subject: str, body: str, extra: dict | None = None) -> str:
    m = EmailMessage()
    m["Message-ID"], m["From"], m["To"] = message_id, sender, "info@northshoresound.example"
    m["Subject"], m["Date"] = subject, "Tue, 03 Mar 2026 10:00:00 -0500"
    for k, v in (extra or {}).items():
        m[k] = v
    m.set_content(body)
    return base64.urlsafe_b64encode(m.as_bytes().replace(b"\n", b"\r\n")).decode().rstrip("=")


def test_list_follows_pages_and_sends_the_filter():
    gmail = FakeGmail({"messages": [Resp(200, {"messages": [{"id": "a"}, {"id": "b"}], "nextPageToken": "p2"})],
                       "messages@p2": [Resp(200, {"messages": [{"id": "c"}]})]})
    assert GmailReader(gmail).list_ids() == ["a", "b", "c"]
    assert "-in:spam" in gmail.calls[0][1]["q"] and "-category:promotions" in gmail.calls[0][1]["q"]


def test_rate_limits_are_retried_then_succeed():
    gmail = FakeGmail({"messages": [Resp(429), Resp(503), Resp(200, {"messages": [{"id": "a"}]})]})
    waits: list[float] = []
    assert GmailReader(gmail, sleep=waits.append, min_interval=0).list_ids() == ["a"]
    assert len(waits) == 2


def rate_limited() -> Resp:
    return Resp(403, {"error": {"code": 403, "errors": [{"reason": "userRateLimitExceeded"}]}})


def test_gmails_403_rate_limit_is_retried():
    """Gmail says "slow down" with a 403 + reason, not only 429 (seen Nov 17 on the full 942-message import)."""
    gmail = FakeGmail({"m1": [rate_limited(), rate_limited(),
                              Resp(200, {"id": "m1", "threadId": "T1", "raw": raw_email("<m1@x>", "k@x", "S", "B")})]})
    waits: list[float] = []
    assert GmailReader(gmail, sleep=waits.append, min_interval=0).get("m1")[1] == "T1"
    assert len(waits) == 2


def test_requests_are_paced_across_threads():
    clock = [0.0]
    waits: list[float] = []
    reader = GmailReader(None, sleep=waits.append, min_interval=0.1, clock=lambda: clock[0])
    for _ in range(3):
        reader._pace()                       # three requests at the same instant
    assert waits == pytest.approx([0.1, 0.2])


def test_retry_after_from_gmail_is_respected():
    slow = Resp(429)
    slow.headers = {"Retry-After": "7"}
    gmail = FakeGmail({"messages": [slow, Resp(200, {"messages": []})]})
    waits: list[float] = []
    GmailReader(gmail, sleep=waits.append, min_interval=0).list_ids()
    assert waits == [7.0]


def test_progress_is_reported():
    raw = raw_email("<m@x>", "k@x", "S", "B")
    gmail = FakeGmail({f"m{i}": [Resp(200, {"id": f"m{i}", "threadId": "T", "raw": raw})] for i in range(3)})
    lines: list[str] = []
    GmailReader(gmail, min_interval=0, progress=lines.append).fetch(["m0", "m1", "m2"])
    assert lines and "3/3" in lines[-1]


def test_a_permission_403_is_not_retried_and_names_the_reason():
    gmail = FakeGmail({"messages": [Resp(403, {"error": {"errors": [{"reason": "insufficientPermissions"}]}})]})
    with pytest.raises(GmailError, match="insufficientPermissions"):
        GmailReader(gmail, sleep=lambda s: pytest.fail("must not retry")).list_ids()


def test_a_real_error_stops_and_says_nothing_secret():
    gmail = FakeGmail({"messages": [Resp(403, {"error": "insufficient scope"})]})
    with pytest.raises(GmailError, match="HTTP 403") as e:
        GmailReader(gmail, sleep=lambda s: None).list_ids()
    assert "Bearer" not in str(e.value)


def test_get_decodes_raw_and_returns_gmails_thread_id():
    raw = raw_email("<m1@x>", "Kim Lee <kim@x.example>", "Wedding DJ?", "Hi, are you free June 6?")
    gmail = FakeGmail({"m1": [Resp(200, {"id": "m1", "threadId": "T1", "raw": raw})]})
    msg, thread_id, gmail_id = GmailReader(gmail).get("m1")
    assert (thread_id, gmail_id, msg["Subject"]) == ("T1", "m1", "Wedding DJ?")
    assert gmail.calls[0] == (f"{MESSAGES_URL}/m1", {"format": "raw"})


def test_gmails_thread_id_beats_a_header_the_sender_wrote():
    """X-GM-THRID inside a raw email is just text anyone can put there; the API's threadId decides."""
    raw = raw_email("<m1@x>", "kim@x.example", "Hi", "Hello", {"X-GM-THRID": "someone-elses-thread"})
    gmail = FakeGmail({"m1": [Resp(200, {"id": "m1", "threadId": "T1", "raw": raw})]})
    threads, gmail_ids = read_messages([GmailReader(gmail).get("m1")], COMPANY)
    assert [t.thread_id for t in threads] == ["T1"]
    assert gmail_ids == {"<m1@x>": "m1"}


def test_company_addresses_make_a_shared_domain_safe():
    """On gmail.com, only the connected account is the company; other gmail.com senders are clients."""
    company = CompanyConfig(domains=[], addresses=["owner@gmail.com"])
    assert company.role_of("Owner@Gmail.com") == "company"
    assert company.role_of("client@gmail.com") == "client"
