"""Tests for load_to_gmail.py: the import plan and the import loop (with a fake Gmail, no network)."""
import base64
import email
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import load_to_gmail as loader  # noqa: E402


def test_plan_matches_the_synthetic_inbox():
    planned = loader.plan()
    assert len(planned) == 939
    assert len({m.thread_key for m in planned}) == 200
    assert sum(m.from_company for m in planned) == 515


def test_plan_puts_every_reply_after_its_parent():
    planned = loader.plan()
    position = {m.message_id: i for i, m in enumerate(planned)}
    replies = [m for m in planned if m.in_reply_to in position]
    assert replies, "expected replies in the inbox"
    assert all(position[m.in_reply_to] < position[m.message_id] for m in replies)


def test_labels_company_mail_as_sent_and_client_mail_as_inbox():
    planned = loader.plan()
    assert {tuple(m.label_ids) for m in planned if m.from_company} == {("SENT",)}
    assert {tuple(m.label_ids) for m in planned if not m.from_company} == {("INBOX",)}


class FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code, self._payload, self.text = status_code, payload or {}, str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeGmail:
    """A tiny Gmail: remembers imported messages by Message-ID, like the real search does.

    Like the real API with these scopes, the import response carries only the message id.
    """

    def __init__(self, refuse_thread_id=False, first_status=None):
        self.calls, self.refuse_thread_id, self.first_status = [], refuse_thread_id, first_status
        self.by_message_id, self.thread_of_id, self.threads = {}, {}, 0

    def post(self, url, params=None, json=None, timeout=None):
        self.calls.append(json)
        if self.first_status and len(self.calls) == 1:
            return FakeResponse(self.first_status)
        if "threadId" in json:
            if self.refuse_thread_id:
                return FakeResponse(400, {"error": "invalid thread"})
            thread = json["threadId"]
        else:
            self.threads += 1
            thread = f"g{self.threads}"
        gmail_id = f"m{len(self.calls)}"
        message_id = email.message_from_bytes(base64.urlsafe_b64decode(json["raw"]))["Message-ID"].strip("<>")
        self.by_message_id[message_id] = gmail_id
        self.thread_of_id[gmail_id] = thread
        return FakeResponse(200, {"id": gmail_id})

    def get(self, url, params=None, timeout=None):
        if params and "q" in params:
            gmail_id = self.by_message_id.get(params["q"].removeprefix("rfc822msgid:"))
            found = [{"id": gmail_id, "threadId": self.thread_of_id[gmail_id]}] if gmail_id else []
            return FakeResponse(200, {"messages": found} if found else {"resultSizeEstimate": 0})
        return FakeResponse(200, {"id": url.rsplit("/", 1)[1], "threadId": self.thread_of_id[url.rsplit("/", 1)[1]]})


def _two_threads():
    planned = loader.plan()
    keys = []
    for m in planned:
        if m.thread_key not in keys:
            keys.append(m.thread_key)
        if len(keys) == 2:
            break
    return [m for m in planned if m.thread_key in keys]


def _run(gmail, planned, state, **kw):
    return loader.import_all(gmail, planned, state, sleep=lambda s: None, save=lambda s: None,
                             log=lambda *a: None, **kw)


def test_replies_reuse_the_gmail_thread_of_their_first_message():
    planned, state, gmail = _two_threads(), {}, FakeGmail()
    stats = _run(gmail, planned, state)
    assert stats["imported"] == len(planned)
    ours, gmails, split = loader.thread_check(state)
    assert (ours, gmails, split) == (2, 2, {})


def test_rerun_skips_what_is_already_imported():
    planned, state, gmail = _two_threads(), {}, FakeGmail()
    _run(gmail, planned, state)
    before = len(gmail.calls)
    stats = _run(gmail, planned, state)
    assert len(gmail.calls) == before and stats["skipped (already imported)"] == len(planned)


def test_lost_progress_never_duplicates_messages_already_in_gmail():
    """Tonight's crash: the first message got in, but progress wasn't saved."""
    planned, gmail = _two_threads(), FakeGmail()
    _run(gmail, planned, {}, limit=1)
    state = {}  # progress file lost
    stats = _run(gmail, planned, state)
    assert stats["found in Gmail, not re-imported"] == 1
    assert stats["imported"] == len(planned) - 1
    assert len(gmail.by_message_id) == len(planned)  # no duplicates
    assert loader.thread_check(state)[2] == {}


def test_limit_stops_after_n_new_messages():
    planned, state = _two_threads(), {}
    stats = _run(FakeGmail(), planned, state, limit=3)
    assert stats["imported"] == 3 and len(state) == 3


def test_falls_back_without_thread_id_when_gmail_refuses_it():
    planned, state = _two_threads(), {}
    stats = _run(FakeGmail(refuse_thread_id=True), planned, state)
    assert stats["imported"] == len(planned)
    assert stats["dropped threadId"] == len(planned) - 2  # every reply needed the fallback
    assert loader.thread_check(state)[2], "the split must be reported, not hidden"


def test_retries_after_rate_limit():
    planned, state = _two_threads()[:1], {}
    gmail = FakeGmail(first_status=429)
    stats = _run(gmail, planned, state)
    assert stats["imported"] == 1 and len(gmail.calls) == 2  # one 429, one success


def test_raw_message_keeps_threading_headers():
    m = next(p for p in loader.plan() if p.in_reply_to)
    raw = m.raw.decode("utf-8", "replace")
    assert "In-Reply-To:" in raw and "References:" in raw and m.message_id in raw
