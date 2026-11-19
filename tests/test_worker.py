# tests/test_worker.py
"""The private worker (dj_inbox/worker/) and the Cloud Tasks hand-off (dj_inbox/gmail/tasks.py).

DB tests run the real import as app_ingest against a throwaway Postgres, with fake Gmail, KMS and models.
"""
from __future__ import annotations

import base64
import json
import mailbox
import os
import uuid
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import psycopg
import pytest
from fastapi.testclient import TestClient

from dj_inbox.extract.extract import ExtractedLead, ThreadLeads
from dj_inbox.gmail.reader import GmailError, GmailNotFound, GmailReader
from dj_inbox.gmail.tasks import enqueue_from_env, task_body
from dj_inbox.worker import app as worker_app
from dj_inbox.worker.importer import (
    GmailAccessLost,
    GmailSession,
    KmsDecrypt,
    RetryLater,
    companies_to_watch,
    company_for_gmail,
    run_import,
    run_sync,
    run_watch,
    set_status,
    start_watch,
)

URL = os.environ.get("TEST_DATABASE_URL")
needs_db = pytest.mark.skipif(not URL, reason="needs TEST_DATABASE_URL (a throwaway Postgres with pgvector)")
MBOX = Path(__file__).resolve().parents[1] / "synthetic_inbox" / "data" / "inbox.mbox"
COMPANY = "c_00000000000000aa"
OWNER = uuid.uuid4()


class Resp:
    def __init__(self, status: int, body: dict | None = None):
        self.status_code, self.body = status, body or {}

    def json(self) -> dict:
        return self.body


def gmail_messages(threads: set[str]) -> dict[str, dict]:
    """Synthetic emails of these threads as Gmail's API returns them (format=raw)."""
    out = {}
    for n, msg in enumerate(mailbox.mbox(str(MBOX))):
        thread = str(msg["X-GM-THRID"]).strip()
        if thread in threads:
            raw = base64.urlsafe_b64encode(msg.as_bytes()).decode().rstrip("=")
            out[f"g{n}"] = {"id": f"g{n}", "threadId": "G" + thread, "raw": raw}
    return out


class FakeGmail:
    """What GmailReader + run_import call on the session: profile, list, get."""
    def __init__(self, messages: dict[str, dict], fail_with: Exception | None = None):
        self.messages, self.fail_with, self.calls = messages, fail_with, 0

    def get(self, url: str, params: dict | None = None, **kw) -> Resp:
        self.calls += 1
        if self.fail_with:
            raise self.fail_with
        if url.endswith("/profile"):
            return Resp(200, {"emailAddress": "northshore.sound.demo@gmail.com", "historyId": "777"})
        if url.endswith("/messages"):
            return Resp(200, {"messages": [{"id": i} for i in self.messages]})
        return Resp(200, self.messages[url.rsplit("/", 1)[-1]])


def fake_oidc(email: str) -> str:
    """A token shaped like the one Cloud Run passes on (header.claims.signature); only the claims are read."""
    claims = base64.urlsafe_b64encode(json.dumps({"email": email, "email_verified": True}).encode()).decode()
    return "e30." + claims.rstrip("=") + ".sig"


def fake_decrypt(key: str, ciphertext: bytes, aad: bytes) -> bytes:
    assert ciphertext == b"ENC:refresh-abc"
    if aad != COMPANY.encode():
        raise RuntimeError("KMS decrypt refused (HTTP 400)")        # KMS checks the company id
    return b"refresh-abc"


def fake_embed(texts):
    return np.random.default_rng(0).normal(size=(len(texts), 768)).astype(np.float32)


def fake_extract(threads):
    return {t.thread_id: ThreadLeads(leads=[ExtractedLead(event_type="wedding", outcome="open")]) for t in threads}


class OnePool:
    def __init__(self, conn):
        self.conn = conn

    def connection(self):
        return nullcontext(self.conn)


@pytest.fixture
def db():
    from dj_inbox.db import CONNECT_KW, safe_connect
    from dj_inbox.db.migrate import migrate

    schema = f"test_{uuid.uuid4().hex[:10]}"
    with safe_connect(URL, **CONNECT_KW) as c:
        c.execute("create extension if not exists vector")
        c.execute(f"create schema {schema}")
        c.execute(f"set search_path = {schema}, public, extensions")
        from pgvector.psycopg import register_vector
        register_vector(c)
        migrate(c)
        c.execute("insert into companies (company_id, name, email_domains) values (%s, 'North', %s)",
                  (COMPANY, ["northshoresound.example"]))
        c.execute("""insert into gmail_connections (company_id, gmail_address, token_ciphertext, kms_key, status,
                     connected_by) values (%s, 'northshore.sound.demo@gmail.com', %s, 'k', 'importing', %s)""",
                  (COMPANY, b"ENC:refresh-abc", OWNER))
        c.execute("set role app_ingest")                 # the worker's database role
        try:
            yield c
        finally:
            c.execute("reset role")
            c.execute(f"drop schema {schema} cascade")


def status_row(conn):
    conn.execute("reset role")
    row = conn.execute("select status, error, history_id, last_import_at is not null from gmail_connections").fetchone()
    conn.execute("set role app_ingest")
    return row


def run(conn, gmail: FakeGmail):
    return run_import(OnePool(conn), COMPANY, decrypt=fake_decrypt, make_session=lambda refresh: gmail,
                      embed=fake_embed, extract=fake_extract, model="fake")


@needs_db
def test_import_runs_and_marks_the_connection_ready(db):
    gmail = FakeGmail(gmail_messages({"t001", "t002", "t003"}))
    result = run(db, gmail)
    assert result.status == "ready" and result.threads_total == 3 and result.leads == 3
    assert status_row(db) == ("ready", None, "777", True)
    from dj_inbox.db.inbox import load_inbox
    _, threads, _ = load_inbox(db, COMPANY)
    roles = {m.sender_role for t in threads for m in t.messages}
    assert roles == {"client", "company"}                # the company domain came from companies.email_domains


@needs_db
def test_a_second_run_fetches_nothing_new(db):
    messages = gmail_messages({"t001", "t002"})
    run(db, FakeGmail(messages))
    again = FakeGmail(messages)
    result = run(db, again)
    assert result.messages_new == 0 and again.calls == 2     # profile + list, no message fetched


@needs_db
def test_lost_gmail_access_asks_the_owner_to_reconnect(db):
    result = run(db, FakeGmail({}, fail_with=GmailAccessLost("invalid_grant")))
    assert result.status == "error"
    status, error, _, _ = status_row(db)
    assert status == "error" and "Reconnect Gmail" in error and "invalid_grant" not in error


@needs_db
def test_a_temporary_problem_is_retried_by_the_queue(db):
    with pytest.raises(RetryLater):
        run(db, FakeGmail({}, fail_with=RetryLater("rate limited")))
    assert status_row(db)[0] == "importing"


@needs_db
def test_a_token_only_decrypts_for_its_own_company(db):
    """A ciphertext copied into another company's row must not decrypt (KMS AAD = company id)."""
    db.execute("reset role")
    other = "c_00000000000000bb"
    db.execute("insert into companies (company_id, name) values (%s, 'Other')", (other,))
    db.execute("""insert into gmail_connections (company_id, gmail_address, token_ciphertext, kms_key, status,
                  connected_by) values (%s, 'other@x.example', %s, 'k', 'importing', %s)""",
               (other, b"ENC:refresh-abc", OWNER))
    db.execute("set role app_ingest")
    result = run_import(OnePool(db), other, decrypt=fake_decrypt, make_session=lambda r: pytest.fail("no session"),
                        embed=fake_embed, extract=fake_extract, model="fake")
    assert result.status == "error"


# ---------- pieces without a database ----------

def test_gmail_session_refreshes_once_and_reuses_the_token():
    class Http:
        def __init__(self):
            self.posts, self.gets = 0, []

        def post(self, url, data=None, timeout=None):
            self.posts += 1
            assert data["grant_type"] == "refresh_token" and data["refresh_token"] == "r"
            return Resp(200, {"access_token": "a1", "expires_in": 3600})

        def get(self, url, params=None, headers=None, timeout=None):
            self.gets.append(headers["Authorization"])
            return Resp(200, {})

    http = Http()
    s = GmailSession(http, "r", "id", "secret", clock=lambda: 1000.0)
    s.get("u1")
    s.get("u2")
    assert http.posts == 1 and http.gets == ["Bearer a1", "Bearer a1"]


def test_a_revoked_refresh_token_is_reported_as_lost_access():
    class Http:
        def post(self, url, data=None, timeout=None):
            return Resp(400, {"error": "invalid_grant"})

    with pytest.raises(GmailAccessLost, match="invalid_grant"):
        GmailSession(Http(), "r", "id", "secret").get("u")


def test_kms_decrypt_sends_the_company_as_aad_and_maps_errors():
    class Http:
        def __init__(self, status):
            self.status, self.sent = status, None

        def post(self, url, json=None, timeout=None):
            self.sent = json
            return Resp(self.status, {"plaintext": base64.b64encode(b"tok").decode()})

    ok = Http(200)
    assert KmsDecrypt(ok)("k", b"c", b"c_1") == b"tok"
    assert base64.b64decode(ok.sent["additionalAuthenticatedData"]) == b"c_1"
    with pytest.raises(RetryLater):
        KmsDecrypt(Http(503))("k", b"c", b"c_1")
    with pytest.raises(RuntimeError):
        KmsDecrypt(Http(400))("k", b"c", b"c_1")


def test_task_body_calls_the_worker_with_an_oidc_token_and_only_the_company_id():
    body = task_body("c_1", "https://worker.example/", "dj-inbox-tasks@p.iam.gserviceaccount.com")["task"]
    req = body["httpRequest"]
    assert req["url"] == "https://worker.example/tasks/import" and req["httpMethod"] == "POST"
    assert json.loads(base64.b64decode(req["body"])) == {"company_id": "c_1"}
    assert req["oidcToken"] == {"serviceAccountEmail": "dj-inbox-tasks@p.iam.gserviceaccount.com",
                                "audience": "https://worker.example"}
    assert body["dispatchDeadline"] == "1800s"


def test_without_queue_settings_enqueue_only_logs(monkeypatch):
    for name in ("TASKS_QUEUE", "WORKER_URL", "TASKS_INVOKER"):
        monkeypatch.delenv(name, raising=False)
    enqueue_from_env()("c_1")                            # no exception, no network


def test_worker_endpoint_validates_and_maps_retry_to_503(monkeypatch):
    calls = []

    def fake_run(pool, company_id, **kw):
        calls.append(company_id)
        if company_id.endswith("ff"):
            raise RetryLater("busy")
        from dj_inbox.worker.importer import ImportResult
        return ImportResult(company_id, "ready", 3, 200, 3, 209, 1.0)

    monkeypatch.setattr(worker_app, "run_import", fake_run)
    deps = {"pool": None, "decrypt": None, "make_session": None, "embed": None, "extract": None, "model": "m",
            "tasks_invoker": "tasks@x.iam.gserviceaccount.com"}
    auth = {"Authorization": "Bearer " + fake_oidc("tasks@x.iam.gserviceaccount.com")}
    with TestClient(worker_app.create_worker_app(deps), headers=auth) as c:
        assert c.post("/tasks/import", json={"company_id": "northshore"}).status_code == 422   # never the demo
        assert c.post("/tasks/import", json={"company_id": "c_00000000000000ff"}).status_code == 503
        r = c.post("/tasks/import", json={"company_id": "c_000000000000000a"})
        assert r.status_code == 200 and r.json()["status"] == "ready" and r.json()["leads"] == 209
    assert calls == ["c_00000000000000ff", "c_000000000000000a"]


def test_import_route_refuses_any_caller_but_cloud_tasks():
    deps = {"pool": None, "decrypt": None, "make_session": None, "embed": None, "extract": None, "model": "m",
            "tasks_invoker": "tasks@x.iam.gserviceaccount.com"}
    with TestClient(worker_app.create_worker_app(deps)) as c:
        body = {"company_id": "c_000000000000000a"}
        assert c.post("/tasks/import", json=body).status_code == 403                       # no token
        push = {"Authorization": "Bearer " + fake_oidc("push@x.iam.gserviceaccount.com")}
        assert c.post("/tasks/import", json=body, headers=push).status_code == 403         # Pub/Sub's identity


# ---- Automatic updates (milestone 5): history.list sync, push endpoint, watch ------------------------------

PUSH = "push@x.iam.gserviceaccount.com"
TASKS = "tasks@x.iam.gserviceaccount.com"


class FakeGmailSync(FakeGmail):
    """FakeGmail plus history.list (always answers with `added`, whatever the start id) and users.watch."""
    def __init__(self, messages: dict[str, dict], added: list[str] | None = None, history_status: int = 200):
        super().__init__(messages)
        self.added, self.history_status = added or [], history_status
        self.history_params: list[dict] = []
        self.fetched: list[str] = []

    def get(self, url: str, params: dict | None = None, **kw) -> Resp:
        if url.endswith("/history"):
            self.calls += 1
            self.history_params.append(dict(params or {}))
            if self.history_status != 200:
                return Resp(self.history_status, {"error": {"errors": [{"reason": "notFound"}]}})
            return Resp(200, {"historyId": "900", "history": [
                {"messagesAdded": [{"message": {"id": i, "labelIds": ["INBOX"]}}]} for i in self.added]})
        if not url.endswith(("/profile", "/messages")):
            self.fetched.append(url.rsplit("/", 1)[-1])
        return super().get(url, params, **kw)

    def post(self, url: str, json: dict | None = None, **kw) -> Resp:
        assert url.endswith("/watch") and json == {"topicName": "projects/p/topics/gmail-push"}
        return Resp(200, {"historyId": "901", "expiration": "1764000000000"})


def sync(conn, gmail):
    return run_sync(OnePool(conn), COMPANY, decrypt=fake_decrypt, make_session=lambda refresh: gmail,
                    embed=fake_embed, extract=fake_extract, model="fake")


def split(messages: dict[str, dict], thread: str) -> tuple[dict, dict]:
    first = {k: v for k, v in messages.items() if v["threadId"] != "G" + thread}
    return first, {k: v for k, v in messages.items() if v["threadId"] == "G" + thread}


def count_emails(conn) -> int:
    return conn.execute("select count(*) from emails").fetchone()[0]


@needs_db
def test_sync_imports_only_what_was_added_since_our_cursor(db):
    everything = gmail_messages({"t001", "t002"})
    before, new = split(everything, "t002")
    run(db, FakeGmail(before))                                         # cursor = 777 (the profile's id)
    gmail = FakeGmailSync(everything, added=list(new))
    result = sync(db, gmail)
    assert gmail.history_params[0]["startHistoryId"] == "777"
    assert result.status == "ready" and result.messages_new == len(new) and sorted(gmail.fetched) == sorted(new)
    assert status_row(db)[2] == "900"                                  # cursor moved to Gmail's newest id
    assert db.execute("select count(*) from emails where imported_at is null").fetchone()[0] == 0


@needs_db
def test_a_replayed_notification_adds_nothing(db):
    everything = gmail_messages({"t001", "t002"})
    before, new = split(everything, "t002")
    run(db, FakeGmail(before))
    sync(db, FakeGmailSync(everything, added=list(new)))
    stored = count_emails(db)
    replay = FakeGmailSync(everything, added=list(new))                # Pub/Sub redelivers the same change
    result = sync(db, replay)
    assert result.messages_new == 0 and replay.fetched == [] and count_emails(db) == stored


@needs_db
def test_history_too_old_falls_back_to_a_full_import(db):
    messages = gmail_messages({"t001"})
    run(db, FakeGmail({}))                                             # connected, cursor 777, nothing imported
    result = sync(db, FakeGmailSync(messages, history_status=404))
    assert result.status == "ready" and result.messages_new == len(messages)
    assert status_row(db)[2] == "777"


@needs_db
def test_no_cursor_yet_means_a_full_import(db):
    messages = gmail_messages({"t001"})
    gmail = FakeGmailSync(messages)
    result = sync(db, gmail)                                           # fixture row has history_id = null
    assert result.messages_new == len(messages) and gmail.history_params == []


@needs_db
def test_the_cursor_never_moves_backwards(db):
    set_status(db, COMPANY, "ready", history_id="900")
    set_status(db, COMPANY, "ready", history_id="850")                 # a slower, older sync finishing late
    assert status_row(db)[2] == "900"
    set_status(db, COMPANY, "ready", history_id="1000")
    assert status_row(db)[2] == "1000"


@needs_db
def test_worker_finds_the_company_of_a_mailbox_but_the_api_cannot_ask(db):
    pool = OnePool(db)
    assert company_for_gmail(pool, "Northshore.Sound.Demo@gmail.com") == COMPANY
    assert company_for_gmail(pool, "someone@else.example") is None
    assert companies_to_watch(pool) == [COMPANY]                       # connected (status importing)
    db.execute("reset role")
    db.execute("set role app_api")
    try:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            db.execute("select gmail_company_for('northshore.sound.demo@gmail.com')")
    finally:
        db.execute("reset role")
        db.execute("set role app_ingest")


@needs_db
def test_watch_saves_when_it_expires(db):
    gmail = FakeGmailSync({})
    outcome = run_watch(OnePool(db), COMPANY, decrypt=fake_decrypt, make_session=lambda r: gmail,
                        topic="projects/p/topics/gmail-push")
    assert outcome == "watching"
    db.execute("reset role")
    expires = db.execute("select watch_expires_at from gmail_connections").fetchone()[0]
    db.execute("set role app_ingest")
    assert expires == datetime.fromtimestamp(1764000000, tz=UTC)


def test_history_list_filters_like_the_import_query_and_follows_pages():
    pages = {None: {"historyId": "50", "nextPageToken": "p2", "history": [
                {"messagesAdded": [{"message": {"id": "a", "labelIds": ["INBOX"]}}]},
                {"messagesAdded": [{"message": {"id": "spam", "labelIds": ["SPAM"]}}]},
                {"messagesAdded": [{"message": {"id": "draft", "labelIds": ["DRAFT"]}}]}]},
             "p2": {"historyId": "60", "history": [
                {"messagesAdded": [{"message": {"id": "a", "labelIds": ["INBOX"]}}]},
                {"messagesAdded": [{"message": {"id": "reply", "labelIds": ["SENT"]}}]}]}}

    class Pages:
        def get(self, url, params=None):
            return Resp(200, pages[params.get("pageToken")])

    ids, latest = GmailReader(Pages(), min_interval=0).list_new_ids("40")
    assert ids == ["a", "reply"] and latest == "60"                   # our own replies count; spam/drafts don't


def test_too_old_history_and_deleted_messages_are_told_apart_from_errors():
    class Gone:
        def get(self, url, params=None):
            return Resp(404, {"error": {"errors": [{"reason": "notFound"}]}})

    reader = GmailReader(Gone(), min_interval=0)
    with pytest.raises(GmailNotFound):
        reader.list_new_ids("1")
    assert reader.fetch(["x", "y"], skip_missing=True) == []


def test_start_watch_maps_gmails_answers():
    class Answer:
        def __init__(self, status, body):
            self.resp = Resp(status, body)

        def post(self, url, json=None):
            return self.resp

    assert start_watch(Answer(200, {"expiration": "1764000000000"}), "t") == datetime.fromtimestamp(1764000000,
                                                                                                    tz=UTC)
    with pytest.raises(RetryLater):
        start_watch(Answer(503, {}), "t")
    with pytest.raises(GmailError):
        start_watch(Answer(403, {"error": "forbidden"}), "t")


def push_envelope(address: str, history_id: int = 901) -> dict:
    data = base64.b64encode(json.dumps({"emailAddress": address, "historyId": history_id}).encode()).decode()
    return {"message": {"data": data, "messageId": "m1"}, "subscription": "projects/p/subscriptions/s"}


def push_app(monkeypatch, outcome="ready"):
    synced = []

    def fake_sync(pool, company_id, **kw):
        synced.append(company_id)
        if outcome == "retry":
            raise RetryLater("busy")
        from dj_inbox.worker.importer import ImportResult
        return ImportResult(company_id, "ready", 1, 0, 1, 1, 0.5)

    monkeypatch.setattr(worker_app, "run_sync", fake_sync)
    monkeypatch.setattr(worker_app, "company_for_gmail",
                        lambda pool, a: COMPANY if a == "northshore.sound.demo@gmail.com" else None)
    deps = {"pool": None, "decrypt": None, "make_session": None, "embed": None, "extract": None, "model": "m",
            "push_invoker": PUSH, "tasks_invoker": TASKS}
    return worker_app.create_worker_app(deps), synced


def as_caller(email: str) -> dict:
    return {"Authorization": "Bearer " + fake_oidc(email)}


def test_push_syncs_the_mailboxs_company(monkeypatch):
    app, synced = push_app(monkeypatch)
    with TestClient(app) as c:
        r = c.post("/pubsub/gmail", json=push_envelope("Northshore.Sound.Demo@gmail.com"), headers=as_caller(PUSH))
    assert r.status_code == 200 and r.json()["new_messages"] == 1 and synced == [COMPANY]


def test_push_only_from_pubsubs_identity(monkeypatch):
    app, synced = push_app(monkeypatch)
    with TestClient(app) as c:
        env = push_envelope("northshore.sound.demo@gmail.com")
        assert c.post("/pubsub/gmail", json=env).status_code == 403
        assert c.post("/pubsub/gmail", json=env, headers=as_caller(TASKS)).status_code == 403
    assert synced == []


def test_push_acks_what_retrying_cannot_fix_and_retries_the_rest(monkeypatch):
    app, synced = push_app(monkeypatch)
    with TestClient(app, headers=as_caller(PUSH)) as c:
        assert c.post("/pubsub/gmail", json={"message": {"data": "bm90IGpzb24="}}).json()["ignored"]
        assert c.post("/pubsub/gmail", json=push_envelope("stranger@x.example")).json()["ignored"] == \
            "unknown mailbox"
    assert synced == []
    app, synced = push_app(monkeypatch, outcome="retry")
    with TestClient(app, headers=as_caller(PUSH)) as c:
        assert c.post("/pubsub/gmail", json=push_envelope("northshore.sound.demo@gmail.com")).status_code == 503


def test_renewal_only_from_the_scheduler_and_counts_outcomes(monkeypatch):
    monkeypatch.setattr(worker_app, "companies_to_watch", lambda pool: ["c_1", "c_2"])
    monkeypatch.setattr(worker_app, "run_watch", lambda pool, cid, **kw: "watching" if cid == "c_1" else "error")
    deps = {"pool": None, "decrypt": None, "make_session": None, "embed": None, "extract": None, "model": "m",
            "push_invoker": PUSH, "tasks_invoker": TASKS, "watch_topic": "projects/p/topics/gmail-push"}
    with TestClient(worker_app.create_worker_app(deps)) as c:
        assert c.post("/tasks/renew-watches", headers=as_caller(PUSH)).status_code == 403
        r = c.post("/tasks/renew-watches", headers=as_caller(TASKS))
    assert r.status_code == 200 and r.json() == {"watching": 1, "error": 1}
