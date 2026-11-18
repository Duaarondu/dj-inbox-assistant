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
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from dj_inbox.extract.extract import ExtractedLead, ThreadLeads
from dj_inbox.gmail.tasks import enqueue_from_env, task_body
from dj_inbox.worker import app as worker_app
from dj_inbox.worker.importer import GmailAccessLost, GmailSession, KmsDecrypt, RetryLater, run_import

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
    deps = {"pool": None, "decrypt": None, "make_session": None, "embed": None, "extract": None, "model": "m"}
    with TestClient(worker_app.create_worker_app(deps)) as c:
        assert c.post("/tasks/import", json={"company_id": "northshore"}).status_code == 422   # never the demo
        assert c.post("/tasks/import", json={"company_id": "c_00000000000000ff"}).status_code == 503
        r = c.post("/tasks/import", json={"company_id": "c_000000000000000a"})
        assert r.status_code == 200 and r.json()["status"] == "ready" and r.json()["leads"] == 209
    assert calls == ["c_00000000000000ff", "c_000000000000000a"]
