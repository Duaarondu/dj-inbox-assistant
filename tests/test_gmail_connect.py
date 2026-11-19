# tests/test_gmail_connect.py
"""Connect Gmail (dj_inbox/gmail/connect.py + the /gmail/* endpoints) with fake Google and KMS.

Attacks covered: a forged or tampered state, a state started by another user or for another company, an
expired state, a redirect URI that isn't on the allow list, consent without the Gmail box ticked, a mailbox
already connected to another company, a non-owner, and the public demo (no sign-in) trying to connect.
"""
from __future__ import annotations

import base64
import uuid
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from dj_inbox.agent import store as store_mod
from dj_inbox.agent.answer import build_workflow
from dj_inbox.agent.store import InboxStore
from dj_inbox.agent.testing import FakeLlm
from dj_inbox.api.app import create_app
from dj_inbox.api.auth import TokenVerifier
from dj_inbox.gmail.connect import (
    GMAIL_SCOPE,
    ConnectConfig,
    ConnectError,
    GmailConnector,
    GoogleOAuth,
    Kms,
    MailboxTaken,
    NotOwner,
    make_state,
    read_state,
)
from tests.test_api import LEADS, T1, T2
from tests.test_auth import SUPABASE, FakeKeys, bearer, token

OWNER, OTHER = str(uuid.uuid4()), str(uuid.uuid4())
CALLBACK = "http://localhost:3000/gmail/callback"
CFG = ConnectConfig("client-123", "client-secret", b"state-secret", (CALLBACK,), "projects/p/keys/gmail-tokens")


class Resp:
    def __init__(self, status: int, body: dict | None = None):
        self.status_code, self.body = status, body or {}

    def json(self) -> dict:
        return self.body


class FakeGoogle:
    """Token endpoint, Gmail profile, revoke and KMS encrypt, all in one fake HTTP client."""
    def __init__(self, scope: str = GMAIL_SCOPE, refresh: str | None = "refresh-abc", token_status: int = 200):
        self.scope, self.refresh, self.token_status = scope, refresh, token_status
        self.posts: list[tuple[str, dict]] = []
        self.revoked: list[str] = []

    def post(self, url: str, **kw):
        self.posts.append((url, kw))
        if url.endswith("/token"):
            body = {"access_token": "access-xyz", "scope": self.scope}
            if self.refresh:
                body["refresh_token"] = self.refresh
            return Resp(self.token_status, body)
        if url.endswith("/revoke"):
            self.revoked.append(kw["data"]["token"])
            return Resp(200)
        if url.endswith(":encrypt"):
            plain = base64.b64decode(kw["json"]["plaintext"])
            aad = base64.b64decode(kw["json"]["additionalAuthenticatedData"])
            return Resp(200, {"ciphertext": base64.b64encode(b"ENC[" + aad + b"]" + plain[::-1]).decode()})
        raise AssertionError(url)

    def get(self, url: str, **kw):
        assert kw["headers"]["Authorization"] == "Bearer access-xyz"
        return Resp(200, {"emailAddress": "Owner@North.example", "historyId": "4242"})


class FakeDb:
    def __init__(self, owners: dict[str, str]):
        self.owners = owners                     # company_id -> owner user_id
        self.rows: dict[str, dict] = {}
        self.queued: list[str] = []

    def save(self, user_id, company_id, address, ciphertext, kms_key, history_id):
        if self.owners.get(company_id) != user_id:
            raise NotOwner
        if any(r["address"] == address for c, r in self.rows.items() if c != company_id):
            raise MailboxTaken
        self.rows[company_id] = {"address": address, "ciphertext": ciphertext, "history_id": history_id}

    def status(self, company_id):
        r = self.rows.get(company_id)
        return {"gmail_address": r["address"], "status": "importing"} if r else None


def connector(google: FakeGoogle, db: FakeDb) -> GmailConnector:
    return GmailConnector(CFG, GoogleOAuth(CFG, google), Kms(CFG.kms_key, google), db.save, db.status,
                          db.queued.append)


# ---------- the state ----------

def test_state_round_trip_gives_back_the_redirect_uri():
    assert read_state(CFG, make_state(CFG, OWNER, "c_a", CALLBACK), OWNER, "c_a") == CALLBACK


@pytest.mark.parametrize("user, company", [(OTHER, "c_a"), (OWNER, "c_b")])
def test_state_for_another_user_or_company_is_refused(user, company):
    with pytest.raises(ConnectError) as e:
        read_state(CFG, make_state(CFG, OWNER, "c_a", CALLBACK), user, company)
    assert e.value.status == 403


def test_tampered_or_forged_state_is_refused():
    body, _, sig = make_state(CFG, OWNER, "c_a", CALLBACK).partition(".")
    forged = make_state(ConnectConfig("x", "x", b"attacker-secret", (CALLBACK,), "k"), OWNER, "c_a", CALLBACK)
    for bad in (body + "x." + sig, body + "." + sig[:-2] + "AA", forged, "", "no-dot"):
        with pytest.raises(ConnectError):
            read_state(CFG, bad, OWNER, "c_a")


def test_expired_state_is_refused():
    state = make_state(CFG, OWNER, "c_a", CALLBACK, now=lambda: 1000)
    with pytest.raises(ConnectError, match="expired"):
        read_state(CFG, state, OWNER, "c_a", now=lambda: 1000 + 601)


# ---------- the flow ----------

def test_start_builds_a_read_only_offline_consent_url():
    url = connector(FakeGoogle(), FakeDb({})).start(OWNER, "c_a", CALLBACK)
    q = parse_qs(urlparse(url).query)
    assert q["scope"] == [GMAIL_SCOPE] and q["access_type"] == ["offline"] and q["prompt"] == ["consent"]
    assert q["redirect_uri"] == [CALLBACK] and q["client_id"] == ["client-123"]
    assert "client-secret" not in url


def test_start_refuses_a_redirect_uri_not_on_the_list():
    with pytest.raises(ConnectError):
        connector(FakeGoogle(), FakeDb({})).start(OWNER, "c_a", "https://evil.example/gmail/callback")


def test_finish_encrypts_the_token_bound_to_the_company_and_queues_the_import():
    google, db = FakeGoogle(), FakeDb({"c_a": OWNER})
    g = connector(google, db)
    state = make_state(CFG, OWNER, "c_a", CALLBACK)
    assert g.finish(OWNER, "c_a", "code-1", state) == "owner@north.example"
    row = db.rows["c_a"]
    assert b"refresh-abc" not in row["ciphertext"]                    # never stored in the clear
    assert row["ciphertext"].startswith(b"ENC[c_a]")                  # KMS was given the company id as AAD
    assert row["history_id"] == "4242" and db.queued == ["c_a"]
    token_call = next(kw for url, kw in google.posts if url.endswith("/token"))
    assert token_call["data"]["redirect_uri"] == CALLBACK             # the URI from the signed state


def test_consent_without_gmail_is_refused_and_nothing_is_revoked():
    google, db = FakeGoogle(scope="openid email"), FakeDb({"c_a": OWNER})
    with pytest.raises(ConnectError, match="Gmail access wasn't granted"):
        connector(google, db).finish(OWNER, "c_a", "code", make_state(CFG, OWNER, "c_a", CALLBACK))
    assert google.revoked == [] and db.rows == {} and db.queued == []


def test_a_used_or_bad_code_is_a_clear_error():
    google = FakeGoogle(token_status=400)
    with pytest.raises(ConnectError, match="didn't accept"):
        connector(google, FakeDb({"c_a": OWNER})).finish(OWNER, "c_a", "old", make_state(CFG, OWNER, "c_a", CALLBACK))


def test_a_mailbox_of_another_company_is_409_and_its_live_connection_is_left_alone():
    """Bug Journal #26: revoking the refused token revoked the whole grant, so the OTHER company's token died."""
    google, db = FakeGoogle(), FakeDb({"c_a": OWNER, "c_b": OTHER})
    db.rows["c_b"] = {"address": "owner@north.example", "ciphertext": b"x", "history_id": None}
    with pytest.raises(ConnectError) as e:
        connector(google, db).finish(OWNER, "c_a", "code", make_state(CFG, OWNER, "c_a", CALLBACK))
    assert e.value.status == 409 and google.revoked == [] and db.queued == []
    assert db.rows["c_b"]["ciphertext"] == b"x"


# ---------- the endpoints ----------

@pytest.fixture
def api():
    store_mod.clear()
    demo = InboxStore("northshore", "Northshore Sound Co.", LEADS, {"t1": T1, "t2": T2}, lambda q, k: ["t1"])
    store_mod.register(demo)
    google, db = FakeGoogle(), FakeDb({"c_a": OWNER})
    members = {OWNER: ["c_a"], OTHER: ["c_b"]}
    app = create_app(store=demo, workflow=build_workflow(FakeLlm()), model="gemini-3.5-flash-lite",
                     verifier=TokenVerifier(SUPABASE, keys=FakeKeys()), memberships=lambda u: members.get(u, []),
                     create_company=lambda u, n: "x", gmail=connector(google, db))
    with TestClient(app) as c:
        yield c, google, db
    store_mod.clear()


def test_full_connect_flow_over_http(api):
    c, google, db = api
    h = bearer(token(sub=OWNER))
    assert c.get("/gmail/status", headers=h).json() == {"connected": False}
    url = c.post("/gmail/connect", json={"redirect_uri": CALLBACK}, headers=h).json()["url"]
    state = parse_qs(urlparse(url).query)["state"][0]
    r = c.post("/gmail/connect/finish", json={"code": "code-1", "state": state}, headers=h)
    assert r.status_code == 200 and r.json() == {"gmail_address": "owner@north.example", "status": "importing"}
    assert c.get("/gmail/status", headers=h).json()["gmail_address"] == "owner@north.example"
    assert "refresh" not in r.text and "access-xyz" not in r.text


def test_the_public_demo_cannot_connect(api):
    c, _, _ = api
    assert c.post("/gmail/connect", json={"redirect_uri": CALLBACK}).status_code == 401
    assert c.get("/gmail/status").status_code == 401


def test_another_users_state_cannot_be_finished_in_my_session(api):
    """The attacker starts consent for THEIR company, then gets me to open the callback: refused."""
    c, _, db = api
    url = c.post("/gmail/connect", json={"redirect_uri": CALLBACK}, headers=bearer(token(sub=OTHER))).json()["url"]
    state = parse_qs(urlparse(url).query)["state"][0]
    r = c.post("/gmail/connect/finish", json={"code": "c", "state": state}, headers=bearer(token(sub=OWNER)))
    assert r.status_code == 403 and db.rows == {}


def test_not_configured_server_says_503():
    store_mod.clear()
    demo = InboxStore("northshore", "N", LEADS, {"t1": T1, "t2": T2}, lambda q, k: ["t1"])
    store_mod.register(demo)
    app = create_app(store=demo, workflow=build_workflow(FakeLlm()), model="gemini-3.5-flash-lite",
                     verifier=TokenVerifier(SUPABASE, keys=FakeKeys()), memberships=lambda u: ["c_a"],
                     create_company=lambda u, n: "x")
    with TestClient(app) as c:
        r = c.post("/gmail/connect", json={"redirect_uri": CALLBACK}, headers=bearer(token(sub=OWNER)))
    assert r.status_code == 503
    store_mod.clear()


def test_a_placeholder_client_id_stops_the_server(monkeypatch):
    """Bug Journal #24: GMAIL_CLIENT_ID=PASTE_CLIENT_ID was deployed twice; now the API refuses to start."""
    env = {"GMAIL_CLIENT_ID": "PASTE_CLIENT_ID", "GMAIL_CLIENT_SECRET": "s", "GMAIL_STATE_SECRET": "x",
           "GMAIL_REDIRECT_URIS": CALLBACK, "GMAIL_KMS_KEY": "k"}
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(SystemExit):
        ConnectConfig.from_env()
    monkeypatch.setenv("GMAIL_CLIENT_ID", "1090658566619-abc123.apps.googleusercontent.com")
    assert ConnectConfig.from_env().client_id.endswith(".apps.googleusercontent.com")


def test_a_failed_exchange_logs_googles_error_code_but_no_secret(caplog):
    class Http:
        def post(self, url, **kw):
            return Resp(401, {"error": "invalid_client", "error_description": "Unauthorized"})

    with caplog.at_level("WARNING", logger="dj_inbox.api"), pytest.raises(ConnectError):
        GoogleOAuth(CFG, Http()).exchange("code-xyz", CALLBACK)
    assert "invalid_client" in caplog.text
    assert "code-xyz" not in caplog.text and "client-secret" not in caplog.text


def test_sync_again_queues_an_import_only_when_connected(api):
    c, _, db = api
    h = bearer(token(sub=OWNER))
    assert c.post("/gmail/sync", headers=h).status_code == 404
    url = c.post("/gmail/connect", json={"redirect_uri": CALLBACK}, headers=h).json()["url"]
    state = parse_qs(urlparse(url).query)["state"][0]
    c.post("/gmail/connect/finish", json={"code": "c", "state": state}, headers=h)
    assert c.post("/gmail/sync", headers=h).json() == {"status": "importing"}
    assert db.queued == ["c_a", "c_a"]
    assert c.post("/gmail/sync").status_code == 401
