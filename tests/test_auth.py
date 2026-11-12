# tests/test_auth.py
"""Sign-in: every way a token can be wrong must be refused, and a user only ever sees their own company.

Tokens are minted here with a test ES256 key (the same kind Supabase uses); a fake key source plays the
JWKS endpoint. No network, no Supabase.
"""
from __future__ import annotations

import json
import logging
import time
import uuid

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient

from dj_inbox.agent import store as store_mod
from dj_inbox.agent.answer import build_workflow
from dj_inbox.agent.store import InboxStore
from dj_inbox.agent.testing import FakeLlm, fake_text
from dj_inbox.api.app import create_app
from dj_inbox.api.auth import AuthError, TokenVerifier, memberships_for, resolve_caller
from tests.test_api import LEADS, T1, T2, T9

SUPABASE = "https://testproject.supabase.co"
ISSUER = SUPABASE + "/auth/v1"
KID = "test-key-1"
REAL_KEY = ec.generate_private_key(ec.SECP256R1())
ATTACKER_KEY = ec.generate_private_key(ec.SECP256R1())
OTHER_USER = str(uuid.uuid4())       # member of other-co
NEW_USER = str(uuid.uuid4())         # signed in, no company yet


class FakeKeys:
    """Plays Supabase's JWKS endpoint: knows only REAL_KEY's public half, under KID."""

    class _Key:
        def __init__(self, key):
            self.key = key

    def __init__(self, fail: Exception | None = None):
        self.fail = fail

    def get_signing_key_from_jwt(self, token):
        if self.fail:
            raise self.fail
        if jwt.get_unverified_header(token).get("kid") != KID:
            raise jwt.PyJWKClientError("Unable to find a signing key that matches")
        return self._Key(REAL_KEY.public_key())


def token(sub: str = OTHER_USER, key=REAL_KEY, alg: str = "ES256", kid: str = KID, **claims) -> str:
    now = int(time.time())
    payload = {"sub": sub, "aud": "authenticated", "iss": ISSUER, "iat": now, "exp": now + 3600,
               "role": "authenticated", **claims}
    return jwt.encode(payload, key, algorithm=alg, headers={"kid": kid})


def lookup(user_id: str) -> list[str]:
    return {OTHER_USER: ["other-co"]}.get(user_id, [])


@pytest.fixture
def client():
    store_mod.clear()
    demo = InboxStore("northshore", "Northshore Sound Co.", LEADS, {"t1": T1, "t2": T2}, lambda q, k: ["t1"])
    store_mod.register(demo)
    store_mod.register(InboxStore("other-co", "Other", [], {"t9": T9}, lambda q, k: ["t9"]))
    app = create_app(store=demo, workflow=build_workflow(FakeLlm(script=[fake_text("Hello from your inbox.")])),
                     model="gemini-3.5-flash-lite", verifier=TokenVerifier(SUPABASE, keys=FakeKeys()),
                     memberships=lookup)
    with TestClient(app) as c:
        yield c
    store_mod.clear()


def bearer(t: str) -> dict:
    return {"Authorization": f"Bearer {t}"}


# ---------- who gets which company ----------

def test_no_token_gets_the_public_demo(client):
    assert client.get("/threads/t1").status_code == 200
    assert client.get("/threads/t9").status_code == 404


def test_a_signed_in_user_gets_only_their_own_company(client):
    h = bearer(token())
    assert client.get("/threads/t9", headers=h).status_code == 200      # their own thread
    assert client.get("/threads/t1", headers=h).status_code == 404      # the demo's thread: not theirs
    assert client.get("/leads", headers=h).json()["count"] == 0         # other-co has no leads, not the demo's 2


def test_signed_in_but_no_company_yet_is_403(client):
    r = client.get("/leads", headers=bearer(token(sub=NEW_USER)))
    assert r.status_code == 403 and "company" in r.json()["detail"]


# ---------- every broken token is a 401, never the demo ----------

BAD_TOKENS = {
    "expired": lambda: token(exp=int(time.time()) - 3600, iat=int(time.time()) - 7200),
    "wrong audience": lambda: token(aud="anon"),
    "wrong issuer": lambda: token(iss="https://evil.example/auth/v1"),
    "signed by another key": lambda: token(key=ATTACKER_KEY),
    "unknown key id": lambda: token(kid="someone-elses-key"),
    "sub is not a uuid": lambda: token(sub="admin"),
    "alg none": lambda: jwt.encode({"sub": OTHER_USER, "aud": "authenticated", "iss": ISSUER,
                                    "iat": int(time.time()), "exp": int(time.time()) + 3600},
                                   None, algorithm="none", headers={"kid": KID}),
    "HS256 with a guessed secret": lambda: token(key="super-secret-jwt-token-with-at-least-32-characters",
                                                 alg="HS256"),
    "garbage": lambda: "not.a.token",
}


@pytest.mark.parametrize("name", BAD_TOKENS)
def test_bad_tokens_are_refused(client, name):
    r = client.get("/threads/t9", headers=bearer(BAD_TOKENS[name]()))
    assert r.status_code == 401, name
    assert r.headers.get("www-authenticate") == "Bearer"


def test_hs256_signed_with_the_public_key_is_refused():
    """Algorithm confusion: the attacker knows the PUBLIC key and uses it as an HMAC secret."""
    pem = REAL_KEY.public_key().public_bytes(serialization.Encoding.PEM,
                                             serialization.PublicFormat.SubjectPublicKeyInfo)
    header = jwt.utils.base64url_encode(json.dumps({"alg": "HS256", "kid": KID, "typ": "JWT"}).encode())
    body = jwt.utils.base64url_encode(json.dumps({"sub": OTHER_USER, "aud": "authenticated", "iss": ISSUER,
                                                  "iat": int(time.time()), "exp": int(time.time()) + 3600}).encode())
    import hashlib
    import hmac
    sig = jwt.utils.base64url_encode(hmac.new(pem, header + b"." + body, hashlib.sha256).digest())
    forged = (header + b"." + body + b"." + sig).decode()
    with pytest.raises(AuthError) as e:
        TokenVerifier(SUPABASE, keys=FakeKeys()).verify(forged)
    assert e.value.status == 401


@pytest.mark.parametrize("header", ["Basic dXNlcjpwYXNz", "Bearer", "Bearer   ", "Token abc"])
def test_malformed_authorization_headers_are_401(client, header):
    assert client.get("/leads", headers={"Authorization": header}).status_code == 401


def test_unreachable_key_server_fails_closed_with_503():
    verifier = TokenVerifier(SUPABASE, keys=FakeKeys(fail=jwt.PyJWKClientConnectionError("timeout")))
    with pytest.raises(AuthError) as e:
        verifier.verify(token())
    assert e.value.status == 503


def test_a_token_is_refused_when_sign_in_is_not_configured():
    with pytest.raises(AuthError) as e:
        resolve_caller(f"Bearer {token()}", verifier=None, lookup=None, demo_company="northshore")
    assert e.value.status == 401


# ---------- what never leaks ----------

def test_tokens_never_appear_in_logs(client, caplog):
    good, bad = token(), token(key=ATTACKER_KEY)
    with caplog.at_level(logging.DEBUG):
        assert client.post("/ask", json={"question": "hi"}, headers=bearer(good)).status_code == 200
        assert client.get("/leads", headers=bearer(bad)).status_code == 401
    assert good not in caplog.text and bad not in caplog.text
    ask_line = json.loads(next(r.message for r in caplog.records if '"event": "ask"' in r.message))
    assert ask_line["company_id"] == "other-co" and ask_line["signed_in"] is True


def test_health_stays_public_and_describes_only_the_demo(client):
    assert client.get("/health").json()["company"] == "Northshore Sound Co."


# ---------- the membership query keeps its own WHERE ----------

class RecordingConn:
    """Captures SQL, to prove the lookup filters on user_id itself (the table owner would skip RLS)."""

    def __init__(self):
        self.sql = []

    def transaction(self):
        import contextlib
        return contextlib.nullcontext()

    def execute(self, sql, params=None):
        self.sql.append((sql, params))

        class _Rows:
            def fetchall(self_inner):
                return [("other-co",)]
        return _Rows()


def test_membership_lookup_filters_on_user_id_even_without_rls():
    conn = RecordingConn()
    assert memberships_for(conn, OTHER_USER) == ["other-co"]
    select_sql, params = conn.sql[-1]
    assert "where user_id = %s" in select_sql and params == (OTHER_USER,)
    assert conn.sql[0] == ("select set_config('app.user_id', %s, true)", (OTHER_USER,))
