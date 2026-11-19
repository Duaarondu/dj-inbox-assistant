# dj_inbox/gmail/connect.py
"""Connect Gmail: the owner gives read-only access; we keep an ENCRYPTED refresh token for the worker.

    1. POST /gmail/connect         -> consent URL on the `gmail-connector` client, with a signed `state`
    2. Google -> web /gmail/callback?code&state -> POST /gmail/connect/finish (with the sign-in token)
    3. finish: check the state, swap the code for tokens, require gmail.readonly, read the mailbox address,
       encrypt the refresh token with Cloud KMS (bound to the company id), save it via
       save_gmail_connection() (migration 006: owner only), queue the import for the worker.

Why the signed state: without it, an attacker could send you a link that finishes THEIR consent in YOUR
session (your company would import their mailbox), or replay an old code. The state is an HMAC over
(user, company, redirect URI, nonce, expiry), checked against the signed-in caller at step 3.

The public API can only ENCRYPT with the key (IAM, Nov 17); only the worker can decrypt.
Nothing here logs or returns a token.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlencode

log = logging.getLogger("dj_inbox.api")

GMAIL_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REVOKE_URL = "https://oauth2.googleapis.com/revoke"
PROFILE_URL = "https://gmail.googleapis.com/gmail/v1/users/me/profile"
KMS_URL = "https://cloudkms.googleapis.com/v1/{key}:encrypt"
STATE_TTL_S = 600
CLIENT_ID = re.compile(r"^[0-9]+-[a-z0-9]+\.apps\.googleusercontent\.com$")


class ConnectError(Exception):
    """Something the owner can fix by trying again; `detail` is safe to show (never a token or a code)."""
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


@dataclass(frozen=True)
class ConnectConfig:
    client_id: str
    client_secret: str
    state_secret: bytes
    redirect_uris: tuple[str, ...]        # exact allow list, same as on the Google client
    kms_key: str                          # projects/.../locations/.../keyRings/dj-inbox/cryptoKeys/gmail-tokens

    @classmethod
    def from_env(cls) -> ConnectConfig | None:
        """None when Connect Gmail isn't configured on this server (the endpoints then answer 503)."""
        names = ("GMAIL_CLIENT_ID", "GMAIL_CLIENT_SECRET", "GMAIL_STATE_SECRET", "GMAIL_REDIRECT_URIS", "GMAIL_KMS_KEY")
        values = [os.environ.get(n, "").strip() for n in names]
        if not all(values):
            return None
        client_id, client_secret, state_secret, uris, key = values
        if not CLIENT_ID.match(client_id):
            raise SystemExit("GMAIL_CLIENT_ID isn't a Google OAuth client id (…apps.googleusercontent.com).")
        return cls(client_id, client_secret, state_secret.encode(),
                   tuple(u.strip() for u in uris.split(",") if u.strip()), key)


# ---------- the signed state ----------

def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def make_state(cfg: ConnectConfig, user_id: str, company_id: str, redirect_uri: str,
               now: Callable[[], float] = time.time) -> str:
    body = _b64(json.dumps({"u": user_id, "c": company_id, "r": redirect_uri, "n": secrets.token_urlsafe(12),
                            "e": int(now()) + STATE_TTL_S}, separators=(",", ":")).encode())
    sig = _b64(hmac.new(cfg.state_secret, body.encode(), hashlib.sha256).digest())
    return f"{body}.{sig}"


def read_state(cfg: ConnectConfig, state: str, user_id: str, company_id: str,
               now: Callable[[], float] = time.time) -> str:
    """The redirect URI the flow started with, if the state is ours, unexpired and for THIS user + company."""
    body, _, sig = (state or "").partition(".")
    want = _b64(hmac.new(cfg.state_secret, body.encode(), hashlib.sha256).digest())
    if not body or not hmac.compare_digest(sig, want):
        raise ConnectError(400, "This Gmail link isn't valid. Start again from Connect Gmail.")
    data = json.loads(_unb64(body))
    if data["e"] < now():
        raise ConnectError(400, "This Gmail link expired. Start again from Connect Gmail.")
    if data["u"] != user_id or data["c"] != company_id:
        raise ConnectError(403, "This Gmail link was started by a different account. Start again from Connect Gmail.")
    return data["r"]


def consent_url(cfg: ConnectConfig, redirect_uri: str, state: str) -> str:
    return AUTH_URL + "?" + urlencode({
        "client_id": cfg.client_id, "redirect_uri": redirect_uri, "response_type": "code", "scope": GMAIL_SCOPE,
        "access_type": "offline",        # we need a refresh token for imports and updates
        "prompt": "consent",             # always show consent, so Google always returns a refresh token
        "include_granted_scopes": "false", "state": state})


# ---------- Google + KMS calls (thin; faked in tests) ----------

class Http(Protocol):
    def post(self, url: str, **kw: Any) -> Any: ...
    def get(self, url: str, **kw: Any) -> Any: ...


class GoogleOAuth:
    def __init__(self, cfg: ConnectConfig, http: Http):
        self.cfg, self.http = cfg, http

    def exchange(self, code: str, redirect_uri: str) -> dict:
        resp = self.http.post(TOKEN_URL, data={"code": code, "client_id": self.cfg.client_id,
                                               "client_secret": self.cfg.client_secret,
                                               "redirect_uri": redirect_uri, "grant_type": "authorization_code"},
                              timeout=30)
        if resp.status_code != 200:
            log.warning(json.dumps({"event": "gmail_code_exchange_failed", "http": resp.status_code,
                                    "google_error": _error_code(resp)}))       # e.g. invalid_client, never a token
            raise ConnectError(400, "Google didn't accept the sign-in code (it may have been used already). "
                                    "Start again from Connect Gmail.")
        return resp.json()

    def profile(self, access_token: str) -> dict:
        resp = self.http.get(PROFILE_URL, headers={"Authorization": f"Bearer {access_token}"}, timeout=30)
        if resp.status_code != 200:
            raise ConnectError(502, "Couldn't read the Gmail account. Try again.")
        return resp.json()

    def revoke(self, token: str) -> None:
        try:
            self.http.post(REVOKE_URL, data={"token": token}, timeout=10)
        except Exception:
            pass                           # best effort: we're already refusing this connection


def _error_code(resp) -> str:
    try:
        return str(resp.json().get("error", ""))[:60]
    except Exception:
        return ""


class Kms:
    """Encrypt-only client for one key. AAD = the company id: the ciphertext only decrypts for that company."""
    def __init__(self, key: str, http: Http):
        self.key, self.http = key, http

    def encrypt(self, plaintext: bytes, aad: bytes) -> bytes:
        resp = self.http.post(KMS_URL.format(key=self.key), json={
            "plaintext": base64.b64encode(plaintext).decode(),
            "additionalAuthenticatedData": base64.b64encode(aad).decode()}, timeout=30)
        if resp.status_code != 200:
            raise ConnectError(503, "Couldn't secure the Gmail connection right now. Try again in a minute.")
        return base64.b64decode(resp.json()["ciphertext"])


# ---------- database ----------

class MailboxTaken(Exception):
    pass


class NotOwner(Exception):
    pass


def save_connection_in_db(conn, user_id: str, company_id: str, address: str, ciphertext: bytes, kms_key: str,
                          history_id: str | None) -> None:
    import psycopg

    try:
        with conn.transaction():
            conn.execute("select set_config('app.user_id', %s, true)", (user_id,))
            conn.execute("select set_config('app.company_id', %s, true)", (company_id,))
            conn.execute("select save_gmail_connection(%s, %s, %s, %s)", (address, ciphertext, kms_key, history_id))
    except psycopg.errors.UniqueViolation:
        raise MailboxTaken from None
    except psycopg.errors.InsufficientPrivilege:
        raise NotOwner from None


def connection_status(conn, company_id: str) -> dict | None:
    with conn.transaction():
        conn.execute("select set_config('app.company_id', %s, true)", (company_id,))
        row = conn.execute("""select gmail_address, status, error, connected_at, last_import_at
                                from gmail_connections where company_id = %s""", (company_id,)).fetchone()
    if row is None:
        return None
    address, status, error, connected_at, last_import_at = row
    return {"gmail_address": address, "status": status, "error": error,
            "connected_at": connected_at.isoformat() if connected_at else None,
            "last_import_at": last_import_at.isoformat() if last_import_at else None}


# ---------- the whole flow ----------

class GmailConnector:
    def __init__(self, cfg: ConnectConfig, oauth: GoogleOAuth, kms: Kms,
                 save: Callable[..., None], status: Callable[[str], dict | None],
                 enqueue_import: Callable[[str], None]):
        self.cfg, self.oauth, self.kms = cfg, oauth, kms
        self._save, self._status, self._enqueue = save, status, enqueue_import

    def start(self, user_id: str, company_id: str, redirect_uri: str) -> str:
        if redirect_uri not in self.cfg.redirect_uris:
            raise ConnectError(400, "This page isn't allowed to connect Gmail.")
        return consent_url(self.cfg, redirect_uri, make_state(self.cfg, user_id, company_id, redirect_uri))

    def finish(self, user_id: str, company_id: str, code: str, state: str) -> str:
        redirect_uri = read_state(self.cfg, state, user_id, company_id)
        tokens = self.oauth.exchange(code, redirect_uri)
        refresh, access = tokens.get("refresh_token"), tokens.get("access_token", "")
        granted = set(tokens.get("scope", "").split())
        # Refusals below DROP the new token but never revoke it. Google revokes the whole grant (this Google
        # account + our client), which also kills a live connection of the same mailbox: Nov 19, a refused
        # Connect on the wrong company revoked the demo mailbox's working token (Bug Journal #26).
        if GMAIL_SCOPE not in granted:
            raise ConnectError(400, "Gmail access wasn't granted. Tick the Gmail box on Google's screen and try again.")
        if not refresh:
            raise ConnectError(400, "Google didn't send a long-lived token. Start again from Connect Gmail.")
        profile = self.oauth.profile(access)
        address = str(profile.get("emailAddress", "")).lower()
        ciphertext = self.kms.encrypt(refresh.encode(), company_id.encode())
        try:
            self._save(user_id, company_id, address, ciphertext, self.cfg.kms_key, profile.get("historyId"))
        except MailboxTaken:
            raise ConnectError(409, "This Gmail account is already connected to another company.") from None
        except NotOwner:
            raise ConnectError(403, "Only the company owner can connect Gmail.") from None
        self._enqueue(company_id)
        return address

    def status(self, company_id: str) -> dict | None:
        return self._status(company_id)

    def sync(self, company_id: str) -> None:
        self._enqueue(company_id)


def db_gmail_connector(cfg: ConnectConfig, pool, enqueue_import: Callable[[str], None]) -> GmailConnector:
    """Production wiring: requests for Google, google-auth (the API's own service account) for KMS."""
    import google.auth
    import requests
    from google.auth.transport.requests import AuthorizedSession

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloudkms"])

    def save(*args) -> None:
        with pool.connection() as conn:
            save_connection_in_db(conn, *args)

    def status(company_id: str) -> dict | None:
        with pool.connection() as conn:
            return connection_status(conn, company_id)

    return GmailConnector(cfg, GoogleOAuth(cfg, requests.Session()), Kms(cfg.kms_key, AuthorizedSession(creds)),
                          save, status, enqueue_import)
