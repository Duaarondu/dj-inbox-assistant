# dj_inbox/worker/importer.py
"""One company's Gmail import, as the worker runs it.

    gmail_connections row (as app_ingest) -> KMS decrypt the refresh token (worker = decrypt-only)
    -> Gmail session -> list (filtered) -> skip already stored -> fetch -> import_messages (proved Nov 17)
    -> status 'ready' + last_import_at + history_id (where automatic updates will continue from)

Only this service can decrypt tokens (IAM), and only for the company in the request: KMS checks the company
id we pass as additional authenticated data, so a ciphertext copied into another company's row won't decrypt.
Errors are stored as short messages for the owner; tokens never appear in logs, errors or responses.
"""
from __future__ import annotations

import base64
import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

from dj_inbox.db.inbox import company_tx, stored_gmail_ids
from dj_inbox.gmail.import_inbox import import_messages
from dj_inbox.gmail.reader import IMPORT_QUERY, GmailError, GmailReader
from dj_inbox.ingest import CompanyConfig

log = logging.getLogger("dj_inbox.worker")
KMS_DECRYPT_URL = "https://cloudkms.googleapis.com/v1/{key}:decrypt"
TOKEN_URL = "https://oauth2.googleapis.com/token"
PROFILE_URL = "https://gmail.googleapis.com/gmail/v1/users/me/profile"


class RetryLater(Exception):
    """A temporary problem (Gmail rate limits, KMS hiccup): the task queue should try again."""


class GmailAccessLost(Exception):
    """The refresh token no longer works (revoked, or 7-day Testing-mode expiry): the owner must reconnect."""


@dataclass
class Connection:
    company_id: str
    gmail_address: str
    ciphertext: bytes
    kms_key: str
    domains: list[str]


def load_connection(conn, company_id: str) -> Connection | None:
    with company_tx(conn, company_id):
        row = conn.execute("""select g.gmail_address, g.token_ciphertext, g.kms_key, c.email_domains
                                from gmail_connections g join companies c using (company_id)
                               where g.company_id = %s and g.status <> 'disconnected'""", (company_id,)).fetchone()
    if row is None:
        return None
    return Connection(company_id, row[0], bytes(row[1]), row[2], list(row[3] or []))


def set_status(conn, company_id: str, status: str, error: str | None = None, history_id: str | None = None) -> None:
    with company_tx(conn, company_id):
        conn.execute("""update gmail_connections
                           set status = %s, error = %s,
                               history_id = coalesce(%s, history_id),
                               last_import_at = case when %s = 'ready' then now() else last_import_at end
                         where company_id = %s""", (status, error, history_id, status, company_id))


class KmsDecrypt:
    def __init__(self, http):
        self.http = http

    def __call__(self, key: str, ciphertext: bytes, aad: bytes) -> bytes:
        resp = self.http.post(KMS_DECRYPT_URL.format(key=key), json={
            "ciphertext": base64.b64encode(ciphertext).decode(),
            "additionalAuthenticatedData": base64.b64encode(aad).decode()}, timeout=30)
        if resp.status_code in (429, 500, 502, 503, 504):
            raise RetryLater("KMS unavailable")
        if resp.status_code != 200:
            raise RuntimeError(f"KMS decrypt refused (HTTP {resp.status_code})")      # e.g. wrong company AAD
        return base64.b64decode(resp.json()["plaintext"])


class GmailSession:
    """A requests-style session that adds a fresh access token (refreshed from the refresh token) to each call."""
    def __init__(self, http, refresh_token: str, client_id: str, client_secret: str,
                 clock: Callable[[], float] = time.time):
        self.http, self._refresh = http, refresh_token
        self._client = (client_id, client_secret)
        self.clock = clock
        self._token, self._expires = "", 0.0

    def _access_token(self) -> str:
        if self.clock() < self._expires - 60:
            return self._token
        resp = self.http.post(TOKEN_URL, data={"grant_type": "refresh_token", "refresh_token": self._refresh,
                                               "client_id": self._client[0], "client_secret": self._client[1]},
                              timeout=30)
        if resp.status_code in (400, 401):
            raise GmailAccessLost(_google_error(resp))
        if resp.status_code != 200:
            raise RetryLater(f"token refresh HTTP {resp.status_code}")
        body = resp.json()
        self._token, self._expires = body["access_token"], self.clock() + float(body.get("expires_in", 3600))
        return self._token

    def get(self, url: str, params: dict | None = None, **kw):
        headers = {**kw.pop("headers", {}), "Authorization": f"Bearer {self._access_token()}"}
        return self.http.get(url, params=params, headers=headers, timeout=60, **kw)


def _google_error(resp) -> str:
    """Google's error CODE only (e.g. invalid_grant), safe to store and log."""
    try:
        return str(resp.json().get("error", f"HTTP {resp.status_code}"))[:60]
    except Exception:
        return f"HTTP {resp.status_code}"


@dataclass
class ImportResult:
    company_id: str
    status: str
    messages_new: int = 0
    threads_total: int = 0
    changed_threads: int = 0
    leads: int = 0
    seconds: float = 0.0
    error: str | None = None


def run_import(pool, company_id: str, *, decrypt, make_session: Callable[[str], object], embed, extract,
               model: str) -> ImportResult:
    """Import everything not stored yet for one company. Raises RetryLater for temporary problems."""
    start = time.perf_counter()
    with pool.connection() as conn:
        found = load_connection(conn, company_id)
    if found is None:
        return ImportResult(company_id, "skipped", error="no Gmail connection")
    try:
        refresh = decrypt(found.kms_key, found.ciphertext, company_id.encode()).decode()
        session = make_session(refresh)
        start_history = str(session.get(PROFILE_URL).json().get("historyId") or "") or None
        reader = GmailReader(session)
        with pool.connection() as conn:
            have = stored_gmail_ids(conn, company_id)
            ids = [i for i in reader.list_ids(IMPORT_QUERY) if i not in have]
            items = reader.fetch(ids)
            company = CompanyConfig(domains=found.domains, addresses=[found.gmail_address])
            stats = import_messages(conn, company_id, items, company, embed, extract, model)
            set_status(conn, company_id, "ready", history_id=start_history)
    except GmailAccessLost as e:
        _fail(pool, company_id, "Gmail access ended. Reconnect Gmail to keep importing.")
        return ImportResult(company_id, "error", seconds=time.perf_counter() - start, error=str(e))
    except (RetryLater, GmailError) as e:
        _fail(pool, company_id, "Gmail is busy; retrying automatically.", status="importing")
        raise RetryLater(str(e)) from None
    except Exception as e:
        _fail(pool, company_id, "Import failed. Try Reconnect Gmail; if it keeps failing, tell support.")
        log.exception(json.dumps({"event": "import_failed", "company_id": company_id, "type": type(e).__name__}))
        return ImportResult(company_id, "error", seconds=time.perf_counter() - start, error=type(e).__name__)
    result = ImportResult(company_id, "ready", len(items), stats.threads_total, stats.changed_threads, stats.leads,
                          round(time.perf_counter() - start, 1))
    log.info(json.dumps({"event": "import_done", **asdict(result)}))
    return result


def _fail(pool, company_id: str, message: str, status: str = "error") -> None:
    try:
        with pool.connection() as conn:
            set_status(conn, company_id, status, error=message)
    except Exception:
        log.exception(json.dumps({"event": "status_update_failed", "company_id": company_id}))
