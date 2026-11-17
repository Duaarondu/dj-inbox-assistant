# dj_inbox/gmail/reader.py
"""Read a mailbox through the Gmail API with the gmail.readonly scope: list message ids, fetch raw emails.

`session` is anything with .get(url, params=...) returning a response with .status_code and .json():
google-auth's AuthorizedSession in real use (it adds and refreshes the OAuth token), a fake in tests.

Which mail is imported is decided by Gmail's own labels (decision Nov 12): no spam, trash, promotions or
social. Everything that comes back is untrusted data (it's whatever anyone emailed the company).
"""
from __future__ import annotations

import base64
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from email.message import EmailMessage

from dj_inbox.ingest import parse_raw

MESSAGES_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages"
IMPORT_QUERY = "-in:spam -in:trash -category:promotions -category:social"
RETRYABLE = {429, 500, 502, 503, 504}
WORKERS = 4                      # requests in flight; messages.get costs 5 quota units, the per-user limit is 250/s


class GmailError(RuntimeError):
    """A Gmail API call failed for good (not a rate limit or a hiccup). The message never includes the token."""


class GmailReader:
    def __init__(self, session, sleep: Callable[[float], None] = time.sleep, tries: int = 6):
        self.session = session
        self.sleep = sleep
        self.tries = tries
        self.calls = 0

    def _get(self, url: str, params: dict) -> dict:
        for attempt in range(1, self.tries + 1):
            self.calls += 1
            resp = self.session.get(url, params=params)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code not in RETRYABLE or attempt == self.tries:
                raise GmailError(f"Gmail API returned HTTP {resp.status_code} for {url.rsplit('/', 1)[-1]}")
            self.sleep(min(2 ** attempt, 30))
        raise AssertionError("unreachable")

    def list_ids(self, query: str = IMPORT_QUERY, limit: int | None = None) -> list[str]:
        """Ids of every message matching `query` (newest first, as Gmail returns them), following page tokens."""
        ids: list[str] = []
        params: dict = {"q": query, "maxResults": 500}
        while True:
            page = self._get(MESSAGES_URL, params)
            ids += [m["id"] for m in page.get("messages", [])]
            if limit is not None and len(ids) >= limit:
                return ids[:limit]
            token = page.get("nextPageToken")
            if not token:
                return ids
            params = {**params, "pageToken": token}

    def get(self, gmail_id: str) -> tuple[EmailMessage, str, str]:
        """(parsed email, Gmail thread id, Gmail message id) for one message."""
        data = self._get(f"{MESSAGES_URL}/{gmail_id}", {"format": "raw"})
        raw = base64.urlsafe_b64decode(data["raw"] + "=" * (-len(data["raw"]) % 4))
        return parse_raw(raw), data["threadId"], data["id"]

    def fetch(self, ids: list[str]) -> list[tuple[EmailMessage, str, str]]:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            return list(pool.map(self.get, ids))
