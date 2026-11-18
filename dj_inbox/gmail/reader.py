# dj_inbox/gmail/reader.py
"""Read a mailbox through the Gmail API with the gmail.readonly scope: list message ids, fetch raw emails.

`session` is anything with .get(url, params=...) returning a response with .status_code and .json():
google-auth's AuthorizedSession in real use (it adds and refreshes the OAuth token), a fake in tests.

Which mail is imported is decided by Gmail's own labels (decision Nov 12): no spam, trash, promotions or
social. Everything that comes back is untrusted data (it's whatever anyone emailed the company).
"""
from __future__ import annotations

import base64
import random
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from email.message import EmailMessage

from dj_inbox.ingest import parse_raw

MESSAGES_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages"
IMPORT_QUERY = "-in:spam -in:trash -category:promotions -category:social"
RETRYABLE = {429, 500, 502, 503, 504}
# Gmail reports usage limits as HTTP 403 (not only 429) with one of these reasons: wait and retry those.
# Any other 403 (e.g. insufficientPermissions) is a real error.
RATE_LIMIT_REASONS = {"rateLimitExceeded", "userRateLimitExceeded"}
WORKERS = 2                      # requests in flight at once
MIN_INTERVAL_S = 0.1             # at most ~10 requests/s overall (messages.get = 5 quota units -> ~50 units/s).
                                 # Nov 17: 4 unpaced workers hit rateLimitExceeded on a 942-message import.
MAX_WAIT_S = 64


class GmailError(RuntimeError):
    """A Gmail API call failed for good (not a rate limit or a hiccup). The message never includes the token."""


def _reasons(resp) -> set[str]:
    """Google's error reasons from the response body, e.g. {"userRateLimitExceeded"}; empty if none."""
    try:
        errors = resp.json().get("error", {}).get("errors", [])
        return {str(e.get("reason")) for e in errors if e.get("reason")}
    except Exception:
        return set()


def _retry_after(resp) -> float | None:
    """Seconds from a Retry-After header, if Gmail sent one (capped)."""
    headers = getattr(resp, "headers", None) or {}
    try:
        return min(float(headers.get("Retry-After")), MAX_WAIT_S)
    except (TypeError, ValueError):
        return None


class GmailReader:
    def __init__(self, session, sleep: Callable[[float], None] = time.sleep, tries: int = 8,
                 min_interval: float = MIN_INTERVAL_S, clock: Callable[[], float] = time.monotonic,
                 progress: Callable[[str], None] | None = None):
        self.session = session
        self.sleep = sleep
        self.tries = tries
        self.min_interval = min_interval
        self.clock = clock
        self.progress = progress
        self.calls = 0
        self.rate_limited = 0                       # how often Gmail said "slow down" (for the log line)
        self._lock = threading.Lock()
        self._next_at = 0.0

    def _pace(self) -> None:
        """Space requests at least min_interval apart, across all worker threads."""
        with self._lock:
            now = self.clock()
            wait = self._next_at - now
            self._next_at = max(now, self._next_at) + self.min_interval
        if wait > 0:
            self.sleep(wait)

    def _get(self, url: str, params: dict) -> dict:
        for attempt in range(1, self.tries + 1):
            self._pace()
            self.calls += 1
            resp = self.session.get(url, params=params)
            if resp.status_code == 200:
                return resp.json()
            reasons = _reasons(resp)
            retry = resp.status_code in RETRYABLE or (resp.status_code == 403 and reasons & RATE_LIMIT_REASONS)
            if not retry or attempt == self.tries:
                why = f" ({', '.join(sorted(reasons))})" if reasons else ""
                raise GmailError(f"Gmail API returned HTTP {resp.status_code}{why} for {url.rsplit('/', 1)[-1]}")
            self.rate_limited += 1
            self.sleep(_retry_after(resp) or min(2 ** attempt + random.uniform(0, 1), MAX_WAIT_S))
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
        out = []
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for n, item in enumerate(pool.map(self.get, ids), 1):
                out.append(item)
                if self.progress and (n % 100 == 0 or n == len(ids)):
                    self.progress(f"   fetched {n}/{len(ids)} (Gmail asked us to slow down {self.rate_limited}x)")
        return out
