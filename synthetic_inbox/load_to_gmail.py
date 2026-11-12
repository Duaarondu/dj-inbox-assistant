"""Load the synthetic Northshore inbox into the demo Gmail account (one-time dev tool).

Every email in data/inbox.mbox goes in through Gmail's messages.import, oldest first. Gmail
rebuilds the same 200 threads from the Message-ID / In-Reply-To / References headers. Import
only places messages in the mailbox: nothing is sent to anyone (all addresses are .example anyway).

Safety:
- Refuses to import unless the signed-in Google account is exactly --account, so 939 fake emails
  can't land in your personal Gmail.
- Scopes: gmail.insert (add messages) + gmail.readonly (check a message isn't already there, read its
  thread id) + openid/email (check which account signed in). None of them can send or delete mail.
- Idempotent: before each import it searches Gmail for the Message-ID, so a rerun never duplicates,
  even if the progress file is lost or a run crashed mid-way.
- The OAuth token stays in memory and is never written to disk. The only file written is a
  progress file (Message-ID -> Gmail ids), so a rerun skips what is already imported.

Setup (once):  pip install google-auth-oauthlib
Usage:
    python synthetic_inbox/load_to_gmail.py --dry-run
    python synthetic_inbox/load_to_gmail.py --client-secrets ~/Downloads/loader-client.json \
        --account northshore.sound.demo@gmail.com --limit 5
    (then the same command without --limit)
"""
from __future__ import annotations

import argparse
import base64
import json
import mailbox
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from email.utils import parseaddr, parsedate_to_datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
MBOX = HERE / "data" / "inbox.mbox"
STATE = HERE / ".gmail_import_state.json"
COMPANY_DOMAIN = "northshoresound.example"

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/gmail.insert",
    "https://www.googleapis.com/auth/gmail.readonly",
]
MESSAGES_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages"
IMPORT_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages/import"
IMPORT_PARAMS = {"internalDateSource": "dateHeader", "neverMarkSpam": "true", "processForCalendar": "false"}
USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
RETRYABLE = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class PlannedMessage:
    message_id: str
    thread_key: str  # X-GM-THRID in the mbox (t185 ...): our thread id, not Gmail's
    in_reply_to: str | None
    sent_at: datetime
    from_company: bool
    raw: bytes

    @property
    def label_ids(self) -> list[str]:
        # Our own replies look like sent mail; client emails land in the inbox.
        return ["SENT"] if self.from_company else ["INBOX"]


def plan(mbox_path: Path = MBOX, company_domain: str = COMPANY_DOMAIN) -> list[PlannedMessage]:
    """Read the mbox and return messages oldest first, so a reply never goes in before its parent."""
    rows = []
    for index, msg in enumerate(mailbox.mbox(str(mbox_path))):
        message_id, thread_key, date = msg["Message-ID"], msg["X-GM-THRID"], msg["Date"]
        if not (message_id and thread_key and date):
            raise ValueError(f"message #{index} is missing Message-ID, X-GM-THRID or Date")
        sender = parseaddr(msg["From"] or "")[1].lower()
        rows.append((
            parsedate_to_datetime(date),
            index,
            PlannedMessage(
                message_id=message_id.strip(),
                thread_key=thread_key.strip(),
                in_reply_to=(msg["In-Reply-To"] or "").strip() or None,
                sent_at=parsedate_to_datetime(date),
                from_company=sender.endswith("@" + company_domain),
                raw=msg.as_bytes(),
            ),
        ))
    rows.sort(key=lambda r: (r[0], r[1]))  # mbox order breaks ties
    planned = [r[2] for r in rows]
    dupes = [m for m, n in Counter(p.message_id for p in planned).items() if n > 1]
    if dupes:
        raise ValueError(f"duplicate Message-IDs: {dupes[:3]}")
    return planned


def load_state(path: Path = STATE) -> dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_state(state: dict[str, dict], path: Path = STATE) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(path)  # atomic: a crash mid-write never leaves a half-written progress file


def _post(session, body: dict, *, tries: int = 5, sleep=time.sleep):
    """POST one import; back off on 429/5xx. Returns the final response."""
    for attempt in range(tries):
        resp = session.post(IMPORT_URL, params=IMPORT_PARAMS, json=body, timeout=60)
        if resp.status_code not in RETRYABLE:
            return resp
        sleep(min(2 ** attempt, 30))
    return resp


def import_message(session, m: PlannedMessage, gmail_thread_id: str | None, *, sleep=time.sleep) -> tuple[dict, str]:
    """Import one message. Returns (Gmail response json, note). Falls back if Gmail rejects a field."""
    body = {"raw": base64.urlsafe_b64encode(m.raw).decode("ascii"), "labelIds": m.label_ids}
    attempts = []
    if gmail_thread_id:
        attempts.append(("", {**body, "threadId": gmail_thread_id}))
        attempts.append(("dropped threadId", body))
    else:
        attempts.append(("", body))
    if m.from_company:
        attempts.append(("SENT label refused, used INBOX", {**body, "labelIds": ["INBOX"]}))
    last = None
    for note, attempt_body in attempts:
        resp = _post(session, attempt_body, sleep=sleep)
        if resp.status_code == 200:
            return resp.json(), note
        last = resp
        if resp.status_code != 400:
            break  # auth or quota problems won't be fixed by changing the body
    raise RuntimeError(f"import failed for {m.message_id}: HTTP {last.status_code} {last.text[:300]}")


def find_in_gmail(session, message_id: str) -> dict | None:
    """The Gmail message ({id, threadId}) with this Message-ID header, or None."""
    resp = session.get(MESSAGES_URL, params={"q": "rfc822msgid:" + message_id.strip("<>"), "maxResults": 1,
                                             "includeSpamTrash": "true"}, timeout=30)
    resp.raise_for_status()
    found = resp.json().get("messages") or []
    return found[0] if found else None


def thread_of(session, gmail_id: str) -> str:
    resp = session.get(f"{MESSAGES_URL}/{gmail_id}", params={"format": "minimal"}, timeout=30)
    resp.raise_for_status()
    return resp.json()["threadId"]


def import_all(session, planned: list[PlannedMessage], state: dict[str, dict], *, limit: int | None = None,
               pause: float = 0.12, sleep=time.sleep, save=save_state, log=print) -> Counter:
    """Import what isn't in Gmail yet, oldest first. Saves progress after every message."""
    gmail_thread = {v["thread_key"]: v["gmail_thread_id"] for v in state.values()}
    stats: Counter = Counter()
    for m in planned:
        if m.message_id in state:
            stats["skipped (already imported)"] += 1
            continue
        if limit is not None and stats["imported"] >= limit:
            break
        existing = find_in_gmail(session, m.message_id)
        if existing:  # imported by an earlier run that didn't get to save its progress
            gmail_id, thread_id, note = existing["id"], existing["threadId"], "found in Gmail, not re-imported"
        else:
            result, note = import_message(session, m, gmail_thread.get(m.thread_key), sleep=sleep)
            gmail_id = result["id"]
            thread_id = result.get("threadId") or thread_of(session, gmail_id)
            stats["imported"] += 1
        gmail_thread.setdefault(m.thread_key, thread_id)
        state[m.message_id] = {"thread_key": m.thread_key, "gmail_id": gmail_id, "gmail_thread_id": thread_id}
        save(state)
        if note:
            stats[note] += 1
        if stats["imported"] and stats["imported"] % 50 == 0:
            log(f"  {stats['imported']} imported ...")
        sleep(pause)  # well under Gmail's per-user rate limit
    return stats


def thread_check(state: dict[str, dict]) -> tuple[int, int, dict[str, set[str]]]:
    """Our threads vs Gmail threads. Returns (our thread count, Gmail thread count, split threads)."""
    by_key: dict[str, set[str]] = defaultdict(set)
    for v in state.values():
        by_key[v["thread_key"]].add(v["gmail_thread_id"])
    split = {k: ids for k, ids in by_key.items() if len(ids) > 1}
    gmail_threads = {t for ids in by_key.values() for t in ids}
    return len(by_key), len(gmail_threads), split


def signed_in_account(session) -> str:
    resp = session.get(USERINFO_URL, timeout=30)
    resp.raise_for_status()
    return resp.json().get("email", "").lower()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--client-secrets", type=Path, help="Desktop OAuth client JSON (keep it outside the repo)")
    ap.add_argument("--account", help="the demo Gmail address; the import refuses any other account")
    ap.add_argument("--limit", type=int, help="import at most N new messages (try 5 first)")
    ap.add_argument("--dry-run", action="store_true", help="only read the mbox and print the plan")
    args = ap.parse_args(argv)

    planned = plan()
    state = load_state()
    company = sum(m.from_company for m in planned)
    print(f"plan: {len(planned)} messages in {len({m.thread_key for m in planned})} threads "
          f"({company} from the company -> SENT, {len(planned) - company} from clients -> INBOX), "
          f"{planned[0].sent_at:%Y-%m-%d} to {planned[-1].sent_at:%Y-%m-%d}; already imported: {len(state)}")
    if args.dry_run:
        return 0
    if not (args.client_secrets and args.account):
        ap.error("--client-secrets and --account are required unless --dry-run")

    from google.auth.transport.requests import AuthorizedSession  # local dev dependency
    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(str(args.client_secrets.expanduser()), SCOPES)
    creds = flow.run_local_server(port=0, login_hint=args.account, prompt="select_account consent")
    session = AuthorizedSession(creds)

    who = signed_in_account(session)
    if who != args.account.lower():
        print(f"STOP: signed in as {who!r}, expected {args.account!r}. Nothing was imported.", file=sys.stderr)
        return 2
    print(f"signed in as {who}")

    started = time.monotonic()
    stats = import_all(session, planned, state, limit=args.limit)
    ours, gmails, split = thread_check(state)
    print(f"done in {time.monotonic() - started:.0f}s: {dict(stats)}")
    print(f"threads: {ours} of ours -> {gmails} Gmail threads (want equal); total imported {len(state)}/{len(planned)}")
    if split:
        print(f"WARNING: {len(split)} of our threads were split by Gmail: {sorted(split)[:10]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
