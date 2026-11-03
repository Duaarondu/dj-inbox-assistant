# dj_inbox/ingest/read_mbox.py
"""Read a Gmail export (.mbox) into cleaned threads.

Threads are rebuilt in this order of trust:
  1. Gmail's X-GM-THRID header (present in Google Takeout exports)
  2. the In-Reply-To / References chain
  3. same normalized subject + same client address within 30 days
"""
from __future__ import annotations

import email
import hashlib
import mailbox
import re
from datetime import datetime, timedelta, timezone
from email import policy
from email.message import EmailMessage
from email.utils import getaddresses, parseaddr, parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path

from .clean import clean_body
from .models import CompanyConfig, Email, EmailThread

SUBJECT_PREFIX = re.compile(r"^\s*((re|fwd?|fw)\s*:\s*)+", re.I)
SUBJECT_WINDOW = timedelta(days=30)


def _load(f) -> EmailMessage:
    # Exports written on Windows use \r\n line endings, and Python's mailbox module there leaves a blank
    # line after the "From " separator, which makes the parser see an email with no headers.
    data = f.read().replace(b"\r\n", b"\n").lstrip(b"\n")
    return email.message_from_bytes(data, policy=policy.default)


def read_mbox(path: Path, company: CompanyConfig) -> list[EmailThread]:
    box = mailbox.mbox(str(path), factory=_load, create=False)
    raw = [_parse(msg, company) for msg in box]
    raw.sort(key=lambda r: r["sent_at"])

    thread_of: dict[str, str] = {}                        # message_id -> thread key
    by_subject: dict[tuple[str, str], tuple[str, datetime]] = {}
    threads: dict[str, list[Email]] = {}
    for r in raw:
        key = _thread_key(r, thread_of, by_subject)
        thread_of[r["message_id"]] = key
        by_subject[(r["norm_subject"], r["client_address"])] = (key, r["sent_at"])
        threads.setdefault(key, []).append(Email(
            message_id=r["message_id"], thread_id=_thread_id(key), sent_at=r["sent_at"],
            sender_role=r["role"], sender_name=r["sender_name"], sender_email=r["sender_email"],
            subject=r["subject"], body_raw=r["body"], body_clean=r["clean"], contacts=r["contacts"]))

    out = [EmailThread(thread_id=_thread_id(k), subject=msgs[0].subject, messages=msgs)
           for k, msgs in threads.items()]
    out.sort(key=lambda t: t.messages[0].sent_at)
    return out


def _parse(msg: EmailMessage, company: CompanyConfig) -> dict:
    sender_name, sender_email = parseaddr(str(msg.get("From", "")))
    sender_email = sender_email.lower()
    recipients = [a.lower() for _, a in getaddresses([str(msg.get("To", ""))]) if a]
    role = company.role_of(sender_email)
    body = _body_text(msg)
    message_id = str(msg.get("Message-ID", "")).strip() or _hash(sender_email, str(msg.get("Date")), body)
    clean, contacts = clean_body(body)
    subject = str(msg.get("Subject", "")).strip()
    # The "client address" pairs a reply with its thread when headers are missing.
    if company.is_form(sender_email) and contacts.emails:
        client_address = contacts.emails[0]  # the couple's own address from the form fields, not noreply@
    elif role == "client":
        client_address = sender_email
    else:
        client_address = recipients[0] if recipients else ""
    refs = str(msg.get("References", "")).split() + str(msg.get("In-Reply-To", "")).split()
    return {
        "message_id": message_id, "sent_at": _date(msg), "role": role, "sender_name": sender_name or sender_email,
        "sender_email": sender_email, "subject": subject, "body": body, "clean": clean, "contacts": contacts,
        "gmail_thread": str(msg.get("X-GM-THRID", "")).strip(), "refs": refs,
        "norm_subject": SUBJECT_PREFIX.sub("", subject).strip().lower(), "client_address": client_address,
    }


def _thread_key(r: dict, thread_of: dict[str, str], by_subject: dict) -> str:
    if r["gmail_thread"]:
        return f"gm:{r['gmail_thread']}"
    for ref in reversed(r["refs"]):                      # nearest parent first
        if ref in thread_of:
            return thread_of[ref]
    hit = by_subject.get((r["norm_subject"], r["client_address"]))
    if hit and r["norm_subject"] and r["sent_at"] - hit[1] <= SUBJECT_WINDOW:
        return hit[0]
    return f"mid:{r['message_id']}"


def _thread_id(key: str) -> str:
    kind, _, value = key.partition(":")
    return value if kind == "gm" else "x" + hashlib.sha1(value.encode()).hexdigest()[:10]


def _body_text(msg: EmailMessage) -> str:
    part = msg.get_body(preferencelist=("plain", "html"))
    if part is None:
        return ""
    text = part.get_content()
    return _html_to_text(text) if part.get_content_type() == "text/html" else text.replace("\r\n", "\n")


def _date(msg: EmailMessage) -> datetime:
    try:
        d = parsedate_to_datetime(str(msg.get("Date")))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return datetime(1970, 1, 1, tzinfo=timezone.utc)  # undated mail sorts first instead of crashing


class _TextOnly(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in ("br", "p", "div", "li", "tr"):
            self.parts.append("\n")


def _html_to_text(html: str) -> str:
    p = _TextOnly()
    p.feed(html)
    return re.sub(r"\n{3,}", "\n\n", "".join(p.parts)).strip()


def _hash(*parts: str) -> str:
    return "<" + hashlib.sha1("|".join(parts).encode()).hexdigest() + "@generated>"
