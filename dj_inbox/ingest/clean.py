# dj_inbox/ingest/clean.py
"""Turn a raw email body into what retrieval and extraction should see.

Order matters:
  1. strip_quoted    - drop the earlier messages each reply repeats below itself
  2. find_contacts   - collect emails/phones (signatures hold the client's phone, so do this BEFORE step 3)
  3. strip_signature - drop sign-offs and signature blocks
  4. redact          - replace emails/phones with [EMAIL] / [PHONE]; only redacted text goes to models
"""
from __future__ import annotations

import re

from .models import Contacts

# "On Tue, Oct 14, 2025 at 8:34 AM Mike Russo <mike@x.com> wrote:"  (Gmail, Apple Mail)
QUOTE_HEADER = re.compile(r"^On\s.{5,250}\swrote:\s*$")
OUTLOOK_HEADER = re.compile(r"^-{2,}\s*Original Message\s*-{2,}\s*$", re.I)
SENT_FROM = re.compile(r"^Sent from my [\w ]+$", re.I)
CLOSING = re.compile(
    r"^(thanks|thank you|thanks so much|best|best regards|regards|kind regards|cheers|sincerely|warmly|"
    r"talk soon|xo+)[\s!,.]*$", re.I)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# 516-555-0165, (516) 555-0165, 516.555.0165, +1 516 555 0165. Digit guards keep prices and dates safe.
PHONE_RE = re.compile(r"(?<![\d$])(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]\d{4}(?![\d])")
DOMAIN_RE = re.compile(r"\b(?:https?://)?[\w-]+(?:\.[\w-]+)*\.(?:com|net|org|co|io|us|biz|example)\b", re.I)


def strip_quoted(body: str) -> str:
    """Cut everything from the first reply header; drop '>' lines. Forwarded messages are kept."""
    lines = body.splitlines()
    out: list[str] = []
    for i, line in enumerate(lines):
        s = line.strip()
        if QUOTE_HEADER.match(s) or OUTLOOK_HEADER.match(s):
            break
        # Gmail sometimes wraps the header over two lines: "On Tue ... Jenna <" / "jenna@x.com> wrote:"
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if s.startswith("On ") and nxt.endswith("wrote:") and len(s) + len(nxt) < 300:
            break
        if s.startswith(">"):
            continue
        out.append(line)
    return "\n".join(out).rstrip()


def _is_signature_block(block: list[str]) -> bool:
    lines = [line.strip() for line in block if line.strip()]
    if not lines:
        return False
    if all(SENT_FROM.match(line) for line in lines):
        return True
    if len(lines) <= 3 and CLOSING.match(lines[0]):                      # "Thanks!\nJenna"
        return True
    has_contact = any(PHONE_RE.search(line) or EMAIL_RE.search(line) or DOMAIN_RE.search(line) for line in lines)
    short = all(len(line) <= 45 for line in lines)
    no_sentences = not any(line.endswith(("?", ".")) and len(line.split()) > 4 for line in lines)
    return len(lines) <= 5 and has_contact and short and no_sentences  # "Mike Russo\nCo.\n(516)...\nsite"


def strip_signature(body: str) -> str:
    """Remove up to two trailing signature-like paragraphs. Never returns an empty body."""
    paragraphs = re.split(r"\n\s*\n", body.strip())
    for _ in range(2):
        if len(paragraphs) > 1 and _is_signature_block(paragraphs[-1].splitlines()):
            paragraphs.pop()
        else:
            break
    # A message that is ONLY a sign-off ("Sent from my iPhone") keeps its text.
    return "\n\n".join(paragraphs).strip() or body.strip()


def find_contacts(text: str) -> Contacts:
    emails = list(dict.fromkeys(e.lower() for e in EMAIL_RE.findall(text)))
    phones = list(dict.fromkeys(_digits(p) for p in PHONE_RE.findall(text)))
    return Contacts(emails=emails, phones=phones)


def redact(text: str) -> str:
    return PHONE_RE.sub("[PHONE]", EMAIL_RE.sub("[EMAIL]", text))


def clean_body(raw: str) -> tuple[str, Contacts]:
    unquoted = strip_quoted(raw)
    contacts = find_contacts(unquoted)
    clean = redact(strip_signature(unquoted))
    return re.sub(r"\n{3,}", "\n\n", clean).strip(), contacts


def _digits(phone: str) -> str:
    """Store phones in one format so the same number always matches: 5165550165."""
    d = re.sub(r"\D", "", phone)
    return d[1:] if len(d) == 11 and d.startswith("1") else d
