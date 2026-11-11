# evals/score.py
"""Grade one answer against one eval question, and measure retrieval.

tests/test_score.py is the spec: every rule here has a test there.
Same idea as the NFL agent's Critic: pull every number out of the text and check it against the truth,
except here the truth is the eval set instead of SQL rows.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel


class Fact(BaseModel):
    kind: Literal["money", "date", "number", "text"]
    value: int | str          # money/number: int dollars or count; date: "2026-05-23"; text: venue name or status


class Evidence(BaseModel):
    thread_id: str
    quote: str


class Question(BaseModel):
    id: str
    type: str
    question: str
    expected_answer: str
    expected_facts: list[Fact]
    gold_evidence: list[Evidence]
    should_defer: bool
    must_not_contain: list[str]


class Score(BaseModel):
    question_id: str
    facts_expected: int
    facts_found: int
    deferred: bool             # did the answer say it couldn't find this?
    forbidden_hits: list[str]  # must_not_contain phrases that appeared
    correct: bool


# Words that mean the answer is a "not found". Lowercase; match anywhere in the answer.
DEFER_PHRASES = ["not found", "couldn't find", "could not find", "no record", "don't see", "do not see",
                 "nothing in your emails", "no emails",
                 # v2 (Nov 11): honest answers to "what's their first-dance song?" usually say the emails don't
                 # mention it; "none of ... mention" answers "which jobs mention a flower wall?"
                 "don't mention", "doesn't mention", "do not mention", "does not mention", "not mentioned",
                 "no mention", "don't say", "doesn't say", "does not say", "do not say", "not specified",
                 "none of"]

# How a status fact can be stated in an answer. Careful: "not booked" contains "booked".
STATUS_WORDS = {
    "booked": ["booked", "paid the deposit", "deposit paid"],
    "lost": ["declined", "passed", "went with", "not going ahead", "decided not", "lost", "postpone",
             "did not book", "didn't book"],
    "cold": ["cold", "no reply", "hasn't replied", "went quiet"],
    "awaiting_reply": ["waiting on us", "owe them a reply", "awaiting our reply", "haven't replied"],
    "waiting_on_client": ["waiting on them", "waiting to hear back", "waiting on the client"],
    # For "did they book?" when the answer is no, but they didn't decline either (still open or cold).
    "not_booked": ["did not book", "didn't book", "not booked", "hasn't booked", "has not booked",
                   "haven't booked", "no deposit", "not yet booked"],
}


MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def find_money(text: str) -> set[int]:
    """Dollar amounts as whole dollars: "$2,395" -> 2395, "$2395" -> 2395. Ignore numbers without a $."""
    # \$ = a literal dollar sign; (\d[\d,]*) = a digit, then more digits/commas, captured as a group.
    # The (?![\d,]*[MmKk]) part skips shorthand like "$2M insurance", which isn't a price.
    return {int(m.replace(",", "")) for m in re.findall(r"\$(\d[\d,]*)(?![\d,]*[MmKk])", text)}


def find_dates(text: str) -> set[date]:
    """Dates in the formats the inbox uses: "2026-05-23", "Saturday, May 23, 2026", "May 23rd, 2026",
    "May 23 2026", "Dec 31, 2026", "5/23/26", "5/23/2026". Dates without a year can be ignored."""
    found: set[date] = set()
    # 1. ISO: 2026-05-23
    for y, m, d in re.findall(r"\b(\d{4})-(\d{2})-(\d{2})\b", text):
        found.add(date(int(y), int(m), int(d)))
    # 2. Month name: "May 23, 2026", "May 23rd 2026", "December 31, 2026". The weekday before it doesn't matter.
    #    ([A-Za-z]{3})[a-z]* keeps the first 3 letters ("Dec") so "Dec" and "December" both work.
    for mon, d, y in re.findall(r"\b([A-Za-z]{3})[a-z]*\.? (\d{1,2})(?:st|nd|rd|th)?,? (\d{4})\b", text):
        if mon.lower() in MONTHS:                     # skips things like "Sat 23, 2026" that aren't months
            found.add(date(int(y), MONTHS[mon.lower()], int(d)))
    # 3. US numeric: 5/23/26 or 5/23/2026 (two-digit years are 20xx)
    for m, d, y in re.findall(r"\b(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})\b", text):
        found.add(date(int(y) + (2000 if len(y) == 2 else 0), int(m), int(d)))
    return found


def find_numbers(text: str) -> set[int]:
    """Plain whole numbers: "about 115 guests" -> {115}. "1,000" -> 1000."""
    # Try "1,000"-style numbers first so the comma version wins over its pieces; then plain digit runs.
    return {int(n.replace(",", "")) for n in re.findall(r"\b\d{1,3}(?:,\d{3})+\b|\b\d+\b", text)}


def _states_status(low: str, status: str) -> bool:
    """True if the answer says this status, ignoring negated mentions like "not booked" or "isn't booked"."""
    for word in STATUS_WORDS[status]:
        for match in re.finditer(re.escape(word), low):
            before = low[max(0, match.start() - 8):match.start()]   # the few characters just before the word
            if not re.search(r"\bnot\s|n't\s", before):
                return True
    return False


def score_answer(answer: str, q: Question) -> Score:
    """Correct means ALL of:
    - should_defer questions: the answer defers (DEFER_PHRASES).  Others: it does NOT defer.
    - every expected fact is found (money/date/number via the finders; text: venue name case-insensitive,
      or a status via STATUS_WORDS, where a negated "not booked" must NOT count as booked)
    - no must_not_contain phrase appears (case-insensitive)
    """
    low = answer.lower()
    deferred = any(phrase in low for phrase in DEFER_PHRASES)

    found = 0
    for fact in q.expected_facts:
        if fact.kind == "money":
            found += fact.value in find_money(answer)
        elif fact.kind == "date":
            found += date.fromisoformat(str(fact.value)) in find_dates(answer)
        elif fact.kind == "number":
            found += fact.value in find_numbers(answer)
        elif fact.value in STATUS_WORDS:              # a status like "booked" or "lost"
            found += _states_status(low, str(fact.value))
        else:                                         # a venue name
            found += str(fact.value).lower() in low

    forbidden = [phrase for phrase in q.must_not_contain if phrase.lower() in low]
    correct = deferred == q.should_defer and found == len(q.expected_facts) and not forbidden
    return Score(question_id=q.id, facts_expected=len(q.expected_facts), facts_found=found,
                 deferred=deferred, forbidden_hits=forbidden, correct=correct)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())           # lowercase, and any run of spaces/newlines -> one space


def recall_at_k(retrieved: list[tuple[str, str]], gold: list[Evidence], k: int = 5) -> float:
    """retrieved = [(thread_id, chunk_text), ...] best first. Return the fraction of gold evidence found in the
    top k: a gold item is found if a chunk FROM THE SAME THREAD contains its quote, comparing with case and
    whitespace normalized. No gold evidence (count / not-in-inbox questions) -> return 1.0."""
    if not gold:
        return 1.0
    top = [(thread_id, _normalize(chunk)) for thread_id, chunk in retrieved[:k]]
    hits = sum(
        any(thread_id == g.thread_id and _normalize(g.quote) in chunk for thread_id, chunk in top)
        for g in gold
    )
    return hits / len(gold)
