# evals/score.py
"""Grade one answer against one eval question, and measure retrieval.

YOU write the five function bodies marked TODO. tests/test_score.py is the spec: when it passes, you're done.
Same idea as the NFL agent's Critic: pull every number out of the text and check it against the truth,
except here the truth is the eval set instead of SQL rows.
"""
from __future__ import annotations

import re  # noqa: F401  (you'll need it)
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
                 "nothing in your emails", "no emails"]

# How a status fact can be stated in an answer. Careful: "not booked" contains "booked".
STATUS_WORDS = {
    "booked": ["booked", "paid the deposit", "deposit paid"],
    "lost": ["declined", "passed", "went with", "not going ahead", "decided not", "lost"],
    "cold": ["cold", "no reply", "hasn't replied", "went quiet"],
    "awaiting_reply": ["waiting on us", "owe them a reply", "awaiting our reply", "haven't replied"],
    "waiting_on_client": ["waiting on them", "waiting to hear back", "waiting on the client"],
}


def find_money(text: str) -> set[int]:
    """Dollar amounts as whole dollars: "$2,395" -> 2395, "$2395" -> 2395. Ignore numbers without a $."""
    raise NotImplementedError  # TODO


def find_dates(text: str) -> set[date]:
    """Dates in the formats the inbox uses: "2026-05-23", "Saturday, May 23, 2026", "May 23rd, 2026",
    "May 23 2026", "Dec 31, 2026", "5/23/26", "5/23/2026". Dates without a year can be ignored."""
    raise NotImplementedError  # TODO


def find_numbers(text: str) -> set[int]:
    """Plain whole numbers: "about 115 guests" -> {115}. "1,000" -> 1000."""
    raise NotImplementedError  # TODO


def score_answer(answer: str, q: Question) -> Score:
    """Correct means ALL of:
    - should_defer questions: the answer defers (DEFER_PHRASES).  Others: it does NOT defer.
    - every expected fact is found (money/date/number via the finders; text: venue name case-insensitive,
      or a status via STATUS_WORDS, where a negated "not booked" must NOT count as booked)
    - no must_not_contain phrase appears (case-insensitive)
    """
    raise NotImplementedError  # TODO


def recall_at_k(retrieved: list[tuple[str, str]], gold: list[Evidence], k: int = 5) -> float:
    """retrieved = [(thread_id, chunk_text), ...] best first. Return the fraction of gold evidence found in the
    top k: a gold item is found if a chunk FROM THE SAME THREAD contains its quote, comparing with case and
    whitespace normalized. No gold evidence (count / not-in-inbox questions) -> return 1.0."""
    raise NotImplementedError  # TODO
