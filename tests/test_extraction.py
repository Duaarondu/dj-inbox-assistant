# tests/test_extraction.py
"""Lead extraction with a fake model that returns the ground truth: tests the plumbing and the scoring."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dj_inbox.extract.extract import ExtractedLead, ThreadLeads, extract_all, status
from dj_inbox.ingest import CompanyConfig, read_mbox
from evals.run_extraction import COMPANY, TODAY, compare, counting, to_rows

DATA = Path(__file__).resolve().parents[1] / "synthetic_inbox" / "data"


@pytest.fixture(scope="module")
def threads():
    return read_mbox(DATA / "inbox.mbox", CompanyConfig(domains=["northshoresound.example"]))


@pytest.fixture(scope="module")
def truth():
    with (DATA / "ground_truth_leads.jsonl").open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def as_lead(r: dict) -> ExtractedLead:
    outcome = r["status"] if r["status"] in ("booked", "lost") else "open"
    return ExtractedLead(event_type=r["event_type"], event_date=r["event_date"], venue=r["venue"],
                         venue_town=r["venue_town"], guest_count=r["guest_count"], hours=r["hours"],
                         budget=r["budget"], package_quoted=r["package_quoted"], addons_quoted=r["addons_quoted"],
                         quoted_price=r["quoted_price"], outcome=outcome)


class PerfectModel:
    """Fake Gemini: answers every thread with its true leads (looked up by the thread text it was sent)."""

    def __init__(self, threads, truth):
        by_text = {f"Subject: {t.subject}\n\n{t.clean_text()}": t.thread_id for t in threads}
        leads = {}
        for r in truth:
            leads.setdefault(r["thread_id"], []).append(as_lead(r))
        self.answers = {text: ThreadLeads(leads=leads[tid]) for text, tid in by_text.items()}
        self.calls = 0
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, model, contents, config):
        self.calls += 1
        assert config.response_schema is ThreadLeads
        return SimpleNamespace(parsed=self.answers[contents[0]], text="",
                               usage_metadata=SimpleNamespace(prompt_token_count=400, candidates_token_count=80,
                                                              thoughts_token_count=0))


def test_perfect_extraction_scores_100_percent_and_counts_right(threads, truth, tmp_path):
    model = PerfectModel(threads, truth)
    extracted, usage, calls = extract_all(model, "gemini-3.5-flash-lite", threads, cache_dir=tmp_path)
    rows = to_rows(threads, extracted)
    result = compare(rows, truth)
    assert all(a == 1.0 for a in result["accuracy"].values()), result["misses"]
    assert result["lead_count_correct"] == "200/200"
    assert all(got == want for _, got, want in counting(rows, truth))
    assert calls == 200 and usage.input == 200 * 400


def test_extractions_are_cached(threads, truth, tmp_path):
    model = PerfectModel(threads, truth)
    extract_all(model, "gemini-3.5-flash-lite", threads[:10], cache_dir=tmp_path)
    _, _, calls = extract_all(model, "gemini-3.5-flash-lite", threads[:10], cache_dir=tmp_path)
    assert calls == 0 and model.calls == 10


def test_wrong_fields_are_reported(threads, truth):
    rows = to_rows(threads[:1], {threads[0].thread_id: ThreadLeads(leads=[
        as_lead(next(r for r in truth if r["thread_id"] == threads[0].thread_id)).model_copy(
            update={"quoted_price": 1})])})
    result = compare(rows, [r for r in truth if r["thread_id"] == threads[0].thread_id])
    assert result["accuracy"]["quoted_price"] == 0 and result["accuracy"]["venue"] == 1


def test_unstated_towns_are_not_scored(threads, truth):
    from evals.run_extraction import unstated_towns
    skip = unstated_towns(threads, truth)
    assert 20 < len(skip) < len(truth)                       # website forms name the venue, not the town
    text = {t.thread_id: t.clean_text() for t in threads}
    assert all(r["venue_town"] not in text[r["thread_id"]] for r in truth if (r["thread_id"], r["lead_index"]) in skip)


def test_status_is_computed_not_extracted(threads):
    t = next(t for t in threads if t.messages[-1].sender_role == "client")
    open_lead = ExtractedLead(event_type="wedding", outcome="open")
    assert status(open_lead, t, TODAY) == "awaiting_reply"
    assert status(open_lead.model_copy(update={"outcome": "booked"}), t, TODAY) == "booked"


def test_source_from_form_address(threads):
    rows = to_rows(threads, {t.thread_id: ThreadLeads(leads=[ExtractedLead(event_type="other", outcome="open")])
                             for t in threads})
    forms = {r["thread_id"] for r in rows if r["source"] == "website_form"}
    assert forms == {t.thread_id for t in threads if COMPANY.is_form(t.messages[0].sender_email)} and len(forms) > 30
