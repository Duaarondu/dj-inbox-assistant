# dj_inbox/api/app.py
"""The HTTP API the Vercel page talks to.

    uvicorn dj_inbox.api.app:app --reload --port 8080      (local; needs .env + gcloud sign-in + leads.jsonl)

    POST /ask            {"question": "..."} -> answer, cited source emails, tools used, tokens, cost, latency
    GET  /leads          the lead tracker: ?status=cold&event_type=wedding&source=website_form (all optional);
                         one row per inquiry, like count_leads. ?all_events=true adds 2nd events (engagement party)
    GET  /threads/{id}   one thread's emails (what a cited source links to)
    GET  /healthz        is it up, and what data is loaded

company_id: DEMO ONLY, it is fixed on the server (DEMO_COMPANY_ID) and never read from a request. After
Google sign-in it comes from the verified user. Every endpoint goes through get_store(company_id).

Cost guard: a daily question cap (MAX_QUESTIONS_PER_DAY) is checked BEFORE every model call, because the
demo is public and every question spends credits. In-memory, so deploy with max-instances=1 until there's a DB.
"""
from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import date
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dj_inbox.agent.runtime import RunResult, run_workflow_async
from dj_inbox.agent.store import InboxStore, get_store
from dj_inbox.ingest import EmailThread

DEFAULT_MODEL = "gemini-3.5-flash-lite"     # Nov 9: same 50/50 as Flash, 4x cheaper, 3x faster
MAX_QUESTION_CHARS = 500
EMAIL_ID = re.compile(r"\bt\d+-\d+\b")
CITATION_TAIL = re.compile(r"\s*\(sources?:[^)]*\)\s*$", re.IGNORECASE)

log = logging.getLogger("dj_inbox.api")


# ---------- request / response shapes ----------

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


class Source(BaseModel):
    email_id: str
    thread_id: str
    subject: str
    sent_at: str
    sender: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]           # emails the answer cited AND a tool actually returned
    matched_threads: list[str]      # threads a count matched (for "show me them")
    tools: list[str]
    model: str
    latency_s: float
    tokens_in: int
    tokens_out: int
    cost_usd: float


# ---------- cost guard ----------

class DailyCap:
    """At most `limit` questions per calendar day. try_take() is called before the model is."""

    def __init__(self, limit: int, today: Callable[[], date] = date.today):
        self.limit, self.today = limit, today
        self.day, self.used = today(), 0

    def try_take(self) -> bool:
        today = self.today()                     # read the clock once: two reads could straddle midnight
        if today != self.day:
            self.day, self.used = today, 0
        if self.used >= self.limit:
            return False
        self.used += 1
        return True


# ---------- helpers ----------

def _walk_email_ids(value: Any) -> set[str]:
    """Every email_id anywhere inside the tool results."""
    if isinstance(value, dict):
        found = {value["email_id"]} if isinstance(value.get("email_id"), str) else set()
        return found.union(*(_walk_email_ids(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(_walk_email_ids(v) for v in value))
    return set()


def _cited_sources(answer: str, result: RunResult, store: InboxStore) -> list[Source]:
    """Only ids the model cited that a tool really returned: a made-up id never becomes a link."""
    returned = _walk_email_ids([call.response for call in result.tool_calls])
    sources = []
    for email_id in dict.fromkeys(EMAIL_ID.findall(answer)):           # keep order, drop repeats
        if email_id not in returned:
            continue
        thread_id, index = email_id.rsplit("-", 1)
        thread = store.threads.get(thread_id)
        if thread is None or int(index) >= len(thread.messages):
            continue
        message = thread.messages[int(index)]
        sources.append(Source(email_id=email_id, thread_id=thread_id, subject=thread.subject,
                              sent_at=message.sent_at.isoformat(), sender=message.sender_name))
    return sources


def _matched_threads(result: RunResult) -> list[str]:
    for call in result.tool_calls:
        if call.name == "count_leads" and isinstance(call.response, dict):
            return list(call.response.get("thread_ids", []))
    return []


def client_label(thread: EmailThread) -> str:
    """Who the lead is, for the tracker: the client's name, the "Name:" line of a website form, or the subject."""
    first = thread.messages[0]
    if first.sender_role == "client" and "form" not in first.sender_name.lower():
        return first.sender_name
    match = re.search(r"^Name:\s*(.+)$", first.body_clean, re.MULTILINE)
    return match.group(1).strip() if match else thread.subject


def _cost(model: str, tokens_in: int, tokens_out: int) -> float:
    from evals.run_baseline import PRICES  # one price table for evals and the API
    price = PRICES.get(model)
    return (tokens_in * price["input"] + tokens_out * price["output"]) / 1e6 if price else 0.0


# ---------- the app ----------

def create_app(store: InboxStore | None = None, workflow: Any = None, model: str | None = None,
               cap: DailyCap | None = None) -> FastAPI:
    """Tests pass a small store, a FakeLlm workflow and a cap. In production everything loads at startup."""
    model = model or os.environ.get("AGENT_MODEL", DEFAULT_MODEL)
    company_id = os.environ.get("DEMO_COMPANY_ID", "northshore")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal store, workflow
        if store is None:
            from dj_inbox.agent.store import load_northshore
            store = load_northshore()
        if workflow is None:
            from dj_inbox.agent.answer import build_workflow
            from evals.run_baseline import TODAY
            workflow = build_workflow(model, store.company_name, TODAY)
        yield

    app = FastAPI(title="DJ Inbox Assistant", lifespan=lifespan)
    app.state.cap = cap or DailyCap(int(os.environ.get("MAX_QUESTIONS_PER_DAY", "300")))
    origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"])

    def company() -> InboxStore:
        return get_store(company_id)                # the ONLY way an endpoint reaches data

    @app.get("/healthz")
    def healthz() -> dict:
        s = company()
        return {"ok": True, "company": s.company_name, "threads": len(s.threads), "leads": len(s.leads),
                "model": model}

    @app.post("/ask", response_model=AskResponse)
    async def ask(body: AskRequest) -> AskResponse:
        question = body.question.strip()
        if not question:
            raise HTTPException(422, "Question is empty.")
        if not app.state.cap.try_take():            # checked BEFORE any model call
            raise HTTPException(429, "The demo's daily question limit is reached. Try again tomorrow.")
        s = company()
        try:
            result = await run_workflow_async(workflow, question, s.company_id)
        except Exception as e:
            log.error(json.dumps({"event": "ask_failed", "company_id": s.company_id, "model": model,
                                  "error": f"{type(e).__name__}: {e}"[:300]}))
            raise HTTPException(502, "The model call failed. Please try again.") from e

        raw = str(result.output or "")
        sources = _cited_sources(raw, result, s)
        cost = _cost(model, result.tokens_in, result.tokens_out)
        log.info(json.dumps({                       # one structured line per question (Cloud Logging reads it)
            "event": "ask", "company_id": s.company_id, "model": model, "question_chars": len(question),
            "tools": [c.name for c in result.tool_calls], "tool_args": [c.args for c in result.tool_calls],
            "email_ids": [x.email_id for x in sources], "model_calls": result.model_calls,
            "tokens_in": result.tokens_in, "tokens_out": result.tokens_out, "cost_usd": round(cost, 6),
            "latency_s": round(result.latency_s, 2)}))
        return AskResponse(answer=CITATION_TAIL.sub("", raw).strip(), sources=sources,
                           matched_threads=_matched_threads(result), tools=[c.name for c in result.tool_calls],
                           model=model, latency_s=round(result.latency_s, 2), tokens_in=result.tokens_in,
                           tokens_out=result.tokens_out, cost_usd=round(cost, 6))

    @app.get("/leads")
    def leads(status: str | None = Query(None), event_type: str | None = Query(None),
              source: str | None = Query(None), all_events: bool = Query(False)) -> dict:
        s = company()
        filters = {k: v for k, v in {"status": status, "event_type": event_type, "source": source}.items() if v}
        # Default: one row per inquiry (lead_index 0), the same rule as count_leads, so the tracker and the
        # agent agree ("60 cold" in an answer = 60 rows here, not 67 with second events).
        rows = [r for r in s.leads if (all_events or r["lead_index"] == 0)
                and all(r.get(k) == v for k, v in filters.items())]
        rows.sort(key=lambda r: (r.get("event_date") or "9999", r["thread_id"], r["lead_index"]))
        out = []
        for r in rows:
            thread = s.threads.get(r["thread_id"])
            out.append({**r, "client": client_label(thread) if thread else None,
                        "subject": thread.subject if thread else None})
        return {"count": len(out), "leads": out}

    @app.get("/threads/{thread_id}")
    def thread(thread_id: str) -> dict:
        t = company().threads.get(thread_id)
        if t is None:                               # also what another company's thread id gets
            raise HTTPException(404, "No such thread.")
        return {"thread_id": t.thread_id, "subject": t.subject, "client": client_label(t),
                "emails": [{"email_id": f"{t.thread_id}-{i}", "sent_at": m.sent_at.isoformat(),
                            "sender_role": m.sender_role, "sender": m.sender_name, "body": m.body_clean}
                           for i, m in enumerate(t.messages)]}

    return app


logging.basicConfig(level=logging.INFO, format="%(message)s")
app = create_app()
