# dj_inbox/worker/app.py
"""The private worker service (Cloud Run `dj-inbox-worker`, no public access).

    POST /tasks/import  {"company_id": "c_..."}   <- Cloud Tasks (Connect, Sync again); then (re)starts the watch
    POST /pubsub/gmail  Pub/Sub push envelope       <- Gmail "this mailbox changed" -> incremental sync
    POST /tasks/renew-watches                       <- Cloud Scheduler, daily (a Gmail watch lasts ~7 days)

Who may call: Cloud Run IAM lets only two identities in at all (run.invoker): dj-inbox-tasks (Cloud Tasks and
Cloud Scheduler) and dj-inbox-push (Pub/Sub). Cloud Run has already checked the token's signature and audience;
the routes below also check WHICH of the two it is (the `email` claim), so the push identity can only deliver
notifications and can't start full imports or renewals. This relies on --no-allow-unauthenticated
(scripts/deploy_worker.sh): without Cloud Run's check, the claim could be forged.
Answers:
    200  done, or a permanent problem stored on the connection (e.g. the owner must reconnect)
    503  temporary problem (Gmail rate limit, KMS hiccup): Cloud Tasks retries with backoff

Runs as the dj-inbox-worker service account: KMS decrypt-only on the token key, DB login app_ingest
(writes only the company of each transaction), Gemini for embeddings + extraction. The request stays open for
the whole import (Cloud Run keeps CPU while a request is in flight; Cloud Tasks deadline: 30 min).
"""
from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from dj_inbox.worker.importer import RetryLater, companies_to_watch, company_for_gmail, run_import, run_sync, run_watch

log = logging.getLogger("dj_inbox.worker")
COMPANY_ID = re.compile(r"^c_[0-9a-f]{16}$")
DEFAULT_MODEL = "gemini-3.5-flash-lite"


class ImportTask(BaseModel):
    company_id: str = Field(min_length=1, max_length=40)


def caller_email(request: Request) -> str | None:
    """The `email` claim of the OIDC token Cloud Run already verified (signature not re-checked here)."""
    auth = request.headers.get("authorization", "")
    parts = auth.removeprefix("Bearer ").split(".")
    if not auth.startswith("Bearer ") or len(parts) < 2:
        return None
    try:
        claims = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))
    except (binascii.Error, ValueError):
        return None
    return claims.get("email") if claims.get("email_verified") else None


def notification(envelope: Any) -> tuple[str, str] | None:
    """(mailbox, Gmail history id) from a Pub/Sub push envelope; None if it isn't a Gmail notification."""
    try:
        data = json.loads(base64.b64decode(envelope["message"]["data"]))
        return str(data["emailAddress"]).lower(), str(int(data["historyId"]))
    except (KeyError, TypeError, ValueError, binascii.Error):
        return None


def create_worker_app(deps: dict[str, Any] | None = None) -> FastAPI:
    """deps: pool, decrypt, make_session, embed, extract, model (tests pass fakes; production builds them)."""
    state: dict[str, Any] = dict(deps or {})

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not state:
            state.update(_production_deps())
        yield

    def require_caller(request: Request, setting: str) -> None:
        expected = state.get(setting)
        if not expected or caller_email(request) != expected:          # unset -> refuse (fail closed)
            raise HTTPException(403, "not allowed")

    def full_deps() -> dict[str, Any]:
        return {k: state[k] for k in ("decrypt", "make_session", "embed", "extract", "model")}

    def watch(company_id: str) -> str:
        if not state.get("watch_topic"):
            return "off"
        try:
            return run_watch(state["pool"], company_id, decrypt=state["decrypt"],
                             make_session=state["make_session"], topic=state["watch_topic"])
        except RetryLater:
            return "retry_tomorrow"                                     # the daily renewal tries again

    app = FastAPI(title="DJ Inbox worker", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.post("/tasks/import")
    async def import_task(task: ImportTask, request: Request) -> dict:
        require_caller(request, "tasks_invoker")
        if not COMPANY_ID.match(task.company_id):
            raise HTTPException(422, "bad company id")                  # also keeps 'northshore' (the demo) out
        try:
            result = await run_in_threadpool(run_import, state["pool"], task.company_id,
                                             decrypt=state["decrypt"], make_session=state["make_session"],
                                             embed=state["embed"], extract=state["extract"], model=state["model"])
        except RetryLater:
            raise HTTPException(503, "temporary problem; retry") from None
        watching = await run_in_threadpool(watch, task.company_id) if result.status == "ready" else "no"
        return {"company_id": result.company_id, "status": result.status, "new_messages": result.messages_new,
                "threads": result.threads_total, "leads": result.leads, "seconds": result.seconds,
                "watch": watching}

    @app.post("/pubsub/gmail")
    async def gmail_push(request: Request) -> dict:
        """2xx = Pub/Sub acks (done or nothing to do); 503 = Pub/Sub redelivers later with backoff."""
        require_caller(request, "push_invoker")
        try:
            found = notification(await request.json())
        except ValueError:
            found = None
        if found is None:
            log.warning(json.dumps({"event": "push_ignored", "why": "not a Gmail notification"}))
            return {"ignored": "not a Gmail notification"}             # ack: retrying can't fix it
        mailbox, history_id = found
        company_id = await run_in_threadpool(company_for_gmail, state["pool"], mailbox)
        if company_id is None:
            log.warning(json.dumps({"event": "push_ignored", "why": "unknown mailbox"}))
            return {"ignored": "unknown mailbox"}
        log.info(json.dumps({"event": "push_received", "company_id": company_id, "history_id": history_id}))
        try:
            result = await run_in_threadpool(run_sync, state["pool"], company_id, **full_deps())
        except RetryLater:
            raise HTTPException(503, "temporary problem; retry") from None
        return {"company_id": company_id, "status": result.status, "new_messages": result.messages_new,
                "seconds": result.seconds}

    @app.post("/tasks/renew-watches")
    async def renew_watches(request: Request) -> dict:
        require_caller(request, "tasks_invoker")
        if not state.get("watch_topic"):
            raise HTTPException(503, "GMAIL_WATCH_TOPIC not set")
        counts: dict[str, int] = {}
        for company_id in await run_in_threadpool(companies_to_watch, state["pool"]):
            outcome = await run_in_threadpool(watch, company_id)
            counts[outcome] = counts.get(outcome, 0) + 1
        log.info(json.dumps({"event": "watches_renewed", **counts}))
        return counts

    return app


def _production_deps() -> dict[str, Any]:
    import google.auth
    import requests
    from google.auth.transport.requests import AuthorizedSession

    from dj_inbox.db import make_pool
    from dj_inbox.extract.extract import extract_all
    from dj_inbox.gcp import gemini_client
    from dj_inbox.rag.embed import Embedder
    from dj_inbox.worker.importer import GmailSession, KmsDecrypt

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloudkms"])
    client_id, client_secret = os.environ["GMAIL_CLIENT_ID"], os.environ["GMAIL_CLIENT_SECRET"]
    http = requests.Session()
    client = gemini_client()
    embedder = Embedder(client, cache_dir=None)
    model = os.environ.get("EXTRACT_MODEL", DEFAULT_MODEL)
    return {
        "pool": make_pool(max_size=2),                                    # DATABASE_URL = the app_ingest login
        "decrypt": KmsDecrypt(AuthorizedSession(creds)),
        "make_session": lambda refresh: GmailSession(http, refresh, client_id, client_secret),
        "embed": lambda texts: embedder.embed(texts, "RETRIEVAL_DOCUMENT"),
        "extract": lambda threads: extract_all(client, model, threads, cache_dir=None)[0],
        "model": model,
        "push_invoker": os.environ.get("PUSH_INVOKER", ""),
        "tasks_invoker": os.environ.get("TASKS_INVOKER", ""),
        "watch_topic": os.environ.get("GMAIL_WATCH_TOPIC", ""),
    }


app = create_worker_app()
logging.basicConfig(level=logging.INFO, format="%(message)s")
log.info(json.dumps({"event": "worker_started"}))
