# dj_inbox/worker/app.py
"""The private worker service (Cloud Run `dj-inbox-worker`, no public access).

    POST /tasks/import  {"company_id": "c_..."}   <- Cloud Tasks only (Cloud Run IAM: only the tasks
                                                     service account may invoke this service)
Answers:
    200  done, or a permanent problem stored on the connection (e.g. the owner must reconnect)
    503  temporary problem (Gmail rate limit, KMS hiccup): Cloud Tasks retries with backoff

Runs as the dj-inbox-worker service account: KMS decrypt-only on the token key, DB login app_ingest
(writes only the company of each transaction), Gemini for embeddings + extraction. The request stays open for
the whole import (Cloud Run keeps CPU while a request is in flight; Cloud Tasks deadline: 30 min).
"""
from __future__ import annotations

import json
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from dj_inbox.worker.importer import RetryLater, run_import

log = logging.getLogger("dj_inbox.worker")
COMPANY_ID = re.compile(r"^c_[0-9a-f]{16}$")
DEFAULT_MODEL = "gemini-3.5-flash-lite"


class ImportTask(BaseModel):
    company_id: str = Field(min_length=1, max_length=40)


def create_worker_app(deps: dict[str, Any] | None = None) -> FastAPI:
    """deps: pool, decrypt, make_session, embed, extract, model (tests pass fakes; production builds them)."""
    state: dict[str, Any] = dict(deps or {})

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not state:
            state.update(_production_deps())
        yield

    app = FastAPI(title="DJ Inbox worker", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.post("/tasks/import")
    async def import_task(task: ImportTask) -> dict:
        if not COMPANY_ID.match(task.company_id):
            raise HTTPException(422, "bad company id")                  # also keeps 'northshore' (the demo) out
        try:
            result = await run_in_threadpool(run_import, state["pool"], task.company_id,
                                             decrypt=state["decrypt"], make_session=state["make_session"],
                                             embed=state["embed"], extract=state["extract"], model=state["model"])
        except RetryLater:
            raise HTTPException(503, "temporary problem; retry") from None
        return {"company_id": result.company_id, "status": result.status, "new_messages": result.messages_new,
                "threads": result.threads_total, "leads": result.leads, "seconds": result.seconds}

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
    }


app = create_worker_app()
logging.basicConfig(level=logging.INFO, format="%(message)s")
log.info(json.dumps({"event": "worker_started"}))
