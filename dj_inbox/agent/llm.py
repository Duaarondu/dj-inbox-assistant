# dj_inbox/agent/llm.py
"""Gemini models for ADK agents, on the same Google Cloud project as the eval runners.

ADK builds its own google-genai client, so the settings from dj_inbox/gcp.py are passed in here:
enterprise=True (Gemini Enterprise Agent Platform), project + location from .env, a 60-second timeout,
and retries on rate limits and temporary server errors.
"""
from __future__ import annotations

import os

from google.adk.models import Gemini
from google.genai import types

from dj_inbox.gcp import RETRYABLE_STATUS, ROOT, TIMEOUT_MS

RETRY_ATTEMPTS = 4


def adk_gemini(model: str) -> Gemini:
    """An ADK model object for a Gemini model name, e.g. adk_gemini("gemini-3.5-flash-lite")."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise SystemExit("Set GOOGLE_CLOUD_PROJECT in .env (your Google Cloud project id).")
    return Gemini(
        model=model,
        client_kwargs={
            "enterprise": True,
            "project": project,
            "location": os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
            "http_options": types.HttpOptions(timeout=TIMEOUT_MS),
        },
        retry_options=types.HttpRetryOptions(attempts=RETRY_ATTEMPTS, http_status_codes=sorted(RETRYABLE_STATUS)),
    )
