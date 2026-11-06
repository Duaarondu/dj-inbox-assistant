# dj_inbox/gcp.py
"""One place to create the Gemini client and to decide which errors are worth retrying.

Every request has a 60-second timeout: without one, a dead connection (e.g. the laptop slept mid-run, Nov 5)
waits forever. With it, the hang becomes an error, and is_retryable() lets the caller try again.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIMEOUT_MS = 60_000
RETRYABLE_STATUS = {429, 500, 502, 503, 504}   # rate limited, or a temporary server problem


def gemini_client():
    """Gemini on Google Cloud, billed to GOOGLE_CLOUD_PROJECT from .env.
    Sign in once with: gcloud auth application-default login"""
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types

    load_dotenv(ROOT / ".env")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise SystemExit("Set GOOGLE_CLOUD_PROJECT in .env (your Google Cloud project id).")
    return genai.Client(enterprise=True, project=project, location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
                        http_options=types.HttpOptions(timeout=TIMEOUT_MS))


def is_retryable(e: Exception) -> bool:
    """Rate limits, temporary server errors and timeouts are worth another try; anything else is a real bug."""
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    return code in RETRYABLE_STATUS or isinstance(e, TimeoutError) or "Timeout" in type(e).__name__
