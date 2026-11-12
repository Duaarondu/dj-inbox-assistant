# tests/conftest.py
"""Shared test safety checks.

Database tests (test_pgvector.py, test_rls.py) create schemas, roles and policies and switch database users.
They must only ever run on a throwaway Postgres (CI's container, or a local one). On Nov 12 TEST_DATABASE_URL
was found pointing at the production Supabase database (Bug Journal #19), so refuse that outright.
"""
from __future__ import annotations

import os
from urllib.parse import urlparse

import pytest

PRODUCTION_HOST_MARKERS = ("supabase.co", "supabase.com")


def pytest_configure(config):
    url = os.environ.get("TEST_DATABASE_URL")
    host = (urlparse(url).hostname or "") if url else ""
    if any(marker in host for marker in PRODUCTION_HOST_MARKERS):
        pytest.exit(
            f"TEST_DATABASE_URL points at a hosted Supabase database ({host}). Database tests create roles and "
            "drop schemas: run them only against a throwaway Postgres. Unset it: `unset TEST_DATABASE_URL`.",
            returncode=2,
        )
