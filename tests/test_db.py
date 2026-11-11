# tests/test_db.py
"""Database helpers that need no database."""
from __future__ import annotations

import traceback

import pytest

from dj_inbox.db import DatabaseConnectError, safe_connect

SECRET = "Sup3r%Secr?t$#pw"


def test_a_malformed_url_never_shows_the_password():
    url = f"postgresql://postgres.abc:[{SECRET}]@pooler.example.com:5432/postgres"   # brackets + % ? $ #
    with pytest.raises(DatabaseConnectError) as caught:
        safe_connect(url, connect_timeout=1)
    shown = "".join(traceback.format_exception(caught.value))                       # what a log would contain
    assert SECRET not in shown and "Sup3r" not in shown
    assert "brackets" in str(caught.value)
