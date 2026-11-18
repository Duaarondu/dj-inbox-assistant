# tests/test_store_loader.py
"""get_store(): registered stores (the demo) stay; loaded stores (signed-in companies) reload after the TTL."""
from __future__ import annotations

import pytest

from dj_inbox.agent import store as st


@pytest.fixture(autouse=True)
def clean():
    st.clear()
    yield
    st.clear()


def make(company_id: str, version: int = 0) -> st.InboxStore:
    return st.InboxStore(company_id, f"{company_id} v{version}", [], {}, lambda q, k: [])


def test_unknown_company_without_a_loader_is_refused():
    with pytest.raises(PermissionError):
        st.get_store("c_x")


def test_registered_demo_store_never_expires_and_never_hits_the_loader():
    st.register(make("northshore"))
    st.set_loader(lambda cid: pytest.fail("the demo must not be loaded from the database"), ttl_s=0)
    assert st.get_store("northshore", now=lambda: 1e9).company_name == "northshore v0"


def test_loaded_store_is_cached_then_reloaded_after_the_ttl():
    loads: list[str] = []

    def loader(cid: str) -> st.InboxStore:
        loads.append(cid)
        return make(cid, len(loads))

    st.set_loader(loader, ttl_s=10)
    assert st.get_store("c_a", now=lambda: 100).company_name == "c_a v1"
    assert st.get_store("c_a", now=lambda: 105).company_name == "c_a v1"       # cached
    assert st.get_store("c_a", now=lambda: 111).company_name == "c_a v2"       # new emails show up
    assert loads == ["c_a", "c_a"]


def test_no_company_id_never_reaches_the_loader():
    st.set_loader(lambda cid: pytest.fail("loader called"), ttl_s=10)
    for bad in (None, ""):
        with pytest.raises(PermissionError):
            st.get_store(bad)
