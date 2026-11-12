# tests/test_onboarding_api.py
"""GET /me and POST /companies: onboarding needs a verified user, and the user can't choose who or which company."""
from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from dj_inbox.agent import store as store_mod
from dj_inbox.agent.answer import build_workflow
from dj_inbox.agent.store import InboxStore
from dj_inbox.agent.testing import FakeLlm
from dj_inbox.api.app import create_app
from dj_inbox.api.auth import AlreadyHasCompany, TokenVerifier
from tests.test_api import LEADS, T1, T2
from tests.test_auth import ATTACKER_KEY, SUPABASE, FakeKeys, bearer, token

NEW_USER = str(uuid.uuid4())


class FakeCompanies:
    """Stands in for create_company(): remembers owners, refuses a second company, records who asked."""

    def __init__(self):
        self.owner_of: dict[str, str] = {}
        self.calls: list[tuple[str, str]] = []

    def create(self, user_id: str, name: str) -> str:
        self.calls.append((user_id, name))
        if not name.strip():
            raise ValueError("Company name must be 1 to 120 characters.")
        if user_id in self.owner_of.values():
            raise AlreadyHasCompany
        company_id = f"c_{len(self.owner_of):016d}"
        self.owner_of[company_id] = user_id
        return company_id

    def memberships(self, user_id: str) -> list[str]:
        return [c for c, owner in self.owner_of.items() if owner == user_id]


@pytest.fixture
def setup():
    store_mod.clear()
    demo = InboxStore("northshore", "Northshore Sound Co.", LEADS, {"t1": T1, "t2": T2}, lambda q, k: ["t1"])
    store_mod.register(demo)
    companies = FakeCompanies()
    app = create_app(store=demo, workflow=build_workflow(FakeLlm()), model="gemini-3.5-flash-lite",
                     verifier=TokenVerifier(SUPABASE, keys=FakeKeys()), memberships=companies.memberships,
                     create_company=companies.create)
    with TestClient(app) as c:
        yield c, companies
    store_mod.clear()


def test_full_onboarding_flow(setup):
    c, companies = setup
    h = bearer(token(sub=NEW_USER))
    assert c.get("/me", headers=h).json() == {"user_id": NEW_USER, "company_id": None}
    assert c.get("/leads", headers=h).status_code == 403                       # no company yet
    r = c.post("/companies", json={"name": "Harbor Lights DJs"}, headers=h)
    assert r.status_code == 201
    company_id = r.json()["company_id"]
    assert c.get("/me", headers=h).json()["company_id"] == company_id
    assert c.get("/leads", headers=h).status_code == 404                       # company exists, no inbox yet
    assert c.get("/threads/t1", headers=h).status_code == 404                  # and never the demo's data


def test_a_second_company_is_409(setup):
    c, _ = setup
    h = bearer(token(sub=NEW_USER))
    assert c.post("/companies", json={"name": "One"}, headers=h).status_code == 201
    assert c.post("/companies", json={"name": "Two"}, headers=h).status_code == 409


@pytest.mark.parametrize("headers", [{}, bearer(token(key=ATTACKER_KEY)), {"Authorization": "Bearer"}])
def test_onboarding_needs_a_verified_user(setup, headers):
    c, companies = setup
    assert c.post("/companies", json={"name": "Sneaky"}, headers=headers).status_code == 401
    assert c.get("/me", headers=headers).status_code == 401
    assert companies.calls == []                                               # never reached the database


def test_the_body_cannot_choose_the_owner_or_the_company(setup):
    c, companies = setup
    victim = str(uuid.uuid4())
    r = c.post("/companies", headers=bearer(token(sub=NEW_USER)),
               json={"name": "Takeover", "user_id": victim, "company_id": "northshore"})
    assert r.status_code == 201 and r.json()["company_id"] != "northshore"
    assert companies.calls == [(NEW_USER, "Takeover")]                         # the token's user, nothing else


@pytest.mark.parametrize("name", ["", "x" * 121])
def test_bad_names_are_422(setup, name):
    c, _ = setup
    assert c.post("/companies", json={"name": name}, headers=bearer(token(sub=NEW_USER))).status_code == 422
