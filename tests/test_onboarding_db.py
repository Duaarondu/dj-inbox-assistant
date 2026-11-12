# tests/test_onboarding_db.py
"""create_company() (db/migrations/004): the one write app_api may do, and the attacks it must stop.

Runs only when TEST_DATABASE_URL is set (CI's pgvector job), in a temporary schema, acting as app_api.
"""
from __future__ import annotations

import os
import uuid

import psycopg
import pytest

from dj_inbox.db import CONNECT_KW, safe_connect
from dj_inbox.db.migrate import migrate

URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="needs TEST_DATABASE_URL (a throwaway Postgres)")


@pytest.fixture
def conn():
    schema = f"test_{uuid.uuid4().hex[:10]}"
    with safe_connect(URL, **CONNECT_KW) as c:
        c.execute("create extension if not exists vector")
        c.execute(f"create schema {schema}")
        c.execute(f"set search_path = {schema}, public, extensions")
        migrate(c)
        c.execute("set role app_api")
        try:
            yield c
        finally:
            c.execute("reset role")
            c.execute(f"drop schema {schema} cascade")


def as_user(conn, user_id, fn):
    with conn.transaction():
        if user_id is not None:
            conn.execute("select set_config('app.user_id', %s, true)", (str(user_id),))
        return fn()


def create(conn, user_id, name="Harbor Lights DJs"):
    return as_user(conn, user_id, lambda: conn.execute("select create_company(%s)", (name,)).fetchone()[0])


def my_companies(conn, user_id):
    return as_user(conn, user_id, lambda: [r[0] for r in conn.execute("select company_id from company_members")])


def test_a_new_user_creates_a_company_and_owns_it(conn):
    user = uuid.uuid4()
    company_id = create(conn, user)
    assert company_id.startswith("c_") and len(company_id) == 18
    assert my_companies(conn, user) == [company_id]
    with conn.transaction():
        conn.execute("select set_config('app.company_id', %s, true)", (company_id,))
        assert conn.execute("select name from companies").fetchall() == [("Harbor Lights DJs",)]


def test_one_company_per_owner(conn):
    user = uuid.uuid4()
    create(conn, user)
    with pytest.raises(psycopg.errors.UniqueViolation):
        create(conn, user, "Second Try DJs")


def test_must_be_signed_in(conn):
    with pytest.raises(psycopg.errors.InvalidAuthorizationSpecification):
        create(conn, None)


@pytest.mark.parametrize("name", ["", "   ", "x" * 121])
def test_name_is_checked(conn, name):
    with pytest.raises(psycopg.errors.InvalidParameterValue):
        create(conn, uuid.uuid4(), name)


def test_attack_join_someone_elses_company_directly(conn):
    """The attack from question 2: insert yourself into an existing company. app_api has no INSERT at all."""
    attacker = uuid.uuid4()
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        as_user(conn, attacker, lambda: conn.execute(
            "insert into company_members (user_id, company_id, role) values (%s, 'northshore', 'owner')",
            (attacker,)))
    assert my_companies(conn, attacker) == []


def test_attack_choose_the_company_id_or_owner(conn):
    """The function takes only a name: there is no way to pass an existing company id or another user."""
    with pytest.raises(psycopg.errors.UndefinedFunction):
        as_user(conn, uuid.uuid4(), lambda: conn.execute(
            "select create_company(%s, %s)", ("northshore", "Takeover")))


def test_names_are_data_not_sql(conn):
    user = uuid.uuid4()
    company_id = create(conn, user, "x'); delete from companies; --")
    with conn.transaction():
        conn.execute("select set_config('app.company_id', %s, true)", (company_id,))
        assert conn.execute("select name from companies").fetchone()[0] == "x'); delete from companies; --"
    conn.execute("reset role")
    assert conn.execute("select count(*) from companies where company_id = 'northshore'").fetchone()[0] == 1
    conn.execute("set role app_api")


def test_function_is_locked_down(conn):
    conn.execute("reset role")
    config, definer = conn.execute("select proconfig, prosecdef from pg_proc where proname = 'create_company' "
                                   "and pronamespace = current_schema()::regnamespace").fetchone()
    assert definer is True
    assert any(c.startswith("search_path=") and "pg_temp" in c for c in config)
    grantees = {r[0] for r in conn.execute(
        "select grantee from information_schema.routine_privileges where routine_name = 'create_company' "
        "and routine_schema = current_schema() and privilege_type = 'EXECUTE'")}
    assert "app_api" in grantees and "PUBLIC" not in grantees
    conn.execute("set role app_api")


def test_the_index_stops_a_second_owned_company_even_past_the_check(conn):
    """Two clicks at once can both pass the 'already have a company?' check; the unique index still says no."""
    conn.execute("reset role")
    user = uuid.uuid4()
    conn.execute("insert into companies (company_id, name) values ('c_race1', 'One'), ('c_race2', 'Two')")
    conn.execute("insert into company_members (user_id, company_id, role) values (%s, 'c_race1', 'owner')", (user,))
    with pytest.raises(psycopg.errors.UniqueViolation):
        with conn.transaction():
            conn.execute("insert into company_members (user_id, company_id, role) values (%s, 'c_race2', 'owner')",
                         (user,))
    conn.execute("set role app_api")


def test_the_api_wrapper_maps_database_errors(conn):
    """create_company_in_db (what POST /companies calls) against the real function, as app_api."""
    from dj_inbox.api.auth import AlreadyHasCompany, create_company_in_db

    user = str(uuid.uuid4())
    company_id = create_company_in_db(conn, user, "  Harbor Lights DJs  ")
    assert my_companies(conn, user) == [company_id]
    with pytest.raises(AlreadyHasCompany):
        create_company_in_db(conn, user, "Again")
    with pytest.raises(ValueError):
        create_company_in_db(conn, str(uuid.uuid4()), "   ")
