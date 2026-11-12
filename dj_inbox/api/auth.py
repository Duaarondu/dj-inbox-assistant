# dj_inbox/api/auth.py
"""Who is asking, and which company's data they get.

    no Authorization header      -> the public demo company (synthetic data, read-only)
    "Bearer <Supabase token>"    -> verify it, then the user's company from company_members
    anything else / a bad token  -> 401 (never a silent fall back to the demo)
    valid token, no company yet  -> 403 (the website sends them to "name your company")

Token checks (Supabase Auth, ES256 signing key; public keys from the project's JWKS endpoint, no secret held):
    - algorithm must be ES256: a token claiming "none" or HS256 is refused before any key is looked at
      (the classic "algorithm confusion" attack: sign with HS256 using the PUBLIC key as the secret)
    - signature checked with the key named by the token's `kid`
    - audience "authenticated", issuer "<SUPABASE_URL>/auth/v1", not expired (30 s clock leeway)
    - `sub` (the user id) must be a UUID

Never log a token: on failure only the reason class is logged.
"""
from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import jwt

ALGORITHMS = ["ES256"]
AUDIENCE = "authenticated"
LEEWAY_S = 30

log = logging.getLogger("dj_inbox.api")


class AuthError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


@dataclass(frozen=True)
class Caller:
    company_id: str
    user_id: str | None        # None = anonymous visitor of the public demo

    @property
    def is_demo(self) -> bool:
        return self.user_id is None


class KeySource(Protocol):
    def get_signing_key_from_jwt(self, token: str) -> Any: ...


class TokenVerifier:
    """Checks Supabase access tokens. Public keys are fetched from JWKS and cached (10 min)."""

    def __init__(self, supabase_url: str, keys: KeySource | None = None):
        self.issuer = supabase_url.rstrip("/") + "/auth/v1"
        self.keys = keys or jwt.PyJWKClient(self.issuer + "/.well-known/jwks.json", cache_keys=True, lifespan=600)

    def verify(self, token: str) -> dict:
        try:
            if jwt.get_unverified_header(token).get("alg") not in ALGORITHMS:
                raise AuthError(401, "Invalid sign-in token.")
            key = self.keys.get_signing_key_from_jwt(token).key
            claims = jwt.decode(token, key, algorithms=ALGORITHMS, audience=AUDIENCE, issuer=self.issuer,
                                leeway=LEEWAY_S, options={"require": ["exp", "iat", "sub", "aud", "iss"]})
            uuid.UUID(claims["sub"])
        except AuthError:
            raise
        except jwt.PyJWKClientConnectionError as e:
            log.warning('{"event": "auth_keys_unreachable"}')
            raise AuthError(503, "Sign-in check is unavailable. Try again shortly.") from e
        except (jwt.PyJWTError, ValueError, KeyError, TypeError) as e:
            log.info('{"event": "auth_rejected", "reason": "%s"}', type(e).__name__)
            raise AuthError(401, "Invalid sign-in token.") from None
        return claims


MembershipLookup = Callable[[str], list[str]]    # user_id -> company ids, oldest membership first


def memberships_for(conn, user_id: str) -> list[str]:
    """The companies this user belongs to.

    Two locks: the WHERE (works even on a connection that skips RLS, like the table owner's) and, as app_api,
    the members_own_rows policy, which reads app.user_id set here for this transaction only.
    """
    with conn.transaction():
        conn.execute("select set_config('app.user_id', %s, true)", (user_id,))
        rows = conn.execute("select company_id from company_members where user_id = %s order by created_at",
                            (user_id,)).fetchall()
    return [company_id for (company_id,) in rows]


def db_memberships(pool) -> MembershipLookup:
    def lookup(user_id: str) -> list[str]:
        with pool.connection() as conn:
            return memberships_for(conn, user_id)
    return lookup


CompanyCreator = Callable[[str, str], str]      # (verified user_id, name) -> new company_id


class AlreadyHasCompany(Exception):
    pass


def signed_in_user(authorization: str | None, verifier: TokenVerifier | None) -> str:
    """The verified user id. Missing, malformed or bad token -> 401 (here sign-in is required, no demo)."""
    if authorization is None:
        raise AuthError(401, "Sign in first.")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthError(401, "Invalid sign-in token.")
    if verifier is None:                            # sign-in not configured on this server: refuse, don't guess
        raise AuthError(401, "Sign-in is not enabled on this server.")
    return verifier.verify(token.strip())["sub"]


def create_company_in_db(conn, user_id: str, name: str) -> str:
    """Calls the create_company() function (migration 004) as the verified user. The user id goes in through
    app.user_id for this transaction only; the function itself picks the new company id."""
    import psycopg

    try:
        with conn.transaction():
            conn.execute("select set_config('app.user_id', %s, true)", (user_id,))
            return conn.execute("select create_company(%s)", (name,)).fetchone()[0]
    except psycopg.errors.UniqueViolation:
        raise AlreadyHasCompany from None
    except psycopg.errors.InvalidParameterValue as e:
        raise ValueError("Company name must be 1 to 120 characters.") from e


def db_company_creator(pool) -> CompanyCreator:
    def create(user_id: str, name: str) -> str:
        with pool.connection() as conn:
            return create_company_in_db(conn, user_id, name)
    return create


def resolve_caller(authorization: str | None, verifier: TokenVerifier | None, lookup: MembershipLookup | None,
                   demo_company: str) -> Caller:
    if authorization is None:
        return Caller(company_id=demo_company, user_id=None)
    if lookup is None:
        raise AuthError(401, "Sign-in is not enabled on this server.")
    user_id = signed_in_user(authorization, verifier)
    companies = lookup(user_id)
    if not companies:
        raise AuthError(403, "No company yet. Finish setting up your account.")
    # A user in several companies gets the first one for now; a company switcher will pass the wanted id,
    # checked against this same list.
    return Caller(company_id=companies[0], user_id=user_id)
