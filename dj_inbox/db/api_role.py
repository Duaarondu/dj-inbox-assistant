# dj_inbox/db/api_role.py
"""Give the API's database login (app_api, from migration 002) a password, and write its connection URL to .env.

    python -m dj_inbox.db.api_role            (run once; run again to rotate the password)

- Connects with your OWNER DATABASE_URL (from .env) to run ALTER ROLE.
- The new password is random (32 bytes) and is sent to Postgres already hashed (SCRAM-SHA-256), so the plain
  password never appears in Postgres' statement logs or stats.
- Writes DATABASE_API_URL=... into .env (git-ignored). Nothing secret is printed.
- Supabase's pooler wants the user as "<role>.<project-ref>", taken from your owner URL ("postgres.<ref>").

Then: test the API's searches as app_api, and only after they pass, put DATABASE_API_URL into Secret Manager.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

from psycopg import sql

ROLE = "app_api"
ITERATIONS = 4096


def scram_sha256(password: str, salt: bytes | None = None, iterations: int = ITERATIONS) -> str:
    """The verifier Postgres stores for SCRAM-SHA-256 (RFC 5802/7677), so we can send it instead of the password."""
    salt = salt or os.urandom(16)
    salted = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    client_key = hmac.new(salted, b"Client Key", hashlib.sha256).digest()
    stored_key = hashlib.sha256(client_key).digest()
    server_key = hmac.new(salted, b"Server Key", hashlib.sha256).digest()

    def b64(b: bytes) -> str:
        return base64.b64encode(b).decode("ascii")

    return f"SCRAM-SHA-256${iterations}:{b64(salt)}${b64(stored_key)}:{b64(server_key)}"


def api_url(owner_url: str, password: str, role: str = ROLE) -> str:
    """Same host, port and database as the owner URL; user and password swapped for app_api's."""
    parts = urlsplit(owner_url)
    owner_user = parts.username or ""
    user = f"{role}.{owner_user.split('.', 1)[1]}" if "." in owner_user else role     # pooler: role.<project-ref>
    host = parts.hostname or ""
    netloc = f"{quote(user, safe='')}:{quote(password, safe='')}@{host}" + (f":{parts.port}" if parts.port else "")
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def write_env_var(env_path: Path, name: str, value: str) -> None:
    """Set NAME=value in .env, replacing an existing line; never prints the value."""
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    lines = [line for line in lines if not re.match(rf"^{re.escape(name)}=", line)]
    lines.append(f"{name}={value}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    from dj_inbox.db import connect, database_url
    from dj_inbox.gcp import ROOT

    owner_url = database_url()
    password = secrets.token_urlsafe(32)
    with connect(owner_url, vectors=False) as conn:
        exists = conn.execute("select 1 from pg_roles where rolname = %s", (ROLE,)).fetchone()
        if not exists:
            raise SystemExit("app_api doesn't exist yet: run `python -m dj_inbox.db.migrate` first.")
        conn.execute(sql.SQL("alter role {} with login password {}").format(
            sql.Identifier(ROLE), sql.Literal(scram_sha256(password))))
    write_env_var(ROOT / ".env", "DATABASE_API_URL", api_url(owner_url, password))
    print(f"{ROLE}: login enabled with a new random password; DATABASE_API_URL written to .env (not printed).")


if __name__ == "__main__":
    main()
