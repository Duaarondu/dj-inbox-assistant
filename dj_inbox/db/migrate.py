# dj_inbox/db/migrate.py
"""Apply db/migrations/*.sql in name order, each exactly once.

    python -m dj_inbox.db.migrate

Each migration runs in its own transaction with its bookkeeping row, so a failed migration leaves nothing behind.
"""
from __future__ import annotations

import psycopg

from dj_inbox.gcp import ROOT

MIGRATIONS = ROOT / "db" / "migrations"


def migrate(conn: psycopg.Connection) -> list[str]:
    """Returns the names of the migrations applied this time (empty when the database is up to date)."""
    conn.execute("""
        create table if not exists schema_migrations (
            name       text primary key,
            applied_at timestamptz not null default now()
        );
        alter table schema_migrations enable row level security;
    """)
    done = {name for (name,) in conn.execute("select name from schema_migrations")}
    applied = []
    for path in sorted(MIGRATIONS.glob("*.sql")):
        if path.name in done:
            continue
        with conn.transaction():
            conn.execute(path.read_text(encoding="utf-8"))
            conn.execute("insert into schema_migrations (name) values (%s)", (path.name,))
        applied.append(path.name)
    return applied


def main() -> None:
    from dj_inbox.db import connect

    with connect(vectors=False) as conn:          # the vector type may not exist yet
        applied = migrate(conn)
    print("applied: " + (", ".join(applied) if applied else "nothing (up to date)"))


if __name__ == "__main__":
    main()
