# dj_inbox/ingest/__main__.py
"""Import a Gmail export.

    python -m dj_inbox.ingest synthetic_inbox/data/inbox.mbox --company-domain northshoresound.example

Writes data/processed/threads.jsonl (one cleaned thread per line) and prints what cleaning removed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .models import CompanyConfig
from .read_mbox import read_mbox


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mbox", type=Path)
    ap.add_argument("--company-domain", action="append", required=True, help="repeat for several domains")
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    args = ap.parse_args()

    threads = read_mbox(args.mbox, CompanyConfig(domains=args.company_domain))
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "threads.jsonl"
    path.write_text("".join(t.model_dump_json() + "\n" for t in threads), encoding="utf-8")

    msgs = [m for t in threads for m in t.messages]
    raw_chars = sum(len(m.body_raw) for m in msgs)
    clean_chars = sum(len(m.body_clean) for m in msgs)
    print(f"{len(threads)} threads, {len(msgs)} messages -> {path}")
    print(f"text kept after cleaning: {clean_chars:,} of {raw_chars:,} characters "
          f"({1 - clean_chars / raw_chars:.0%} removed: quoted history, signatures)")
    print(f"messages with contact details found: {sum(bool(m.contacts.emails or m.contacts.phones) for m in msgs)}")


if __name__ == "__main__":
    main()
