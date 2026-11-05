# dj_inbox/rag/chunk.py
"""Step 1 of RAG: split the inbox into pieces that can be searched and cited.

Strategy v1: ONE CHUNK PER EMAIL. An email is a natural unit, it's what the owner clicks to read,
and a quote's price lines never get split across chunks.

The header line matters: an email that just says "Sounds good, let's book!" means nothing on its own.
Prefixing the thread's subject, the sender and the date gives the embedding context about WHICH event
it's about. (That idea, grown up, is "contextual retrieval": a later rung of the ladder.)

tests/test_chunk.py is the spec.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from dj_inbox.ingest import EmailThread


class Chunk(BaseModel):
    chunk_id: str          # "<thread_id>-<message number>", e.g. "t055-0", "t055-1"
    thread_id: str
    message_id: str
    sent_at: datetime
    text: str              # header line + "\n" + the email's cleaned body


def chunk_thread(thread: EmailThread) -> list[Chunk]:
    """One Chunk per message, in order. Each chunk's text is a header line, a newline, then body_clean.

    Header format:  "<subject> | <ROLE> <sender name> | <YYYY-MM-DD>"
    Example:        "Fwd: DJ? (Brennan wedding) | CLIENT Gina Romano | 2025-11-21"
    ROLE is the message's sender_role in capitals (CLIENT or COMPANY).
    """
    chunks = []
    for i, m in enumerate(thread.messages):          # i counts 0, 1, 2... ; m is one Email
        header = f"{thread.subject} | {m.sender_role.upper()} {m.sender_name} | {m.sent_at:%Y-%m-%d}"
        chunks.append(Chunk(
            chunk_id=f"{thread.thread_id}-{i}",
            thread_id=thread.thread_id,
            message_id=m.message_id,
            sent_at=m.sent_at,
            text=header + "\n" + m.body_clean,       # the cleaned body: no quoted history, contacts redacted
        ))
    return chunks


def chunk_inbox(threads: list[EmailThread]) -> list[Chunk]:
    return [chunk for t in threads for chunk in chunk_thread(t)]
