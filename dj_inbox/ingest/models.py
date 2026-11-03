# dj_inbox/ingest/models.py
"""Data shapes for an imported inbox. These map 1:1 onto the `threads` and `messages` tables later."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

SenderRole = Literal["client", "company"]


class CompanyConfig(BaseModel):
    """What the reader needs to know about the company whose inbox this is."""
    domains: list[str]                                   # e.g. ["northshoresound.example"]
    form_senders: list[str] = Field(default_factory=lambda: ["noreply", "no-reply", "forms"])
    
    def is_form(self, address: str) -> bool:
        """True for the company's own website-form address, e.g. noreply@northshoresound.example."""
        local, _, domain = address.lower().partition("@")
        return domain in {d.lower() for d in self.domains} and local in self.form_senders

    def role_of(self, address: str) -> SenderRole:
        local, _, domain = address.lower().partition("@")
        if domain in {d.lower() for d in self.domains} and local not in self.form_senders:
            return "company"
        return "client"  # clients, planners, venues, and the website form (it carries a client's inquiry)


class Contacts(BaseModel):
    """Contact details found in a message body before redaction. Feeds the lead record without a model call."""
    emails: list[str] = Field(default_factory=list)
    phones: list[str] = Field(default_factory=list)


class Email(BaseModel):
    message_id: str
    thread_id: str
    sent_at: datetime
    sender_role: SenderRole
    sender_name: str
    sender_email: str
    subject: str
    body_raw: str          # as received; shown to the owner, never sent to a model
    body_clean: str        # quoted history + signatures removed, emails/phones replaced by [EMAIL]/[PHONE]
    contacts: Contacts


class EmailThread(BaseModel):
    thread_id: str
    subject: str
    messages: list[Email]

    @property
    def last(self) -> Email:
        return self.messages[-1]

    def clean_text(self) -> str:
        """The whole thread as a model should read it: oldest first, one header line per message."""
        return "\n\n".join(
            f"[{m.sent_at:%Y-%m-%d %H:%M}] {m.sender_role.upper()} ({m.sender_name}):\n{m.body_clean}"
            for m in self.messages
        )
