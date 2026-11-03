import email
import mailbox
from email import policy

from dj_inbox.ingest.models import CompanyConfig
from dj_inbox.ingest.read_mbox import _parse

company = CompanyConfig(domains=["northshoresound.example"])
box = mailbox.mbox("synthetic_inbox/data/inbox.mbox",
                   factory=lambda f: email.message_from_binary_file(f, policy=policy.default))
for msg in box:
    if "06/20/2026" in str(msg["Subject"]) and "Re:" not in str(msg["Subject"]):
        r = _parse(msg, company)
        print(r["sent_at"].date(), r["role"], "|", r["norm_subject"], "|", r["client_address"], "|", r["refs"])
