# tests/test_clean.py
"""Edge cases for cleaning and redaction. Every bug found in the cleaner gets a case here."""
from dj_inbox.ingest.clean import clean_body, find_contacts, redact, strip_quoted, strip_signature

REPLY = """Thanks for the quick reply! One question: How early do you get there?

Best,
Jenna

On Tue, Oct 14, 2025 at 8:34 AM Mike Russo <mike@northshoresound.example> wrote:
> Hi Jenna,
>
> - Signature package (5 hours): $1,895
> Total: $1,895"""

COMPANY = """Hi Jenna,

For 5 hours I'd recommend our Signature package.

- Signature package (5 hours): $1,895
- Uplighting: $295

Total: $2,190

We hold dates with a 30% deposit. Happy to hop on a call if that's easier.

Mike Russo
Northshore Sound Co.
(516) 555-0100
northshoresound.example"""


def test_quoted_history_is_removed():
    out = strip_quoted(REPLY)
    assert "wrote:" not in out and "$1,895" not in out
    assert out.startswith("Thanks for the quick reply")


def test_wrapped_gmail_header_is_removed():
    body = "Sounds good!\n\nOn Tue, Oct 14, 2025 at 8:34 AM Jenna Rossi <\njenna@example.com> wrote:\n> old text"
    assert strip_quoted(body) == "Sounds good!"


def test_outlook_original_message_is_removed():
    body = "We'd love to book.\n\n-----Original Message-----\nFrom: Mike\nSent: Monday\nOld quote $1,895"
    assert strip_quoted(body) == "We'd love to book."


def test_forwarded_message_is_kept():
    body = ("Hi Mike, forwarding this couple.\n\n---------- Forwarded message ---------\nFrom: Olivia\n\n"
            "We're getting married August 23, 2026 at Harbor View Manor (190 guests).")
    out = strip_quoted(body)
    assert "190 guests" in out and "Harbor View Manor" in out


def test_company_signature_is_removed_but_prices_stay():
    out = strip_signature(COMPANY)
    assert "Northshore Sound Co." not in out and "555-0100" not in out
    assert "Total: $2,190" in out and "Uplighting: $295" in out
    assert out.endswith("Happy to hop on a call if that's easier.")


def test_client_signoffs_are_removed():
    assert strip_signature("Deposit sent!\n\nThanks!\nJenna") == "Deposit sent!"
    assert strip_signature("Deposit sent!\n\nJenna Rossi\n(516) 555-0165") == "Deposit sent!"
    assert strip_signature("Deposit sent!\n\nSent from my iPhone") == "Deposit sent!"
    assert strip_signature("Deposit sent!\n\nBest,\nJenna\n\nSent from my iPhone") == "Deposit sent!"


def test_never_returns_empty():
    assert strip_signature("Sent from my iPhone") == "Sent from my iPhone"


def test_short_question_with_phone_is_not_a_signature():
    body = "Hi Mike,\n\nCan you call me tomorrow at (516) 555-0165 to talk about the date?"
    assert "talk about the date" in strip_signature(body)


def test_redact_phone_formats():
    for phone in ["(516) 555-0165", "516-555-0165", "516.555.0165", "+1 516 555 0165"]:
        assert redact(f"call {phone} please") == "call [PHONE] please", phone


def test_redact_leaves_prices_dates_and_counts_alone():
    text = "Total $1,295 for 05/23/2026 (2026-10-28), 150 guests, 1,000 threads, $2M insurance, 12/31"
    assert redact(text) == text


def test_redact_emails():
    assert redact("write jenna.rossi12@mail.example or me") == "write [EMAIL] or me"


def test_contacts_found_before_signature_is_stripped():
    clean, contacts = clean_body("We'd love to book!\n\nJenna Rossi\n(516) 555-0165")
    assert clean == "We'd love to book!"
    assert contacts.phones == ["5165550165"]


def test_contacts_ignore_quoted_history():
    _, contacts = clean_body(REPLY)
    assert contacts.emails == []  # mike@ appears only in the quoted header
