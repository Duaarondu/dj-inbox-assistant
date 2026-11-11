# evals/make_questions_v2.py
"""Eval set v2: questions aimed at what v1 couldn't see (v1 hit 50/50 on Nov 9 and stopped telling good from better).

    python -m evals.make_questions_v2        -> writes evals/questions_v2.jsonl

v1's blind spots: 40/50 questions named a client with a unique name, so exact name lookup did the work, and
nothing asked for a sum, a list or a calendar. v2's types (owner questions from Ultimate Sound DJs, Nov 11):

  sum             revenue = sum of quoted prices of BOOKED events dated in a year (definition agreed Nov 11)
  list            how many booked events include an add-on; which clients match a filter
  calendar        events in a month; who is booked on a date
  not_in_emails   planning details inquiry emails never contain (first dance, ceremony time, do-not-play,
                  grand entrance cue, flower walls): the right answer says the emails don't say
  no_name         "who is booked at <venue> on <date>?": search by details, not by name
  described       "who is planning a <event> for about <n> guests in <month>?"
  misspelled      a real client's name with a typo in the last name: exact lookup misses on purpose
  repeat_client   a name with two events (two threads, or a 2nd event in one thread): which quote?
  compare         two quotes, which is higher

Facts come from the synthetic ground truth; quotes for retrieval come from the CLEANED email text.
tests/test_eval_questions_v2.py checks every answer key against the data. Same seed -> same questions.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from datetime import date
from pathlib import Path

from dj_inbox.ingest import read_mbox
from evals.make_questions import COMPANY, DATA, MONTHS, line_with, long_date, money, price_line

OUT = Path(__file__).resolve().parent / "questions_v2.jsonl"
TODAY = date(2026, 10, 28)          # the synthetic inbox's "now": "this year" = 2026
PER_TYPE = {"no_name": 5, "described": 4, "misspelled": 5, "repeat_client": 4, "compare": 4}


def deposit_bases(thread) -> set[int]:
    """The quote totals our deposit emails were computed on: "the deposit is $525 (30% of $1,750)" -> {1750}."""
    return {int(base.replace(",", "")) for base in
            re.findall(r"deposit is \$[\d,]+ \(30% of \$([\d,]+)\)", thread.clean_text())}


def is_booked(lead: dict, threads: dict, main_price: dict) -> bool:
    """Business rule (Aaron, Nov 11): a second event (rehearsal dinner, engagement party) is a SEPARATE event,
    booked only when a deposit covers it. A deposit computed on the main event's quote alone books the main
    event only. (The synthetic ground truth marked 4 such second events "booked"; the emails don't support it.)"""
    if lead["status"] != "booked":
        return False
    if lead["lead_index"] == 0:
        return True
    bases = deposit_bases(threads[lead["thread_id"]])
    return lead["quoted_price"] in bases or main_price[lead["thread_id"]] + lead["quoted_price"] in bases


def misspell(last: str, rng: random.Random) -> str:
    """A realistic typo in a last name: double a letter, or swap two inner letters."""
    i = rng.randrange(1, len(last) - 1)
    if rng.random() < 0.5 or last[i] == last[i + 1]:
        return last[: i + 1] + last[i] + last[i + 1:]                     # Santos -> Santtos
    return last[:i] + last[i + 1] + last[i] + last[i + 2:]               # Santos -> Satnos


def build() -> list[dict]:
    rng = random.Random(22)
    threads = {t.thread_id: t for t in read_mbox(DATA / "inbox.mbox", COMPANY)}
    with (DATA / "ground_truth_leads.jsonl").open(encoding="utf-8") as fh:
        leads = [json.loads(row) for row in fh]
    main = [lead for lead in leads if lead["lead_index"] == 0]
    main_price = {lead["thread_id"]: lead["quoted_price"] or 0 for lead in main}
    booked = [lead for lead in leads if is_booked(lead, threads, main_price)]   # every booked EVENT, by the rule
    year = str(TODAY.year)
    questions: list[dict] = []

    def add(kind: str, question: str, answer: str, facts: list[dict], evidence: list[dict] | None = None,
            defer: bool = False, must_not: list[str] | None = None) -> None:
        n = sum(q["type"] == kind for q in questions) + 1
        questions.append({"id": f"{kind}-{n:02d}", "type": kind, "question": question, "expected_answer": answer,
                          "expected_facts": facts, "gold_evidence": evidence or [], "should_defer": defer,
                          "must_not_contain": must_not or []})

    def ev(lead: dict, quote: str | None) -> list[dict] | None:
        return [{"thread_id": lead["thread_id"], "quote": quote}] if quote else None

    # ---- sum: revenue (answered from the lead table; no quote) ----
    def revenue(pred) -> tuple[int, int]:
        rows = [lead for lead in booked if pred(lead)]
        return sum(lead["quoted_price"] or 0 for lead in rows), len(rows)

    for question, pred in [
        ("How much revenue do we have from events booked for this year?",
         lambda lead: (lead["event_date"] or "")[:4] == year),
        ("How much revenue is booked for weddings this year?",
         lambda lead: (lead["event_date"] or "")[:4] == year and lead["event_type"] == "wedding"),
        ("How much revenue did we book for events in 2025?", lambda lead: (lead["event_date"] or "")[:4] == "2025"),
        ("What's the total of our quotes for booked sweet 16s this year?",
         lambda lead: (lead["event_date"] or "")[:4] == year and lead["event_type"] == "sweet16"),
    ]:
        total, n = revenue(pred)
        add("sum", question, f"{money(total)} across {n} booked events", [{"kind": "money", "value": total}])

    # ---- list: add-ons on booked events (all booked events, like revenue) ----
    for addon, phrase in [("photo booth", "a photo booth"), ("dancing on clouds", "dancing on clouds"),
                          ("cold sparks", "cold sparks")]:
        n = sum(addon in lead["addons_quoted"] for lead in booked)
        add("list", f"How many of our booked events include {phrase}?", str(n), [{"kind": "number", "value": n}])
    # Which clients: a filter with exactly 2 or 3 matches, so the answer is a short list of names
    combos = Counter((lead["event_type"], a) for lead in booked if (lead["event_date"] or "")[:4] == year
                     for a in lead["addons_quoted"])
    for (etype, addon), n in sorted(combos.items()):
        names = sorted({lead["client_name"] for lead in booked if lead["event_type"] == etype
                        and (lead["event_date"] or "")[:4] == year and addon in lead["addons_quoted"]})
        if 2 <= n <= 3 and len(names) == n and etype in ("sweet16", "birthday", "corporate", "mitzvah"):
            label = {"sweet16": "sweet 16s", "mitzvah": "mitzvahs"}.get(etype, f"{etype} events")
            add("list", f"Which clients booked {label} with {addon} this year?", ", ".join(names),
                [{"kind": "text", "value": name} for name in names])
            break

    # ---- calendar ----
    by_month = Counter(lead["event_date"][:7] for lead in booked if lead["event_date"])
    for month in [f"{year}-06", f"{year}-11"]:
        m = int(month[5:])
        add("calendar", f"How many events do we have booked in {MONTHS[m - 1]} {year}?", str(by_month[month]),
            [{"kind": "number", "value": by_month[month]}])
    by_date = Counter(lead["event_date"] for lead in booked if lead["event_date"])
    one_event_days = sorted(d for d, c in by_date.items() if c == 1 and d[:4] == year)
    for d in rng.sample(one_event_days, 2):
        lead = next(lead for lead in booked if lead["event_date"] == d)
        add("calendar", f"Do we have anything booked on {long_date(d)}? Who is it for?",
            f"Yes: {lead['client_name']}'s {lead['event_label']}", [{"kind": "text", "value": lead["client_name"]}],
            ev(lead, line_with(threads[lead["thread_id"]], long_date(d), "company")))

    # ---- not in the emails (planning details) ----
    alltext = " ".join(t.subject + " " + t.clean_text() for t in threads.values()).lower()
    names = Counter(lead["client_name"] for lead in main)
    weddings = [lead for lead in booked if lead["event_type"] == "wedding" and names[lead["client_name"]] == 1
                and lead["lead_index"] == 0]
    w = rng.sample(weddings, 4)
    for question, absent in [
        ("Which of our photo booth jobs mention a flower wall?", "flower wall"),
        (f"What song does {w[0]['client_name']} want for their first dance?", "first dance"),
        (f"What time does the ceremony start for {w[1]['client_name']}'s wedding?", " pm"),
        (f"Which song should start the grand entrance for {w[2]['client_name']}'s wedding, and at what second?",
         "grand entrance"),
    ]:
        assert absent not in alltext or absent == " pm", absent
        add("not_in_emails", question, "The emails don't say", [], defer=True)
    t3 = threads[w[3]["thread_id"]].clean_text().lower()
    if "do-not-play" not in t3:
        add("not_in_emails", f"What's on {w[3]['client_name']}'s do-not-play list?", "The emails don't say", [],
            defer=True)

    # ---- retrieval traps on unique-name main leads (v1 skipped none of these types) ----
    usable = [lead for lead in main if names[lead["client_name"]] == 1 and "vague" not in lead["tags"]]
    rng.shuffle(usable)
    used: set[str] = set()

    def take(kind: str, pick, make) -> None:
        for lead in usable:
            if sum(q["type"] == kind for q in questions) == PER_TYPE[kind]:
                return
            if lead["thread_id"] in used or not pick(lead):
                continue
            if make(lead, threads[lead["thread_id"]]):
                used.add(lead["thread_id"])

    venue_date = Counter((lead["venue"], lead["event_date"]) for lead in leads)

    def q_no_name(lead, t) -> bool:
        quote = line_with(t, lead["venue"], "client", last=False)
        if not quote:
            return False
        add("no_name", f"Who did we book at {lead['venue']} on {long_date(lead['event_date'])}?",
            f"{lead['client_name']} ({lead['event_label']})", [{"kind": "text", "value": lead["client_name"]}],
            ev(lead, quote))
        return True

    take("no_name", lambda lead: lead["status"] == "booked" and lead["venue"] and lead["event_date"]
         and lead["venue"] != "Private residence" and venue_date[(lead["venue"], lead["event_date"])] == 1,
         q_no_name)

    described = Counter((lead["event_type"], lead["guest_count"], (lead["event_date"] or "")[:7]) for lead in leads)

    def q_described(lead, t) -> bool:
        quote = line_with(t, str(lead["guest_count"]), "client", last=False)
        if not quote:
            return False
        d = date.fromisoformat(lead["event_date"])
        add("described", f"Who's planning a {lead['event_label']} for about {lead['guest_count']} guests in "
            f"{MONTHS[d.month - 1]} {d.year}?", lead["client_name"], [{"kind": "text", "value": lead["client_name"]}],
            ev(lead, quote))
        return True

    take("described", lambda lead: lead["guest_count"] and lead["event_date"] and lead["event_type"] != "wedding"
         and described[(lead["event_type"], lead["guest_count"], lead["event_date"][:7])] == 1, q_described)

    real_words = {w.lower() for lead in leads for w in lead["client_name"].split()}

    def q_misspelled(lead, t) -> bool:
        first, last = lead["client_name"].split()[0], lead["client_name"].split()[-1]
        typo = misspell(last, rng)
        quote = price_line(t, lead["quoted_price"])
        if typo.lower() in real_words or not quote:
            return False
        add("misspelled", f"What did we quote {first} {typo} for their {lead['event_label']}?",
            f"{money(lead['quoted_price'])} (the client is {lead['client_name']})",
            [{"kind": "money", "value": lead["quoted_price"]}], ev(lead, quote))
        return True

    take("misspelled", lambda lead: lead["quoted_price"] and len(lead["client_name"].split()[-1]) >= 5
         and "quoted_on_2025_sheet" not in lead["tags"] and "date_changed" not in lead["tags"], q_misspelled)

    def q_compare(lead, t) -> bool:
        other = next((o for o in usable if o["thread_id"] not in used | {lead["thread_id"]}
                      and o["quoted_price"] and o["quoted_price"] != lead["quoted_price"]
                      and o["event_type"] == lead["event_type"]
                      and "quoted_on_2025_sheet" not in o["tags"] and "date_changed" not in o["tags"]), None)
        qa, qb = price_line(t, lead["quoted_price"]), other and price_line(threads[other["thread_id"]],
                                                                            other["quoted_price"])
        if not other or not qa or not qb:
            return False
        hi = max((lead, other), key=lambda x: x["quoted_price"])
        add("compare", f"Who got the higher quote for their {lead['event_label']}: {lead['client_name']} or "
            f"{other['client_name']}?", f"{hi['client_name']} ({money(hi['quoted_price'])} vs "
            f"{money(min(lead['quoted_price'], other['quoted_price']))})",
            [{"kind": "money", "value": lead["quoted_price"]}, {"kind": "money", "value": other["quoted_price"]}],
            ev(lead, qa) + ev(other, qb))
        used.add(other["thread_id"])
        return True

    take("compare", lambda lead: lead["quoted_price"] and "quoted_on_2025_sheet" not in lead["tags"]
         and "date_changed" not in lead["tags"], q_compare)

    # ---- repeat clients: the same name on two events ----
    repeat: list[tuple[dict, str]] = []
    for lead in leads:          # a 2nd event in the same thread (engagement party, rehearsal dinner)
        if lead["lead_index"] == 1 and lead["quoted_price"] and names[lead["client_name"]] == 1:
            repeat.append((lead, f"their {lead['event_label']}"))
    rng.shuffle(repeat)
    for lead, what in repeat:
        if sum(q["type"] == "repeat_client" for q in questions) == PER_TYPE["repeat_client"]:
            break
        quote = price_line(threads[lead["thread_id"]], lead["quoted_price"])
        main_price = next(m["quoted_price"] for m in main if m["thread_id"] == lead["thread_id"])
        if quote and main_price != lead["quoted_price"]:
            add("repeat_client", f"What did we quote {lead['client_name']} for {what}?",
                f"{money(lead['quoted_price'])} (not the {money(main_price)} main event)",
                [{"kind": "money", "value": lead["quoted_price"]}], ev(lead, quote),
                must_not=[money(main_price)])
    return questions


def main() -> None:
    questions = build()
    OUT.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in questions), encoding="utf-8")
    print(f"{len(questions)} questions -> {OUT}")
    print(dict(Counter(q["type"] for q in questions)))


if __name__ == "__main__":
    main()
