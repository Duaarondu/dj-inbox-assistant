# Synthetic inbox: Northshore Sound Co. (fictional)

A fake DJ company's inquiry inbox, for building and testing before real emails arrive.
Everything is made up: company, clients, venues, `.example` email domains, 555-01xx phone numbers.

```bash
python generate_inbox.py --threads 200 --seed 7   # writes data/
python -m pytest -q                               # checks the labels agree with the emails
```

Same seed, same inbox, so eval labels stay stable. Needs only the Python standard library (pytest for tests).

## What's in `data/`

| File | What it is | Use it for |
| --- | --- | --- |
| `inbox.mbox` | 200 threads, ~940 emails, Gmail-export format with `X-GM-THRID` thread ids | The import step, exactly as a real Gmail export would arrive |
| `threads.jsonl` | Same threads as JSON | Reading threads by eye |
| `ground_truth_leads.jsonl` | The correct lead record for every thread (~209 leads) | Extraction eval labels, field by field |
| `venue_leads.csv` | Monthly lists from the 3 house-DJ venues, same columns every month | Structured leads parsed with code, no model; `matching_thread_id` links some to email threads |
| `company_kb.md` | 2026 price sheet and policies | What answers and quotes should be grounded in |
| `price_sheet_2025.md` | The superseded 2025 sheet | Stale-price conflicts: 2025 threads quote these prices |

## Statuses (as of Oct 28, 2026)

- `booked`: client paid the deposit
- `lost`: client declined
- `awaiting_reply`: the client wrote last; the company owes a reply
- `waiting_on_client`: the company wrote last, under 21 days ago
- `cold`: the company wrote last, 21+ days ago

## Hard cases built in (see each lead's `tags`)

Quoted reply history, prices from the old 2025 sheet, date changes mid-thread, package upgrades and re-quotes,
two events in one thread, dates without a year ("June 14th"), website-form inquiries, planner-forwarded emails,
venue referrals, holiday surcharges, travel fees, vague "how much?" inquiries, informal lowercase messages,
client phone numbers in signatures, and a few prompt-injection attempts ("ignore your previous instructions...").

## Known limits

Generated from templates, so it's cleaner and more repetitive than a real inbox: no typos beyond the informal
style, no attachments, no Instagram or phone leads, no threads that drift off topic. Use it to build the pipeline
and catch bugs; report real numbers only from real emails.
