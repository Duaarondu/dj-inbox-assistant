"use client";
// web/components/InboxView.tsx: ask the inbox a question; see the answer, the exact emails it came from,
// and the lead tracker built from the same data.
//   no getToken  -> the public demo (Northshore Sound Co., synthetic inbox)          used by app/page.tsx
//   getToken     -> the signed-in user's OWN company (the API picks it from the token) used by app/inbox/page.tsx
//                   + the tracker re-checks every 15 s, so emails that arrive (automatic updates) show up
// Email text is untrusted: it is always rendered as plain React text, never as HTML.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import Link from "next/link";

import { ask, AskResponse, getLeads, getThread, Lead, Thread } from "@/lib/api";

const DEMO_EXAMPLES = [
  "What did we quote Omar Santos?",
  "How many leads have gone cold?",
  "How many weddings have we booked?",
  "Has anyone asked us about karaoke?",
];
const COMPANY_EXAMPLES = [
  "Which leads are awaiting our reply?",
  "How many leads have gone cold?",
  "What's our booked revenue this year?",
  "Who asked about uplighting?",
];
const POLL_MS = 15_000;

type TokenFn = () => Promise<string | undefined>;

const STATUS_LABEL: Record<string, string> = {
  booked: "Booked",
  lost: "Lost",
  awaiting_reply: "Awaiting our reply",
  waiting_on_client: "Waiting on client",
  cold: "Cold",
};

const money = (n: number | null) => (n == null ? "–" : `$${n.toLocaleString("en-US")}`);
const label = (s: string) => s.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
const shortDate = (iso: string) =>
  new Date(iso).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });

export default function InboxView({ getToken }: { getToken?: () => Promise<string | null> }) {
  const signedIn = Boolean(getToken);
  // The current access token for each call (supabase-js refreshes it); none in the demo. Signed out -> /signin.
  const token: TokenFn = useCallback(async () => {
    if (!getToken) return undefined;
    const t = await getToken();
    if (!t) {
      window.location.replace("/signin");
      throw new Error("Please sign in again.");
    }
    return t;
  }, [getToken]);

  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const [result, setResult] = useState<AskResponse | null>(null);
  const [askError, setAskError] = useState<string | null>(null);

  const [leads, setLeads] = useState<Lead[]>([]);
  const [leadsError, setLeadsError] = useState<string | null>(null);
  const [status, setStatus] = useState("");
  const [eventType, setEventType] = useState("");
  const [onlyThreads, setOnlyThreads] = useState<string[] | null>(null);
  const [fresh, setFresh] = useState<Set<string>>(new Set());       // rows that appeared while the page was open
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null);
  const known = useRef<Set<string> | null>(null);

  const [openThread, setOpenThread] = useState<{ id: string; highlight?: string } | null>(null);
  const trackerRef = useRef<HTMLElement>(null);

  const loadLeads = useCallback(async () => {
    try {
      const d = await getLeads(await token());
      const ids = d.leads.map((l) => `${l.thread_id}-${l.lead_index}`);
      const before = known.current;
      if (before) {
        const added = ids.filter((id) => !before.has(id));
        if (added.length) setFresh((f) => new Set([...f, ...added]));
      }
      known.current = new Set(ids);
      setLeads(d.leads);
      setLeadsError(null);
      setUpdatedAt(new Date());
    } catch (e) {
      setLeadsError((e as Error).message);
    }
  }, [token]);

  useEffect(() => {
    loadLeads();
  }, [loadLeads]);

  // Signed in: new emails arrive on their own (Gmail push -> worker), so re-check while the tab is visible.
  useEffect(() => {
    if (!signedIn) return;
    const timer = setInterval(() => {
      if (document.visibilityState === "visible") loadLeads();
    }, POLL_MS);
    return () => clearInterval(timer);
  }, [signedIn, loadLeads]);

  async function submit(q: string) {
    const text = q.trim();
    if (!text || asking) return;
    setQuestion(text);
    setAsking(true);
    setAskError(null);
    setResult(null);
    try {
      setResult(await ask(text, await token()));
    } catch (e) {
      setAskError((e as Error).message);
    } finally {
      setAsking(false);
    }
  }

  const eventTypes = useMemo(() => [...new Set(leads.map((l) => l.event_type))].sort(), [leads]);
  const shown = leads.filter(
    (l) =>
      (!status || l.status === status) &&
      (!eventType || l.event_type === eventType) &&
      (!onlyThreads || onlyThreads.includes(l.thread_id)),
  );

  function showMatched(threads: string[]) {
    setStatus("");
    setEventType("");
    setOnlyThreads(threads);
    trackerRef.current?.scrollIntoView({ behavior: "smooth" });
  }

  return (
    <main className="page">
      <header className="header">
        <div className="headRow">
          <h1>{signedIn ? "Your inbox" : "Ask your inbox"}</h1>
          {signedIn ? <Link href="/setup">Settings</Link> : <Link href="/signin">Sign in to use your own inbox</Link>}
        </div>
        {signedIn ? (
          <p className="sub">
            Your lead tracker, built from your Gmail. New emails show up here on their own within about 30 seconds.
            Every answer links to the emails it came from.
          </p>
        ) : (
          <p className="sub">
            Demo for <strong>Northshore Sound Co.</strong>, a fictional DJ company with a synthetic inbox of 200
            email threads. Every answer links to the emails it came from.
          </p>
        )}
      </header>

      <section className="card">
        <form
          className="askForm"
          onSubmit={(e) => {
            e.preventDefault();
            submit(question);
          }}
        >
          <input
            aria-label="Your question"
            value={question}
            maxLength={500}
            placeholder="e.g. What did we quote the Brennans?"
            onChange={(e) => setQuestion(e.target.value)}
          />
          <button type="submit" disabled={asking || !question.trim()}>
            {asking ? "Thinking…" : "Ask"}
          </button>
        </form>
        <div className="chips">
          {(signedIn ? COMPANY_EXAMPLES : DEMO_EXAMPLES).map((q) => (
            <button key={q} className="chip" onClick={() => submit(q)} disabled={asking}>
              {q}
            </button>
          ))}
        </div>

        {asking && <p className="muted">Searching the inbox… (the first question after a quiet spell can take ~15s while the server wakes up)</p>}
        {askError && <p className="error">{askError}</p>}

        {result && (
          <div className="answer">
            <p className="answerText">{result.answer}</p>

            {result.sources.length > 0 && (
              <div>
                <h3>Sources</h3>
                <ul className="sources">
                  {result.sources.map((s) => (
                    <li key={s.email_id}>
                      <button className="link" onClick={() => setOpenThread({ id: s.thread_id, highlight: s.email_id })}>
                        {s.sender} · {shortDate(s.sent_at)} · “{s.subject}”
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {result.matched_threads.length > 0 && (
              <button className="link" onClick={() => showMatched(result.matched_threads)}>
                Show these {result.matched_threads.length} leads in the tracker ↓
              </button>
            )}

            <p className="meta">
              {result.tools.length ? `Tools: ${result.tools.join(", ")}` : "No tools"} · {result.latency_s}s ·{" "}
              {result.tokens_in.toLocaleString()} tokens in / {result.tokens_out} out · ${result.cost_usd.toFixed(4)} ·{" "}
              {result.model}
            </p>
          </div>
        )}
      </section>

      <section className="card" ref={trackerRef}>
        <div className="trackerHead">
          <h2>Lead tracker</h2>
          <span className="muted">
            {shown.length} {shown.length === 1 ? "inquiry" : "inquiries"}
            {signedIn && updatedAt && ` · checked ${updatedAt.toLocaleTimeString()}`}
          </span>
        </div>
        <div className="filters">
          <select aria-label="Status" value={status} onChange={(e) => { setStatus(e.target.value); setOnlyThreads(null); }}>
            <option value="">All statuses</option>
            {Object.entries(STATUS_LABEL).map(([v, t]) => (
              <option key={v} value={v}>{t}</option>
            ))}
          </select>
          <select aria-label="Event type" value={eventType} onChange={(e) => { setEventType(e.target.value); setOnlyThreads(null); }}>
            <option value="">All event types</option>
            {eventTypes.map((t) => (
              <option key={t} value={t}>{label(t)}</option>
            ))}
          </select>
          {onlyThreads && (
            <button className="chip" onClick={() => setOnlyThreads(null)}>
              Showing the {onlyThreads.length} leads from the answer ✕
            </button>
          )}
        </div>

        {leadsError && <p className="error">{leadsError}</p>}
        <div className="tableWrap">
          <table>
            <thead>
              <tr>
                <th>Client</th><th>Event</th><th>Date</th><th>Venue</th><th>Guests</th><th>Quote</th><th>Status</th><th>Source</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((l) => (
                <tr key={`${l.thread_id}-${l.lead_index}`}
                    className={fresh.has(`${l.thread_id}-${l.lead_index}`) ? "new" : undefined} onClick={() => setOpenThread({ id: l.thread_id })} tabIndex={0}
                    onKeyDown={(e) => e.key === "Enter" && setOpenThread({ id: l.thread_id })}>
                  <td>{l.client ?? "–"}</td>
                  <td>{label(l.event_type)}</td>
                  <td>{l.event_date ?? "–"}</td>
                  <td>{l.venue ?? "–"}</td>
                  <td>{l.guest_count ?? "–"}</td>
                  <td>{l.package_quoted ? `${l.package_quoted} · ` : ""}{money(l.quoted_price)}</td>
                  <td><span className={`badge ${l.status}`}>{STATUS_LABEL[l.status] ?? l.status}</span></td>
                  <td>{l.source === "website_form" ? "Website form" : "Email"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <footer className="footer muted">
        {signedIn ? "Read-only access to your Gmail; spam, promotions, social mail and drafts are skipped. " : "Synthetic data only. "} Built with Gemini + Google ADK (tool routing over a lead table and RAG), FastAPI on
        Cloud Run. Statuses are as of Oct 28, 2026, the synthetic inbox&apos;s &quot;today&quot;.
      </footer>

      {openThread && (
        <ThreadPanel id={openThread.id} highlight={openThread.highlight} token={token} onClose={() => setOpenThread(null)} />
      )}
    </main>
  );
}

function ThreadPanel({ id, highlight, token, onClose }: {
  id: string; highlight?: string; token: TokenFn; onClose: () => void;
}) {
  const [thread, setThread] = useState<Thread | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    token()
      .then((t) => getThread(id, t))
      .then((t) => live && setThread(t))
      .catch((e: Error) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [id, token]);

  useEffect(() => {
    if (thread && highlight) document.getElementById(highlight)?.scrollIntoView({ block: "center" });
  }, [thread, highlight]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="overlay" onClick={onClose}>
      <aside className="panel" onClick={(e) => e.stopPropagation()} aria-label="Email thread">
        <div className="panelHead">
          <div>
            <h2>{thread?.subject ?? "Loading…"}</h2>
            {thread && <p className="muted">{thread.client} · {thread.emails.length} emails</p>}
          </div>
          <button className="close" onClick={onClose} aria-label="Close">✕</button>
        </div>
        {error && <p className="error">{error}</p>}
        {thread?.emails.map((m) => (
          <article key={m.email_id} id={m.email_id} className={`email ${m.email_id === highlight ? "cited" : ""}`}>
            <p className="emailHead">
              <strong>{m.sender}</strong> {m.sender_role === "company" ? "(us)" : ""} · {shortDate(m.sent_at)}
              {m.email_id === highlight && <span className="citedTag">cited</span>}
            </p>
            <p className="body">{m.body}</p>
          </article>
        ))}
      </aside>
    </div>
  );
}
