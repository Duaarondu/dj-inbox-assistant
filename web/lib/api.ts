// web/lib/api.ts: typed calls to the FastAPI service on Cloud Run.
// NEXT_PUBLIC_API_URL is baked in at build time (set it in Vercel's project settings).
export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8080").replace(/\/$/, "");

export type Source = { email_id: string; thread_id: string; subject: string; sent_at: string; sender: string };

export type AskResponse = {
  answer: string;
  sources: Source[];
  matched_threads: string[];
  tools: string[];
  model: string;
  latency_s: number;
  tokens_in: number;
  tokens_out: number;
  cost_usd: number;
};

export type Lead = {
  thread_id: string;
  lead_index: number;
  client: string | null;
  subject: string | null;
  event_type: string;
  event_date: string | null;
  venue: string | null;
  guest_count: number | null;
  package_quoted: string | null;
  quoted_price: number | null;
  status: string;
  source: string;
};

export type ThreadEmail = { email_id: string; sent_at: string; sender_role: string; sender: string; body: string };
export type Thread = { thread_id: string; subject: string; client: string; emails: ThreadEmail[] };

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, init);
  } catch {
    throw new Error("Can't reach the server. It may be waking up; try again in a few seconds.");
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const detail = typeof body?.detail === "string" ? body.detail : null;
    if (res.status === 429) throw new Error(detail ?? "The demo's daily question limit is reached.");
    if (res.status === 422) throw new Error("Please type a question (up to 500 characters).");
    throw new Error(detail ?? `Something went wrong (HTTP ${res.status}).`);
  }
  return res.json() as Promise<T>;
}

export const ask = (question: string) =>
  call<AskResponse>("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

export const getLeads = () => call<{ count: number; leads: Lead[] }>("/leads");

export const getThread = (threadId: string) => call<Thread>(`/threads/${encodeURIComponent(threadId)}`);
