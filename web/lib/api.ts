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

async function call<T>(path: string, init?: RequestInit, messages: Record<number, string> = {}): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, init);
  } catch {
    throw new Error("Can't reach the server. It may be waking up; try again in a few seconds.");
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const detail = typeof body?.detail === "string" ? body.detail : null;
    if (messages[res.status]) throw new Error(messages[res.status]);
    if (res.status === 429) throw new Error(detail ?? "The demo's daily question limit is reached.");
    throw new Error(detail ?? `Something went wrong (HTTP ${res.status}).`);
  }
  return res.json() as Promise<T>;
}

const bearer = (token: string) => ({ Authorization: `Bearer ${token}` });

export const ask = (question: string) =>
  call<AskResponse>(
    "/ask",
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question }) },
    { 422: "Please type a question (up to 500 characters)." },
  );

export const getLeads = () => call<{ count: number; leads: Lead[] }>("/leads");

export const getThread = (threadId: string) => call<Thread>(`/threads/${encodeURIComponent(threadId)}`);

// Signed-in calls: the Supabase access token goes in the Authorization header on every request.
// The server takes the user (and company) from the verified token, never from the request body.
export type Me = { user_id: string; company_id: string | null };

const SESSION_EXPIRED = "Your sign-in expired. Please sign in again.";

export const getMe = (token: string) => call<Me>("/me", { headers: bearer(token) }, { 401: SESSION_EXPIRED });

export const createCompany = (token: string, name: string) =>
  call<{ company_id: string }>(
    "/companies",
    { method: "POST", headers: { "Content-Type": "application/json", ...bearer(token) }, body: JSON.stringify({ name }) },
    { 401: SESSION_EXPIRED, 409: "You already have a company.", 422: "Please enter a company name (up to 120 characters)." },
  );
