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

const SESSION_EXPIRED = "Your sign-in expired. Please sign in again.";

const bearer = (token: string) => ({ Authorization: `Bearer ${token}` });

// ask / getLeads / getThread serve the public demo without a token, and the signed-in user's OWN company with
// one (the server picks the company from the verified token).
const maybeBearer = (token?: string): Record<string, string> => (token ? bearer(token) : {});

export const ask = (question: string, token?: string) =>
  call<AskResponse>(
    "/ask",
    { method: "POST", headers: { "Content-Type": "application/json", ...maybeBearer(token) }, body: JSON.stringify({ question }) },
    { 401: SESSION_EXPIRED, 422: "Please type a question (up to 500 characters)." },
  );

export const getLeads = (token?: string) =>
  call<{ count: number; leads: Lead[] }>("/leads", { headers: maybeBearer(token) }, { 401: SESSION_EXPIRED });

export const getThread = (threadId: string, token?: string) =>
  call<Thread>(`/threads/${encodeURIComponent(threadId)}`, { headers: maybeBearer(token) }, { 401: SESSION_EXPIRED });

// Signed-in calls: the Supabase access token goes in the Authorization header on every request.
// The server takes the user (and company) from the verified token, never from the request body.
export type Me = { user_id: string; company_id: string | null };

export const getMe = (token: string) => call<Me>("/me", { headers: bearer(token) }, { 401: SESSION_EXPIRED });

export const createCompany = (token: string, name: string) =>
  call<{ company_id: string }>(
    "/companies",
    { method: "POST", headers: { "Content-Type": "application/json", ...bearer(token) }, body: JSON.stringify({ name }) },
    { 401: SESSION_EXPIRED, 409: "You already have a company.", 422: "Please enter a company name (up to 120 characters)." },
  );

// Connect Gmail (read-only). The API signs a `state` for this user + company; Google sends the browser back to
// /gmail/callback, which hands the code and state to the API. Tokens never pass through this site's code.
export type GmailStatus = {
  connected: boolean;
  gmail_address?: string;
  status?: "importing" | "ready" | "error" | "disconnected";
  error?: string | null;
  last_import_at?: string | null;
};

export const getGmailStatus = (token: string) =>
  call<GmailStatus>("/gmail/status", { headers: bearer(token) }, { 401: SESSION_EXPIRED });

export const startGmailConnect = (token: string, redirectUri: string) =>
  call<{ url: string }>(
    "/gmail/connect",
    { method: "POST", headers: { "Content-Type": "application/json", ...bearer(token) }, body: JSON.stringify({ redirect_uri: redirectUri }) },
    { 401: SESSION_EXPIRED },
  );

export const finishGmailConnect = (token: string, code: string, state: string) =>
  call<{ gmail_address: string; status: string }>(
    "/gmail/connect/finish",
    { method: "POST", headers: { "Content-Type": "application/json", ...bearer(token) }, body: JSON.stringify({ code, state }) },
    { 401: SESSION_EXPIRED },
  );

export const syncGmail = (token: string) =>
  call<{ status: string }>("/gmail/sync", { method: "POST", headers: bearer(token) }, { 401: SESSION_EXPIRED });
