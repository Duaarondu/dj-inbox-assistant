"use client";
// web/app/setup/page.tsx: after sign-in. Asks the API who we are (/me):
//   signed out -> /signin; no company -> "name your company"; has a company -> next step (connect Gmail).
// The user and company ids come from the token on the server, never from this page.
import Link from "next/link";
import { useEffect, useState } from "react";

import { createCompany, getGmailStatus, getMe, GmailStatus, Me, startGmailConnect } from "@/lib/api";
import { accessToken, supabase } from "@/lib/supabase/client";

export default function Setup() {
  const [me, setMe] = useState<Me | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [saving, setSaving] = useState(false);
  const [gmail, setGmail] = useState<GmailStatus | null>(null);
  const [connecting, setConnecting] = useState(false);

  async function load(): Promise<Me | null> {
    const token = await accessToken();
    if (!token) {
      window.location.replace("/signin");
      return null;
    }
    return getMe(token);
  }

  useEffect(() => {
    load()
      .then((m) => m && setMe(m))
      .catch((e: Error) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!me?.company_id) return;
    accessToken()
      .then((token) => (token ? getGmailStatus(token) : null))
      .then((g) => g && setGmail(g))
      .catch((e: Error) => setError(e.message));
  }, [me?.company_id]);

  async function connectGmail() {
    setConnecting(true);
    setError(null);
    try {
      const token = await accessToken();
      if (!token) return window.location.replace("/signin");
      const { url } = await startGmailConnect(token, `${window.location.origin}/gmail/callback`);
      window.location.assign(url);
    } catch (e) {
      setError((e as Error).message);
      setConnecting(false);
    }
  }

  async function submit() {
    const text = name.trim();
    if (!text || saving) return;
    setSaving(true);
    setError(null);
    try {
      const token = await accessToken();
      if (!token) return window.location.replace("/signin");
      await createCompany(token, text);
      setMe(await getMe(token));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  }

  async function signOut() {
    await supabase().auth.signOut();
    window.location.replace("/");
  }

  return (
    <main className="page narrow">
      <section className="card">
        {!me && !error && <p className="muted">Loading…</p>}

        {me && !me.company_id && (
          <>
            <h1>Name your company</h1>
            <p className="muted">This is the name your team will see. You can have one company per account.</p>
            <form
              className="askForm"
              onSubmit={(e) => {
                e.preventDefault();
                submit();
              }}
            >
              <input
                aria-label="Company name"
                value={name}
                maxLength={120}
                placeholder="e.g. Northshore Sound Co."
                onChange={(e) => setName(e.target.value)}
              />
              <button type="submit" disabled={saving || !name.trim()}>
                {saving ? "Saving…" : "Continue"}
              </button>
            </form>
          </>
        )}

        {me?.company_id && (
          <>
            <h1>You&apos;re set up</h1>
            {gmail?.connected ? (
              <p className="muted">
                Gmail: <strong>{gmail.gmail_address}</strong> ·{" "}
                {gmail.status === "importing" && "importing your emails…"}
                {gmail.status === "ready" && "up to date"}
                {gmail.status === "error" && `import failed: ${gmail.error ?? "unknown error"}`}
                {gmail.status === "disconnected" && "disconnected"}
              </p>
            ) : (
              <p className="muted">
                Next: connect Gmail so we can build your lead tracker. We only ask to <strong>read</strong> your
                email; spam, promotions and social mail are skipped.
              </p>
            )}
            {(!gmail?.connected || gmail.status === "error") && (
              <button className="primary" onClick={connectGmail} disabled={connecting || !gmail}>
                {connecting ? "Opening Google…" : gmail?.connected ? "Reconnect Gmail" : "Connect Gmail"}
              </button>
            )}
          </>
        )}

        {error && <p className="error">{error}</p>}
        <p className="muted">
          <Link href="/">Back to the demo</Link> ·{" "}
          <button className="link" onClick={signOut}>
            Sign out
          </button>
        </p>
      </section>
    </main>
  );
}
