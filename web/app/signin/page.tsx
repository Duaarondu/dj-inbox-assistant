"use client";
// web/app/signin/page.tsx: "Continue with Google". Scopes stay openid/email/profile (Supabase's default);
// Gmail access is a separate consent later, on its own client.
import Link from "next/link";
import { useEffect, useState } from "react";

import { supabase } from "@/lib/supabase/client";

const ERRORS: Record<string, string> = {
  missing_code: "Google didn't send a sign-in code back. Please try again.",
  sign_in_failed: "Sign-in didn't complete. Please try again.",
};

export default function SignIn() {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const code = params.get("error");
    const why = params.get("why");
    supabase()
      .auth.getSession()
      .then(({ data }) => {
        if (data.session) window.location.replace("/setup");
        else if (code) setError((ERRORS[code] ?? "Sign-in failed. Please try again.") + (why ? ` (${why})` : ""));
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  async function google() {
    setBusy(true);
    setError(null);
    try {
      const { error } = await supabase().auth.signInWithOAuth({
        provider: "google",
        options: { redirectTo: `${window.location.origin}/auth/callback` },
      });
      if (error) throw error;
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  }

  return (
    <main className="page narrow">
      <section className="card">
        <h1>Sign in</h1>
        <p className="muted">
          Sign in to connect your own inbox. Signing in only shares your name and email; Gmail access is a
          separate step you choose later.
        </p>
        <button className="primary" onClick={google} disabled={busy}>
          {busy ? "Opening Google…" : "Continue with Google"}
        </button>
        {error && <p className="error">{error}</p>}
        <p className="muted">
          Just looking? <Link href="/">Try the demo</Link> without signing in.
        </p>
      </section>
    </main>
  );
}
