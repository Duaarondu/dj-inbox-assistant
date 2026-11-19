"use client";
// web/app/inbox/page.tsx: the signed-in user's own inbox: tracker + ask box for THEIR company.
// signed out -> /signin; no company yet -> /setup. Which company is decided by the API from the token, never here.
import { useEffect, useState } from "react";

import InboxView from "@/components/InboxView";
import { getMe } from "@/lib/api";
import { accessToken } from "@/lib/supabase/client";

export default function Inbox() {
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      const token = await accessToken();
      if (!token) return window.location.replace("/signin");
      const me = await getMe(token);
      if (!me.company_id) return window.location.replace("/setup");
      setReady(true);
    })().catch((e: Error) => setError(e.message));
  }, []);

  if (error) return <main className="page"><p className="error">{error}</p></main>;
  if (!ready) return <main className="page"><p className="muted">Loading your inbox…</p></main>;
  return <InboxView getToken={accessToken} />;
}
