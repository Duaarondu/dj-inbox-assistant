"use client";
// web/app/gmail/callback/page.tsx: Google sends the browser here after the Gmail consent screen.
// We hand the one-time code and the signed state to the API (with the sign-in token); the API checks the
// state belongs to this user + company, stores the encrypted token and starts the import. This page never
// sees a Gmail token. The code is single-use, so it's sent once (React may run effects twice in dev).
import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import { finishGmailConnect } from "@/lib/api";
import { accessToken } from "@/lib/supabase/client";

export default function GmailCallback() {
  const [message, setMessage] = useState("Connecting Gmail…");
  const [failed, setFailed] = useState(false);
  const sent = useRef(false);

  useEffect(() => {
    if (sent.current) return;
    sent.current = true;
    const params = new URLSearchParams(window.location.search);
    const code = params.get("code");
    const state = params.get("state");
    const denied = params.get("error");
    window.history.replaceState(null, "", "/gmail/callback");     // don't leave the code in the address bar

    const run = async () => {
      if (denied || !code || !state) {
        throw new Error(denied === "access_denied" ? "You didn't allow Gmail access." : "Google didn't finish the connection.");
      }
      const token = await accessToken();
      if (!token) {
        window.location.replace("/signin");
        return null;
      }
      return finishGmailConnect(token, code, state);
    };
    run()
      .then((r) => r && setMessage(`Connected ${r.gmail_address}. Importing your emails; this can take a few minutes.`))
      .catch((e: Error) => {
        setFailed(true);
        setMessage(e.message);
      });
  }, []);

  return (
    <main className="page narrow">
      <section className="card">
        <h1>Gmail</h1>
        <p className={failed ? "error" : "muted"}>{message}</p>
        <p className="muted">
          <Link href="/setup">Back to setup</Link>
        </p>
      </section>
    </main>
  );
}
